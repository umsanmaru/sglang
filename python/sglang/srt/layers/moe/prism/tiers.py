"""GPU 티어 — 계약 ④의 `GpuTier`와 두 구현.

계약 ①이 "hot과 warm의 계산 계약은 완전히 동일하다"고 못박은 것을 타입으로
표현한다. 두 구현은 스토어가 device냐 pinned냐 하나로만 갈리고 커널·출력
레이아웃·호출 규약이 같다 — `gemv_worklist_indexed`와 그 pinned 쌍둥이의
차이가 곧 이 두 클래스의 차이 전부다.

warm이 "전송"되던 시절에는 `stage → arena → GEMM`이라는 별도 사슬이 필요했다.
제자리 UVA 읽기가 되면서 그 사슬이 포인터 종류 하나로 축소됐고, 그래서
`DeviceArena`·stager·grouping이 함께 사라졌다.

**cold는 이 Protocol에 들어오지 않는다.** submit/sync 2-phase에 CPU 완료 대기가
있어 `run` 하나로 접히지 않고, 억지로 통합하면 "submit은 no-op, sync에서 다 함"
같은 거짓 구현이 생긴다 (계약 ④).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Protocol

import torch

from sglang.srt.layers.moe.prism.plan import Plan, Proj, Tier
from sglang.srt.layers.moe.prism.weights import PreparedWeights, TierShard

# 어느 GPU 티어가 sparse로 계산하는가 (2026-08-26 사용자 결정).
#
# WARM만이다. 건너뛴 W 로드가 warm에서는 그대로 **PCIe** 절약인데(warm 0.2의
# prefill이 hot과 무관하게 148 tok/s에 평평했던 것이 그 벽의 실측이다) hot은
# VRAM 상주라 아끼는 것이 VRAM 대역폭뿐이고, 점수 재료(a, c)를 VRAM에 얹는
# 비용과 점수 계산이 그 이득을 상쇄할 수 있다.
#
# **대가**: hot 행은 마스킹되지 않으므로 세 티어의 마스크 합집합이 full-K
# 마스크가 아니다 — 같은 행을 warm↔hot으로 옮기면 출력이 달라진다(계약 ⑤의
# plan 불변성이 sparse plan + hot에서 성립하지 않는다). 더 정확해지는 방향의
# 차이지만(hot은 전 정밀도) "같은 계산의 다른 배치"는 아니게 된다. hot도
# sparse로 돌리려면 이 집합에 Tier.HOT을 더하면 된다 — 커널·배관은 이미 양쪽을
# 지원한다.
SPARSE_TIERS = frozenset({Tier.WARM})


@dataclass(frozen=True)
class SparseSpec:
    """커널이 threshold를 스스로 계산하는 데 필요한 전부 — 전부 device 상주.

    라우터 가중은 여기 없다: 이 객체는 로드 타임에 한 번 만들어 영구 보관하는
    상수 묶음이고, 가중은 스텝마다 갈리므로 `run`의 인자로 온다.
    """

    a: torch.Tensor      # [Σₑ k[e]] fp32 — wn² (weight와 같은 오프셋)
    c: torch.Tensor      # [Σₑ k[e] / 2] fp32 — 인접열 내적
    thr: torch.Tensor    # [E, ng] fp32 — sparsity → threshold 곡선
    p: float
    lam: float
    pmax: float
    grid: float
    ng: int
    renorm_it: int


class GpuTier(Protocol):
    """한 (layer, proj)의 GPU 티어 하나.

    `run`은 current stream에 launch만 하고 즉시 반환한다 — sync point가 없다
    (계약 ④의 표). 출력은 호출자 소유이고, 티어는 로드 타임에 소유된 스토어·
    인덱스만 읽는다 (영구 할당 금지 규칙).

    `topk_weights`는 sparse 구현만 쓴다 (threshold가 라우터 가중의 함수다).
    dense 구현도 인자를 받는 이유는 호출부가 티어의 종류를 몰라야 하기
    때문이다 — 분기가 executor로 새면 티어 다형성이 무의미해진다.

    `masking`은 **스텝별** 결정이다 (계약 ①: sparsity는 decode 전용, prefill은
    dense). 티어 종류는 로드 타임에 고정되지만 마스킹 여부는 매 스텝 갈리므로,
    sparse 티어도 masking=False면 dense 커널을 부른다. cold가 prefill에서
    마스킹하지 않는데 warm이 하면 두 티어가 서로 다른 마스크로 계산한 부분합을
    더하게 된다.
    """

    shard: TierShard

    def run(self, x2d: torch.Tensor, topk_ids: torch.Tensor,
            topk_weights: torch.Tensor,
            out3d: torch.Tensor, out_col_off: int, *,
            x_row_is_pair: bool, masking: bool) -> None: ...


@dataclass(frozen=True)
class _IndexedTier:
    """dense 두 구현의 공통 본체 — 다른 것은 어떤 커널 래퍼냐뿐이다."""

    shard: TierShard

    def _fn(self):
        raise NotImplementedError

    def run(self, x2d, topk_ids, topk_weights, out3d, out_col_off, *,
            x_row_is_pair, masking) -> None:
        s = self.shard
        self._fn()(
            x2d, topk_ids, s.w_flat, s.row_off, s.k_index, out3d,
            out_col_off, x_row_is_pair, torch.cuda.current_stream(),
        )


@dataclass(frozen=True)
class _SparseIndexedTier:
    """sparse 두 구현의 공통 본체. dense와 갈리는 것은 `spec`이 더 붙는 것뿐이다."""

    shard: TierShard
    spec: SparseSpec

    def _fn(self):
        raise NotImplementedError

    def _dense_fn(self):
        raise NotImplementedError

    def run(self, x2d, topk_ids, topk_weights, out3d, out_col_off, *,
            x_row_is_pair, masking) -> None:
        s = self.shard
        stream = torch.cuda.current_stream()
        if not masking:
            self._dense_fn()(
                x2d, topk_ids, s.w_flat, s.row_off, s.k_index, out3d,
                out_col_off, x_row_is_pair, stream,
            )
            return
        # 커널이 fp32 라우터 가중을 요구한다 (kt의 slot_sparsity와 같은 타입).
        # 캐스팅을 스텝 경로에서 하면 할당이 생기므로 호출부의 dtype을 계약으로
        # 삼고 여기서 거절한다.
        if topk_weights.dtype is not torch.float32:
            raise TypeError(
                f"sparse tier requires fp32 topk_weights, got {topk_weights.dtype}"
            )
        self._fn()(
            x2d, topk_ids, topk_weights, s.w_flat, s.row_off, s.k_index, out3d,
            self.spec, out_col_off, x_row_is_pair, stream,
        )


@dataclass(frozen=True)
class ResidentTier(_IndexedTier):
    """HOT — 스토어가 VRAM 상주. GPU가 device 포인터로 읽는다."""

    def _fn(self):
        from sglang.jit_kernel.prism_gemv import gemv_worklist_indexed

        return gemv_worklist_indexed


@dataclass(frozen=True)
class PinnedDirectTier(_IndexedTier):
    """WARM — 스토어가 pinned host 상주. GPU가 UVA로 제자리 읽는다.

    배치 안에 같은 expert가 두 번 나오면 그만큼 PCIe를 두 번 탄다 (bs=1에서는
    발생 불가). 재사용이 이득이 되는 구간이 실측되면 그때 세 번째 구현
    (select → device 버퍼 → 재사용)이 이 Protocol에 붙는다.
    """

    def _fn(self):
        from sglang.jit_kernel.prism_gemv import gemv_worklist_indexed_pinned

        return gemv_worklist_indexed_pinned


@dataclass(frozen=True)
class SparseResidentTier(_SparseIndexedTier):
    """HOT의 sparse 변형 — 현재 SPARSE_TIERS가 고르지 않는다 (hot은 dense).

    구현을 남겨두는 이유: hot을 sparse로 돌리는 결정이 `SPARSE_TIERS` 한 줄이
    되도록 하는 것이 이 대칭의 목적이다. 죽은 코드가 아니라 정책의 다른 쪽 값이다.
    """

    def _fn(self):
        from sglang.jit_kernel.prism_gemv import gemv_worklist_indexed_sparse

        return gemv_worklist_indexed_sparse

    def _dense_fn(self):
        from sglang.jit_kernel.prism_gemv import gemv_worklist_indexed

        return gemv_worklist_indexed


@dataclass(frozen=True)
class SparsePinnedDirectTier(_SparseIndexedTier):
    """WARM의 sparse 변형 — 죽은 페어의 PCIe 로드를 발행하지 않는다."""

    def _fn(self):
        from sglang.jit_kernel.prism_gemv import (
            gemv_worklist_indexed_pinned_sparse,
        )

        return gemv_worklist_indexed_pinned_sparse

    def _dense_fn(self):
        from sglang.jit_kernel.prism_gemv import gemv_worklist_indexed_pinned

        return gemv_worklist_indexed_pinned


_DENSE_IMPL = {Tier.HOT: ResidentTier, Tier.WARM: PinnedDirectTier}
_SPARSE_IMPL = {Tier.HOT: SparseResidentTier, Tier.WARM: SparsePinnedDirectTier}


# ─── phase 단위 티어 (gate+up 융합) ────────────────────────────────────────
#
# executor가 도는 축은 (gateup, down)이고 "gate phase"는 존재하지 않는다. 그래서
# 이 계층의 단위도 phase다: `GateUpRunner` 하나가 한 티어의 gate와 up을 함께
# 들고, 가능하면 **커널 하나로** 발행한다.
#
# 왜 융합이 이득인가: 이 커널의 grid는 `(ceil(N/64), M×top_k)`뿐이라 bs=1에서
# 블록이 SM 수에 못 미친다 (35B gate: 96블록 / H100 114 SM). gate와 up을 한
# 커널에 넣으면 grid.z로 블록이 2배가 되고, 그것이 곧 성능이다 — 2026-08-26 실측
# hot 30.8 → 17.7 µs (1.74배), warm sparse 31.7 → 22.8 µs (1.39배). 출력 원소당
# 누산 순서가 불변이라 두 번 launch한 것과 **비트일치**한다 (계약 ⑤).
class GateUpRunner(Protocol):
    """한 티어의 gateup phase. `out3d`는 [M, k, 2·inter]이고 gate가 앞 절반,
    up이 뒤 절반이다 (계약 ②의 overwrite 의미론).

    `x_row_is_pair`가 인자에 없는 것은 phase가 그것을 결정하기 때문이다 —
    gateup의 x는 hidden [M, H]이므로 항상 False다.
    """

    writes_all: bool   # 두 절반을 모두 덮는가 (아니면 호출자가 0으로 채워야 한다)

    def run(self, x2d: torch.Tensor, topk_ids: torch.Tensor,
            topk_weights: torch.Tensor, out3d: torch.Tensor, inter: int, *,
            masking: bool) -> None: ...


@dataclass(frozen=True)
class _GateUpDense:
    """dense 두 구현의 공통 본체. 갈리는 것은 어느 래퍼냐뿐이다."""

    gate: TierShard
    up: TierShard
    writes_all: bool = True

    def _single(self):
        raise NotImplementedError

    def _fused(self):
        """융합 래퍼 — 없으면 None (그 조합의 진입점이 아직 없다는 뜻)."""
        return None

    def run(self, x2d, topk_ids, topk_weights, out3d, inter, *, masking) -> None:
        stream = torch.cuda.current_stream()
        fused = self._fused() if _worth_fusing(x2d) else None
        if fused is not None:
            fused(x2d, topk_ids,
                  self.gate.w_flat, self.gate.row_off, self.gate.k_index,
                  self.up.w_flat, self.up.row_off, self.up.k_index,
                  out3d, 0, inter, False, stream)
            return
        fn = self._single()
        fn(x2d, topk_ids, self.gate.w_flat, self.gate.row_off, self.gate.k_index,
           out3d, 0, False, stream)
        fn(x2d, topk_ids, self.up.w_flat, self.up.row_off, self.up.k_index,
           out3d, inter, False, stream)


@dataclass(frozen=True)
class _GateUpSparse:
    """sparse 두 구현의 공통 본체. dense와 갈리는 것은 spec이 더 붙는 것뿐이다."""

    gate: TierShard
    up: TierShard
    gate_spec: SparseSpec
    up_spec: SparseSpec
    writes_all: bool = True

    def _single(self):
        raise NotImplementedError

    def _single_dense(self):
        raise NotImplementedError

    def _fused(self):
        return None

    def _fused_dense(self):
        return None

    def run(self, x2d, topk_ids, topk_weights, out3d, inter, *, masking) -> None:
        stream = torch.cuda.current_stream()
        if not masking:
            # prefill은 dense다 (계약 ①). 그 조합의 융합 진입점은 아직 없어서
            # 2회 launch로 떨어지는데, prefill은 M이 커서 grid.y = M×top_k만으로
            # 블록이 수천 개라 융합의 이득(블록 배증)이 애초에 없다 — 없는 것을
            # 안 만든 것이고, 필요해지면 `_fused_dense`에 붙이면 된다.
            fused = self._fused_dense() if _worth_fusing(x2d) else None
            if fused is not None:
                fused(x2d, topk_ids,
                      self.gate.w_flat, self.gate.row_off, self.gate.k_index,
                      self.up.w_flat, self.up.row_off, self.up.k_index,
                      out3d, 0, inter, False, stream)
                return
            fn = self._single_dense()
            fn(x2d, topk_ids, self.gate.w_flat, self.gate.row_off,
               self.gate.k_index, out3d, 0, False, stream)
            fn(x2d, topk_ids, self.up.w_flat, self.up.row_off, self.up.k_index,
               out3d, inter, False, stream)
            return
        if topk_weights.dtype is not torch.float32:
            raise TypeError(
                f"sparse tier requires fp32 topk_weights, got {topk_weights.dtype}"
            )
        fused = self._fused() if _worth_fusing(x2d) else None
        if fused is not None:
            fused(x2d, topk_ids, topk_weights,
                  self.gate.w_flat, self.gate.row_off, self.gate.k_index,
                  self.up.w_flat, self.up.row_off, self.up.k_index,
                  out3d, self.gate_spec, self.up_spec, 0, inter, False, stream)
            return
        fn = self._single()
        fn(x2d, topk_ids, topk_weights, self.gate.w_flat, self.gate.row_off,
           self.gate.k_index, out3d, self.gate_spec, 0, False, stream)
        fn(x2d, topk_ids, topk_weights, self.up.w_flat, self.up.row_off,
           self.up.k_index, out3d, self.up_spec, inter, False, stream)


def _worth_fusing(x2d: torch.Tensor) -> bool:
    """decode(M==1)에서만 융합한다.

    융합의 이득은 블록 배증이고, 블록은 `ceil(N/64) × M × top_k`다. prefill은 M이
    커서 이미 블록이 넘치므로 이득이 없고(실측: 768블록에서 이미 1788 GB/s로
    대역폭 쪽에 붙어 있다), 그때 융합하면 리덕션·검증만 늘어난다. M을 여기서
    보는 이유는 커널이 아니라 **호출 형태**를 고르는 결정이기 때문이다.
    """
    return x2d.shape[0] == 1


@dataclass(frozen=True)
class ResidentGateUp(_GateUpDense):
    """HOT의 gateup — device 상주 W."""

    def _single(self):
        from sglang.jit_kernel.prism_gemv import gemv_worklist_indexed

        return gemv_worklist_indexed

    def _fused(self):
        from sglang.jit_kernel.prism_gemv import gemv_worklist_indexed_gateup

        return gemv_worklist_indexed_gateup


@dataclass(frozen=True)
class PinnedGateUp(_GateUpDense):
    """WARM의 gateup (dense) — sparsity 없는 plan의 warm이 여기로 온다.
    융합 진입점(pinned+dense)은 아직 없어 2회 launch다."""

    def _single(self):
        from sglang.jit_kernel.prism_gemv import gemv_worklist_indexed_pinned

        return gemv_worklist_indexed_pinned


@dataclass(frozen=True)
class SparseResidentGateUp(_GateUpSparse):
    """HOT의 sparse gateup — 현재 SPARSE_TIERS가 고르지 않는다 (hot은 dense)."""

    def _single(self):
        from sglang.jit_kernel.prism_gemv import gemv_worklist_indexed_sparse

        return gemv_worklist_indexed_sparse

    def _single_dense(self):
        from sglang.jit_kernel.prism_gemv import gemv_worklist_indexed

        return gemv_worklist_indexed

    def _fused_dense(self):
        from sglang.jit_kernel.prism_gemv import gemv_worklist_indexed_gateup

        return gemv_worklist_indexed_gateup


@dataclass(frozen=True)
class SparsePinnedGateUp(_GateUpSparse):
    """WARM의 sparse gateup — 실제 plan이 타는 경로."""

    def _single(self):
        from sglang.jit_kernel.prism_gemv import (
            gemv_worklist_indexed_pinned_sparse,
        )

        return gemv_worklist_indexed_pinned_sparse

    def _single_dense(self):
        from sglang.jit_kernel.prism_gemv import gemv_worklist_indexed_pinned

        return gemv_worklist_indexed_pinned

    def _fused(self):
        from sglang.jit_kernel.prism_gemv import (
            gemv_worklist_indexed_pinned_sparse_gateup,
        )

        return gemv_worklist_indexed_pinned_sparse_gateup


@dataclass(frozen=True)
class _GateUpSingle:
    """gate 또는 up **한쪽만** 이 티어에 있는 plan용 어댑터. 융합 대상이 없으므로
    기존 단일 proj 티어를 그대로 감싸고, 호출자가 나머지 절반을 0으로 채운다."""

    tier: GpuTier
    col_off: int
    writes_all: bool = False

    def run(self, x2d, topk_ids, topk_weights, out3d, inter, *, masking) -> None:
        self.tier.run(x2d, topk_ids, topk_weights, out3d, self.col_off,
                      x_row_is_pair=False, masking=masking)


@dataclass(frozen=True)
class LayerTiers:
    """한 레이어의 GPU 티어 — **phase 단위**다 (executor가 도는 축과 같다)."""

    gateup: Mapping[Tier, GateUpRunner]
    down: Mapping[Tier, GpuTier]


_GATEUP_DENSE = {Tier.HOT: ResidentGateUp, Tier.WARM: PinnedGateUp}
_GATEUP_SPARSE = {Tier.HOT: SparseResidentGateUp, Tier.WARM: SparsePinnedGateUp}


def _sparse_spec(
    shard: TierShard, thr: torch.Tensor, plan: Plan, layer_idx: int, proj: Proj,
    where: str,
) -> SparseSpec:
    """티어의 점수 재료를 device로 올려 커널 인자 묶음을 만든다 (로드 타임 1회).

    `a = wn²`를 여기서 물질화하는 이유: `CalibShard.wn_sq`는 접근마다 제곱하는
    property라 스텝 경로에 두면 매번 계산·할당한다. 커널이 읽는 것은 상수이므로
    한 번 만들어 device에 두는 것이 맞다.

    p/λ가 expert 0에서 읽히는 것은 kt와 같은 규약이다 (`cold_backend._build_config`)
    — plan 스키마는 (expert, proj)별 예산을 표현할 수 있지만 두 소비자 모두
    layer×proj 단위로만 굽는다. expert마다 다른 예산을 쓰려면 양쪽을 같이 고쳐야
    한다.
    """
    calib = shard.calib
    if calib is None:
        raise ValueError(
            f"{where}: plan has sparsity but this tier carries no calib shard "
            f"— prepare_layer_weights가 calib을 받지 못했다"
        )
    spec = plan.sparsity
    if spec is None:
        raise ValueError(f"{where}: sparse tier requested but plan has no sparsity")
    ep = plan.expert(layer_idx, 0).proj(proj)
    dev = shard.row_off.device
    return SparseSpec(
        a=calib.wn_sq.to(dev, torch.float32),
        c=calib.pair_dot.to(dev, torch.float32),
        thr=thr.to(dev, torch.float32),
        p=float(ep.sparsity_p), lam=float(ep.sparsity_lambda),
        pmax=spec.pmax, grid=spec.grid, ng=spec.ng, renorm_it=spec.renorm_it,
    )


def build_layer_tiers(
    prepared: PreparedWeights,
    plan: Plan,
    layer_idx: int,
) -> LayerTiers:
    """Stage 2 산출물 → phase 단위 티어. 없는 티어는 항목 자체가 없다.

    plan에 sparsity가 있으면 `SPARSE_TIERS`에 속한 티어만 sparse 구현을 받는다
    — 나머지는 dense 구현 그대로다 (계약 ⑤의 대가는 SPARSE_TIERS 주석 참조).

    gateup은 gate와 up을 **한 객체**로 묶는다: 융합이 가능한 조합에서 커널 하나로
    발행하기 위해서고, 그 판단은 그 객체 안에 있다 (executor는 모른다).
    """
    sparse_on = plan.sparsity is not None
    shards: dict = {}
    specs: dict = {}
    for proj in Proj:
        avail = (
            (Tier.HOT, None if prepared.hot is None else prepared.hot.band(proj)),
            (Tier.WARM, prepared.warm.band(proj)),
        )
        for tier, shard in avail:
            if shard is None:
                continue
            shards[(proj, tier)] = shard
            if sparse_on and tier in SPARSE_TIERS:
                where = f"layer {layer_idx} {proj.value} {tier.value}"
                specs[(proj, tier)] = _sparse_spec(
                    shard, prepared.thr[proj], plan, layer_idx, proj, where)

    def single(proj: Proj, tier: Tier) -> GpuTier:
        shard = shards[(proj, tier)]
        spec = specs.get((proj, tier))
        if spec is not None:
            return _SPARSE_IMPL[tier](shard, spec)
        return _DENSE_IMPL[tier](shard)

    gateup: dict = {}
    down: dict = {}
    for tier in (Tier.HOT, Tier.WARM):
        if (Proj.DOWN, tier) in shards:
            down[tier] = single(Proj.DOWN, tier)
        has_g = (Proj.GATE, tier) in shards
        has_u = (Proj.UP, tier) in shards
        if has_g and has_u:
            g, u = shards[(Proj.GATE, tier)], shards[(Proj.UP, tier)]
            sg, su = specs.get((Proj.GATE, tier)), specs.get((Proj.UP, tier))
            if (sg is None) != (su is None):
                # sparsity는 (proj, tier)가 아니라 티어 단위 결정이므로 한쪽만
                # sparse인 상태는 자산/plan의 모순이다 — 조용히 섞지 않는다.
                raise ValueError(
                    f"layer {layer_idx} {tier.value}: gate and up disagree on "
                    f"sparsity (gate={sg is not None}, up={su is not None})")
            gateup[tier] = (_GATEUP_SPARSE[tier](g, u, sg, su) if sg is not None
                            else _GATEUP_DENSE[tier](g, u))
        elif has_g:
            gateup[tier] = _GateUpSingle(single(Proj.GATE, tier), 0)
        elif has_u:
            gateup[tier] = _GateUpSingle(
                single(Proj.UP, tier), plan.dims.intermediate_size)
    return LayerTiers(gateup=gateup, down=down)

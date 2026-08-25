"""Stage 2: weight 절단·변환·배치. 산출물은 PreparedWeights (계약 ③).

소유권 (계약 ③):
- hot  → Python 소유 device 텐서 (VRAM 상주, 전송 없음)
- warm → Python 소유 pinned 텐서 (GPU가 UVA로 제자리 읽는다)
- cold → C++ MOE 객체 핸들. 단, cold backend(K3/S5)가 붙기 전까지는
  PendingColdTensors가 슬라이스본을 임시 소유한다 — backend 접속 시
  텐서를 넘기고 핸들로 대체되는 것이 P0의 로딩 흐름이다.

Stage 2 종료 후 full-K 텐서는 어디에도 존재하지 않는다: 이 모듈은 입력
w13/w2에 대한 참조를 보관하지 않으며, 호출자는 반환 즉시 원본을 놓는다.

레이아웃:
- ckpt 방향: w13 [E, 2·inter, hidden] (gate가 앞 절반, up이 뒤 절반 —
  fused_moe_triton/layer.py:431-432가 명시하는 sglang w13 관례),
  w2 [E, hidden, inter].
- 주의: quant method가 `load_up_proj_weight_first=True`(layer.py:434,
  trtllm cutlass 계열)를 세팅하면 w13 내부 순서가 뒤집힌다. 이 로더는
  기본 순서를 가정하므로, method 통합(S7)에서 해당 플래그가 False임을
  assert해야 한다 — 위반 시 gate/up이 조용히 뒤바뀐다.
- hot/warm store는 **flat + offset** [Σₑ k[e], N] (K-major) — `[E, k, N]`
  균일 적층이 expert마다 다른 k를 표현할 수 없어서다 (계약 ① 2026-08-25).
  두 티어가 같은 타입인 이유는 계산 계약이 같기 때문이고, 다른 것은 거처
  (device / pinned) 하나뿐이다. K-major는 no-transpose 정준 방향으로 로드
  시점에 고정한 것이다.
- 인덱스(`k_index`)와 스토어는 **같은 오프셋 테이블을 공유한다** — 둘 다
  expert당 k[e]개이므로 `row_off` 하나가 셋(weight·인덱스·점수 테이블)을
  서비스한다.
- cold 슬라이스는 ckpt 방향 [E, N, k_cold] 유지 — 소비자인 kt-kernel pack이
  기대하는 방향이고, 그쪽이 KIndex를 받는 K3까지 밴드 기하로 남는다.

sparsity(계약 ①, schema_version 2): `wn`/`pair_dot`은 K축이라 weight와 **같은
절단**을 받아 동행한다. `thr` 곡선은 절단과 무관한 per-(layer, expert)라
PreparedWeights.thr에 통째로 둔다. 셋 다 CPU fp32로 남긴다.

전환기 상태 (2026-08-25):
- 티어당 다중 밴드, expert 간 기하 불일치(가변 k) — **둘 다 지원된다.**
  인덱스 표현에는 애초에 그런 제약이 없고, 밴드 시절의 NotImplementedError
  두 개가 여기서 사라졌다.
- 남은 제약은 **소비자 쪽**이다: cold는 kt가 밴드 기하만 받아 균일·연속을
  요구하고(K3까지), calib gather도 그에 맞춰 밴드 절단으로 남는다. GPU 티어의
  밴드 경로(bmm/stager)도 `weights`/`k_offset` 접근 시 균일·연속을 요구하며,
  아니면 즉사한다 — 조용히 틀린 값을 주느니 죽는 편이 낫다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import torch

from sglang.srt.layers.moe.prism.calib import CalibBand, CalibTables
from sglang.srt.layers.moe.prism.index import (
    LayerIndex,
    TierIndex,
    from_bands,
    validate_layer,
)
from sglang.srt.layers.moe.prism.numa import alloc_pinned_on_node
from sglang.srt.layers.moe.prism.plan import (
    ExpertProjPlan,
    Plan,
    PlanError,
    Proj,
    Tier,
)


@dataclass
class TierShard:
    """GPU 티어(hot/warm) 하나의 스토어 — **flat + offset** (계약 ③ 2026-08-25).

    `[E, k, N]` 균일 적층은 expert마다 k가 다른 것을 표현할 수 없다. flat이
    소유자이고, 인덱스는 스토어와 **같은 오프셋 테이블을 공유한다** (둘 다
    expert당 k[e]개).

    hot과 warm이 같은 타입인 이유는 계약 ①이 둘의 계산 계약을 같다고 못박기
    때문이다 — 다른 것은 `w_flat`이 device냐 pinned냐 하나뿐이다.

    전환기 호환(`weights`/`k_offset`/`k_rows`): 기존 밴드 경로(bmm/stager)가
    아직 살아 있어 `[E, k, N]` 뷰와 밴드 스칼라를 노출한다. 균일·연속이 아닌
    plan에서는 **즉사한다** — 조용히 틀린 값을 주느니 죽는 편이 낫고, 그 시점이
    곧 밴드 경로를 지울 때다.
    """

    w_flat: torch.Tensor       # bf16 [Σₑ k[e], N] — hot=device / warm=pinned
    row_off: torch.Tensor      # int32 [E+1] (device — 커널이 읽는다)
    k_index: torch.Tensor      # uint16 [Σₑ k[e]] (device)
    contiguous: bool
    # 밴드 퇴화형일 때만 채워진다 (전환기 호환용, 로드 시 host에서 계산).
    uniform_k: Optional[int] = None
    band_start: Optional[int] = None
    calib: Optional[CalibBand] = None

    @property
    def num_experts(self) -> int:
        return int(self.row_off.numel()) - 1

    @property
    def total_rows(self) -> int:
        return int(self.w_flat.shape[0])

    # ── 전환기 호환 ──────────────────────────────────────────────────────
    @property
    def k_rows(self) -> int:
        if self.uniform_k is None:
            raise NotImplementedError(
                "k_rows: expert마다 행 수가 다르다 — 밴드 경로는 이 plan을 "
                "실행할 수 없다 (인덱스 경로를 쓸 것)"
            )
        return self.uniform_k

    @property
    def k_offset(self) -> int:
        if self.band_start is None:
            raise NotImplementedError(
                "k_offset: 연속 밴드가 아니다 — 밴드 경로는 이 plan을 실행할 "
                "수 없다 (인덱스 경로를 쓸 것)"
            )
        return self.band_start

    @property
    def weights(self) -> torch.Tensor:
        """[E, k, N] 뷰 — flat의 재해석이므로 복사가 없다."""
        return self.w_flat.view(self.num_experts, self.k_rows, self.w_flat.shape[1])

    @classmethod
    def from_band(cls, weights: torch.Tensor, k_offset: int = 0) -> "TierShard":
        """[E, k, N] 밴드 텐서에서 shard를 만든다 (전환기 픽스처용).

        flat은 뷰라 복사가 없다 — 밴드가 인덱스의 퇴화형이라는 사실의 가장
        짧은 표현이다."""
        E, k, N = weights.shape
        return cls(
            w_flat=weights.reshape(E * k, N),
            row_off=torch.arange(E + 1, dtype=torch.int32) * k,
            k_index=torch.arange(k_offset, k_offset + k).repeat(E).to(torch.uint16),
            contiguous=True,
            uniform_k=k,
            band_start=k_offset,
        )


@dataclass
class HotStore:
    gate: Optional[TierShard]
    up: Optional[TierShard]
    down: Optional[TierShard]

    def band(self, proj: Proj) -> Optional[TierShard]:
        return {Proj.GATE: self.gate, Proj.UP: self.up, Proj.DOWN: self.down}[proj]


@dataclass
class WarmStore:
    gate: Optional[TierShard]
    up: Optional[TierShard]
    down: Optional[TierShard]

    def band(self, proj: Proj) -> Optional[TierShard]:
        return {Proj.GATE: self.gate, Proj.UP: self.up, Proj.DOWN: self.down}[proj]


# 전환기 별칭 — 타입은 이미 하나로 합쳐졌고(계약 ①: hot/warm 계산 계약 동일),
# 이름만 밴드 경로가 지워질 때까지 남는다.
HotBand = TierShard
WarmBand = TierShard


@dataclass
class ColdBand:
    """한 proj의 cold 밴드: CPU bf16 [E, N, k_rows] (ckpt 방향)."""

    k_offset: int
    weights: torch.Tensor
    # sparsity 점수 재료 (이 밴드 구간만). plan.sparsity가 있으면 존재.
    calib: Optional[CalibBand] = None
    # 이 티어가 소유하는 K행 (계약 ③). cold **텐서 레이아웃**은 kt pack이
    # 정하므로 K3까지 ckpt 방향 [E, N, k]로 두고, 인덱스만 먼저 동행시킨다 —
    # kt가 KIndex를 받게 되는 시점(S7)의 입력이다.
    index: Optional[TierIndex] = None

    @property
    def k_rows(self) -> int:
        return self.weights.shape[2]


@dataclass
class PendingColdTensors:
    """cold backend 접속 전까지의 임시 소유자. backend가 붙으면 이 텐서들은
    C++로 pack되어 넘어가고 PreparedWeights.cold는 핸들로 대체된다."""

    gate: Optional[ColdBand]
    up: Optional[ColdBand]
    down: Optional[ColdBand]

    def band(self, proj: Proj) -> Optional[ColdBand]:
        return {Proj.GATE: self.gate, Proj.UP: self.up, Proj.DOWN: self.down}[proj]


@dataclass
class PreparedWeights:
    """Stage 2의 유일한 산출물이자 weight lifetime owner (계약 ③)."""

    hot: Optional[HotStore]  # VRAM 상주 밴드 (없으면 세 proj 모두 None)
    warm: WarmStore
    cold: PendingColdTensors  # S5 이후: ColdHandle
    # proj → [E, ng] fp32 threshold 곡선. dense plan이면 None.
    thr: Optional[Mapping[Proj, torch.Tensor]] = None


def _uniform_band(ti: TierIndex) -> tuple[Optional[int], Optional[int]]:
    """(uniform_k, band_start) — 밴드 퇴화형이면 채워지고 아니면 (None, None).

    전환기 호환 속성(`weights`/`k_offset`/`k_rows`)이 유효한지의 판정이다.
    host에서 한 번 계산해 두는 이유: 판정에 인덱스 값이 필요한데, 스토어가
    device로 간 뒤에는 그걸 읽는 것이 곧 동기화이기 때문이다.
    """
    E = ti.num_experts
    ks = {ti.k_rows(e) for e in range(E)}
    uniform = ks.pop() if len(ks) == 1 else None
    if uniform is None or not ti.contiguous or uniform == 0:
        return uniform, None
    starts = {int(ti.idx[int(ti.row_off[e])]) for e in range(E)}
    return uniform, (starts.pop() if len(starts) == 1 else None)


def _gather_flat(src: torch.Tensor, ti: TierIndex, uniform_k, band_start):
    """[E, N, K] 소스에서 이 티어의 flat 스토어 [Σₑ k[e], N]를 만든다 (CPU).

    밴드 퇴화형이면 한 번의 transpose+contiguous로 끝난다 — 기존 로더와 같은
    비용이라 현행 plan 40개의 로딩이 느려지지 않는다. 일반 경로는 expert 루프인데,
    배치 gather로 하려면 [E, N, k] 크기의 int64 인덱스를 물질화해야 해서(gate 기준
    6 GB) 루프가 오히려 싸다.
    """
    E, N, _ = src.shape
    if band_start is not None:
        return (
            src[:, :, band_start : band_start + uniform_k]
            .transpose(1, 2)
            .contiguous()
            .reshape(-1, N)
        )
    out = torch.empty(ti.total_rows, N, dtype=src.dtype)
    for e in range(E):
        o0, o1 = int(ti.row_off[e]), int(ti.row_off[e + 1])
        if o1 > o0:
            rows = ti.for_expert(e).to(torch.int64)
            out[o0:o1] = src[e].t().index_select(0, rows)
    return out


def _build_shard(
    src: torch.Tensor,
    ti: TierIndex,
    *,
    idx_device: torch.device,
    place,
    calib: Optional[CalibBand],
) -> TierShard:
    """티어 스토어 하나. `place`가 flat을 최종 거처로 옮긴다 (hot=device /
    warm=pinned). row_off·k_index는 **항상 커널이 읽는 device**로 간다."""
    uniform_k, band_start = _uniform_band(ti)
    flat = _gather_flat(src, ti, uniform_k, band_start)
    return TierShard(
        w_flat=place(flat),
        row_off=ti.row_off.to(idx_device),
        k_index=ti.idx.to(idx_device),
        contiguous=ti.contiguous,
        uniform_k=uniform_k,
        band_start=band_start,
        calib=calib,
    )


def _proj_source(w13: torch.Tensor, w2: torch.Tensor, inter: int, proj: Proj):
    """proj의 ckpt-방향 소스 [E, N, K] 뷰."""
    if proj is Proj.GATE:
        return w13[:, :inter, :]
    if proj is Proj.UP:
        return w13[:, inter:, :]
    return w2


def prepare_layer_weights(
    layer_idx: int,
    w13: torch.Tensor,
    w2: torch.Tensor,
    plan: Plan,
    *,
    calib: Optional[CalibTables] = None,
    pin_memory: bool = True,
    device: Optional[torch.device] = None,
    warm_node: Optional[int] = None,
) -> PreparedWeights:
    """한 레이어의 full weight를 Plan대로 절단·변환·배치한다.

    process_weights_after_loading 훅(rank 0)에서 호출된다. 반환 후 호출자는
    w13/w2 참조를 놓아야 한다 (full 텐서 소멸 계약).

    calib은 plan.sparsity의 존재와 짝이어야 한다 (all-or-nothing) — 한쪽만
    있으면 마스킹이 조용히 사라지거나 테이블이 버려지므로 즉사한다.

    pin_memory=False는 CUDA 없는 테스트용 탈출구다.
    warm_node는 warm pinned store가 상주해야 하는 NUMA 노드다 — hot의 device와
    같은 급의 **로더 입력**이고(계약 ③), 값을 정하는 것은 조립 지점의 몫이다
    (method.py가 gpu_numa_node로 GPU의 PCIe root 소켓을 읽어 넘긴다). None이면
    바인딩 없음 = 할당 스레드가 어디 떠 있었느냐에 달린 운.
    device는 HOT 밴드가 있을 때만 필요하다 (없으면 요구하지 않는다 — CPU
    전용 테스트가 hot 없는 plan으로 계속 돌 수 있어야 하므로).
    """
    dims = plan.dims
    if (plan.sparsity is None) != (calib is None):
        raise PlanError(
            f"layer {layer_idx}: plan.sparsity and calib must both be present "
            f"or both absent (sparsity={plan.sparsity is not None}, "
            f"calib={calib is not None})"
        )
    if calib is not None:
        calib.check_dims(dims, plan.sparsity)
    expected_w13 = (dims.num_experts, 2 * dims.intermediate_size, dims.hidden_size)
    expected_w2 = (dims.num_experts, dims.hidden_size, dims.intermediate_size)
    if tuple(w13.shape) != expected_w13 or tuple(w2.shape) != expected_w2:
        raise PlanError(
            f"layer {layer_idx}: weight shape mismatch vs plan dims: "
            f"w13 {tuple(w13.shape)} (expected {expected_w13}), "
            f"w2 {tuple(w2.shape)} (expected {expected_w2}) — "
            f"plan이 다른 모델에 적용되고 있을 가능성"
        )

    # 티어 멤버십을 인덱스로 확정하고 **순열·페어를 검증한다** (계약 ①).
    # 밴드 검증이 plan.py에서 사라진 자리를 여기가 메운다 — 로드마다 돈다.
    layer_index = from_bands(plan, layer_idx)
    validate_layer(layer_index, dims, layer_idx)

    hot_shards: dict[Proj, Optional[TierShard]] = {}
    warm_shards: dict[Proj, Optional[TierShard]] = {}
    cold_bands: dict[Proj, Optional[ColdBand]] = {}

    if any(layer_index.get(p, Tier.HOT) is not None for p in Proj) and device is None:
        raise PlanError(
            f"layer {layer_idx}: plan has HOT rows but no device was given — "
            f"hot store는 VRAM 상주라 배치 device가 로더의 입력이어야 한다"
        )
    idx_device = device if device is not None else torch.device("cpu")

    for proj in Proj:
        pp = plan.expert(layer_idx, 0).proj(proj)
        where = f"layer {layer_idx} {proj.value}"
        src = _proj_source(w13, w2, dims.intermediate_size, proj)  # [E, N, K]

        def band_calib(ti, tier_name):
            """전환기: calib은 아직 밴드 절단이다 (cold만 소비하고, 그 소비자인
            kt는 K3까지 밴드 기하를 쓴다). 인덱스 gather는 S7에서 붙는다."""
            if calib is None:
                return None
            k0, k1 = _uniform_band(ti)
            if k1 is None:
                raise NotImplementedError(
                    f"{where} {tier_name}: sparsity + 비밴드 인덱스는 아직 "
                    f"미지원 (calib gather는 kt가 KIndex를 받을 때 붙는다)"
                )
            return calib.slice_band(layer_idx, proj, k1, k1 + k0, f"{where} {tier_name}")

        hot_ti = layer_index.get(proj, Tier.HOT)
        hot_shards[proj] = None if hot_ti is None else _build_shard(
            src, hot_ti, idx_device=idx_device,
            place=lambda t: t.to(device, non_blocking=False),
            calib=band_calib(hot_ti, "hot"),
        )

        warm_ti = layer_index.get(proj, Tier.WARM)
        def place_warm(t, where=where):
            if not pin_memory:
                return t.contiguous()
            store = alloc_pinned_on_node(
                tuple(t.shape), t.dtype, warm_node, f"{where} warm store")
            store.copy_(t)
            return store
        warm_shards[proj] = None if warm_ti is None else _build_shard(
            src, warm_ti, idx_device=idx_device, place=place_warm,
            calib=band_calib(warm_ti, "warm"),
        )

        cold_ti = layer_index.get(proj, Tier.COLD)
        if cold_ti is None:
            cold_bands[proj] = None
        else:
            # cold 텐서 레이아웃은 kt pack이 정한다 — K3까지 ckpt 방향
            # [E, N, k] 유지. 인덱스는 동행만 시킨다 (S7의 입력).
            k0, k1 = _uniform_band(cold_ti)
            if k1 is None:
                raise NotImplementedError(
                    f"{where} cold: kt가 아직 밴드 기하만 받는다 (K3 이후 해제)"
                )
            cold_bands[proj] = ColdBand(
                k_offset=k1,
                weights=src[:, :, k1 : k1 + k0].contiguous(),
                calib=band_calib(cold_ti, "cold"),
                index=cold_ti,
            )

    return PreparedWeights(
        hot=HotStore(
            gate=hot_shards[Proj.GATE], up=hot_shards[Proj.UP],
            down=hot_shards[Proj.DOWN],
        ),
        warm=WarmStore(
            gate=warm_shards[Proj.GATE], up=warm_shards[Proj.UP],
            down=warm_shards[Proj.DOWN],
        ),
        cold=PendingColdTensors(
            gate=cold_bands[Proj.GATE], up=cold_bands[Proj.UP], down=cold_bands[Proj.DOWN]
        ),
        thr=(
            None if calib is None
            else {proj: calib.thr(layer_idx, proj) for proj in Proj}
        ),
    )

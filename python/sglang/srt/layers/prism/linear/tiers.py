"""dense GPU 티어 — 계약 ④의 `GpuTier`를 dense 좌표로.

MoE `moe/prism/tiers.py`와 같은 것을 한다: hot과 warm의 차이는 스토어가 device냐
pinned냐 **하나뿐**이고(계약 ①), 이 모듈은 그 차이를 `pinned` 플래그 하나로만
안다. cold는 여기 없다 — submit/sync 2-phase에 CPU 완료 대기가 있어 `run` 하나로
접히지 않는다 (계약 ④).

**커널은 MoE 것을 E=1로 퇴화시켜 쓴다.** dense에는 expert가 없으므로:

    topk_ids   = [M, 1] int64, 전부 0        (expert 0 하나)
    row_off    = [2] int32 = [0, k_tier]     (expert 경계 테이블의 퇴화형)
    out3d      = [M, 1, N_total]             (pair 축이 길이 1)
    x_row_is_pair = False                    (k=1이라 pair 인덱스 = 토큰 인덱스)

이 퇴화 인자들은 **스토어가 아니라 여기**에 산다. `LinearTierShard`는 dense의
사실만 말하고(`row_off` 없음), 커널 ABI와의 번역은 이 어댑터가 한 곳에서 한다 —
`LinearColdShard.real_rows`를 스칼라로 둔 것과 같은 선택이다.

**N축 조각과 출력 열.** 분할된 linear(`gate_up_proj`)는 조각마다 티어가 다를 수
있다 — gate는 hot인데 up은 warm일 수 있다. 조각은 자기 열 범위
`[n_start, n_end)`에만 쓰므로, 한 티어의 버퍼에 **모든 조각이 쓰지는 않는다**.
그래서 executor가 그 티어를 안 쓰는 조각이 있으면 버퍼를 0으로 할당한다
(MoE `_run_gateup`의 `writes_all`과 같은 이유).

**커널이 M에 따라 둘이다.** decode/소배치는 pair-native worklist GEMV, 큰 M은
**grouped GEMM**이다. worklist는 pair마다 W를 다시 읽는데 dense는 E=1이라 중복도가
곧 M이고(MoE는 M·k/E), warm은 그 재읽기가 전부 PCIe다 — Qwen3.8 M=2048에서 forward
한 번에 89.7 TB, 약 30분이었다 (2026-09-01 실측). grouped는 W를 **한 번만** 읽는다.
둘은 같은 스토어·같은 출력 레이아웃을 쓰므로 커널 선택은 호출 형태의 결정이고
계약이 아니다.

**융합은 아직 없다.** MoE는 `gemv_gateup`으로 gate·up을 한 launch에 낸다. dense도
같은 구조가 되었으니 붙일 수 있지만, 두 조각이 같은 티어에 있을 때만 유효하고
launch 하나를 줄이는 이득이라 미룬다 (TODO).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.layers.prism.geometry import Tier
from sglang.srt.layers.prism.linear.weights import LinearTierShard, PreparedPart

# 어느 GPU 티어가 sparse로 계산하는가 — MoE와 같은 선택 (`moe/prism/tiers.py`의
# SPARSE_TIERS). WARM만이다: 건너뛴 W 로드가 warm에서는 그대로 **PCIe** 절약인데
# hot은 VRAM 상주라 아끼는 것이 대역폭뿐이고, 점수 재료(a, c)를 VRAM에 얹는 비용이
# 그 이득을 상쇄할 수 있다. cold는 이 집합 밖에서 kt가 항상 마스킹한다.
#
# **대가**: hot 행은 마스킹되지 않으므로 세 티어 마스크의 합집합이 full-K 마스크가
# 아니다 — 같은 행을 warm↔hot으로 옮기면 출력이 달라진다(계약 ⑤의 plan 불변성이
# sparse plan + hot에서 성립하지 않는다).
SPARSE_TIERS = frozenset({Tier.WARM})


@dataclass(frozen=True)
class LinearGpuTier:
    """한 (조각, 티어)의 GPU 실행 어댑터.

    `spec`이 있으면 sparse 커널을 부를 수 있다. 실제로 부를지는 **스텝마다**
    갈린다 (계약 ①: sparsity는 decode 전용) — sparse 티어도 `masking=False`면
    dense 커널을 부른다. cold가 prefill에서 마스킹하지 않는데 warm이 하면 두
    티어가 서로 다른 마스크로 계산한 부분합을 더하게 된다.
    """

    shard: LinearTierShard
    row_off: torch.Tensor        # [2] int32 device — expert 경계의 E=1 퇴화형
    out_col: int                 # out3d의 열 오프셋 = 조각의 n_start
    pinned: bool
    spec: object = None          # SparseSpec | None (sparse 배선 시 채워진다)

    @classmethod
    def build(cls, shard: LinearTierShard, out_col: int, pinned: bool,
              spec=None) -> "LinearGpuTier":
        row_off = torch.tensor([0, shard.k_rows], dtype=torch.int32,
                               device=shard.k_index.device)
        return cls(shard=shard, row_off=row_off, out_col=out_col, pinned=pinned,
                   spec=spec)

    def run(self, x2d: torch.Tensor, topk_ids: torch.Tensor,
            topk_weights: torch.Tensor, out3d: torch.Tensor, *,
            masking: bool, grouping=None) -> None:
        """current stream에 launch만 하고 즉시 반환한다 — sync point가 없다.

        출력은 호출자 소유이고, 티어는 로드 타임에 소유된 스토어·인덱스만 읽는다
        (영구 할당 금지 규칙).

        `grouping`이 주어지면 grouped GEMM(prefill 형태)을 탄다. sparsity는 decode
        전용이므로(계약 ①) grouped와 masking은 동시에 오지 않는다 — 호출자가 보장한다.
        """
        s = self.shard
        stream = torch.cuda.current_stream()
        if grouping is not None:
            if masking:
                raise ValueError(
                    "grouped GEMM에 masking을 걸 수 없다 (sparsity는 decode 전용) — "
                    "cold가 prefill에서 마스킹하지 않는데 warm이 하면 두 티어가 서로 "
                    "다른 마스크로 계산한 부분합을 더하게 된다"
                )
            s.fmt.grouped(pinned=self.pinned)(
                x2d, grouping, *s.store_args(), self.row_off, s.k_index,
                out3d, self.out_col, False, stream,
            )
            return
        if not masking or self.spec is None:
            s.fmt.gemv(pinned=self.pinned, sparse=False)(
                x2d, topk_ids, *s.store_args(), self.row_off, s.k_index,
                out3d, self.out_col, False, stream,
            )
            return
        # 커널이 fp32 라우터 가중을 요구한다 (kt의 slot_sparsity와 같은 타입).
        # dense는 그 자리에 1.0을 넣는다 — k=1에서 s = clip(p)로 퇴화한다.
        if topk_weights.dtype is not torch.float32:
            raise TypeError(
                f"sparse tier requires fp32 weights, got {topk_weights.dtype}"
            )
        s.fmt.gemv(pinned=self.pinned, sparse=True)(
            x2d, topk_ids, topk_weights, *s.store_args(), self.row_off, s.k_index,
            out3d, self.spec, self.out_col, False, stream,
        )


def build_part_specs(part: PreparedPart, sparsity) -> dict:
    """조각이 나르는 재료 → `{Tier: SparseSpec}`. 마스킹하지 않으면 빈 dict.

    `SparseSpec`은 MoE 것을 그대로 쓴다 — 커널이 같은 필드를 읽으므로 타입을 둘로
    두면 정의점이 갈린다 (`linear/executor.py`가 `moe.prism.grouping`을 가져오는
    것과 같은 선례). 지연 import: `moe.prism.tiers`는 무겁고 이 경로에서만 필요하다.

    `thr`은 `[1, ng]`로 복원한다 — 커널이 `[E, ng]`를 기대하고 dense는 E=1이다.
    패딩 행은 a=0, c=0이라 energy 0이고 weight도 0이라 어느 쪽이든 무해하다.
    """
    from sglang.srt.layers.moe.prism.tiers import SparseSpec

    if sparsity is None or part.thr is None:
        return {}
    out = {}
    for tier in SPARSE_TIERS:
        shard = part.tier(tier)
        if shard is None or getattr(shard, "calib", None) is None:
            continue
        dev = shard.k_index.device
        out[tier] = SparseSpec(
            a=shard.calib.wn_sq.to(dev, torch.float32),
            c=shard.calib.pair_dot.to(dev, torch.float32),
            thr=part.thr.to(dev, torch.float32).unsqueeze(0),
            p=float(part.sparsity_p), lam=float(part.sparsity_lambda),
            pmax=sparsity.pmax, grid=sparsity.grid, ng=sparsity.ng,
            renorm_it=sparsity.renorm_it,
        )
    return out


def build_part_tiers(part: PreparedPart, specs=None, sparsity=None) -> dict:
    """한 조각의 GPU 티어들 (`{Tier: LinearGpuTier}`). cold는 포함하지 않는다.

    `specs`는 `{Tier: SparseSpec}` — 명시로 주면 그것을 쓰고(테스트용 우회), 없으면
    `sparsity`로 조각에서 조립한다. 둘 다 없으면 dense로만 돈다. SPARSE_TIERS 밖의
    티어에는 spec을 달지 않는다 (hot에 spec을 달면 조용히 마스킹된다).
    """
    specs = specs if specs else build_part_specs(part, sparsity)
    out = {}
    for tier, pinned in ((Tier.HOT, False), (Tier.WARM, True)):
        shard = part.tier(tier)
        if shard is None:
            continue
        spec = specs.get(tier) if tier in SPARSE_TIERS else None
        out[tier] = LinearGpuTier.build(shard, part.n_start, pinned, spec)
    return out

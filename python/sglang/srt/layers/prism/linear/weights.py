"""dense Stage 2: weight 절단·배치. 산출물은 `PreparedLinear` (계약 ③).

`moe/prism/weights.py`의 dense 대응물. **expert 루프가 사라진 것이 차이의 전부**는
아니고, 그것이 지우는 것들이 차이다:

| | MoE | dense |
|---|---|---|
| 소스 | `w13 [E, 2·inter, H]` + `w2 [E, H, inter]` | `weight [N, K]` 하나 |
| 산출 | proj 3개 × 티어 3개 = shard 9개 | 티어 3개 |
| 스토어 | flat + `row_off[E+1]` (expert마다 k가 다르다) | flat 하나 (`row_off`가 필요 없다) |
| gate/up 순서 가정 | w13 내부 절반 순서 (뒤집히면 조용히 오답) | 없음 |

소유권은 그대로다 (계약 ③):

  hot  → Python 소유 device 텐서 (VRAM 상주, 전송 없음)
  warm → Python 소유 pinned 텐서 (GPU가 UVA로 제자리 읽는다)
  cold → 임시 소유. cold backend에 주입되면 C++로 넘어가고 여기 참조는 소멸한다.

이 모듈은 입력 `weight`에 대한 참조를 보관하지 않는다 — 호출자는 반환 즉시 원본을
놓고, 그 시점에 full-K 텐서가 프로세스에서 사라진다.

**방향 규약.** sglang linear의 `weight`는 `[N, K]`다 (`y = x @ W.t()`;
`UnquantizedLinearMethod.create_weights` 참조). 티어별 절단은 그 K축의 부분집합이다:

  hot/warm → `[k_tier, N]` **K-major**. transpose 없는 정준 방향으로 로드 시점에
             고정한다 (MoE와 같은 이유·같은 방향이라 GPU 커널이 공유된다).
  cold     → `[N, k_pad]` **ckpt 방향 유지**. 소비자인 kt pack(`from_mat`)이 그
             방향의 `[N, k]`를 읽는다. 패딩은 커널 타일 배수까지 0으로 채운다
             (`kernels.cold_pack_tile_rows` — 커널 키가 함의하는 값).

**포맷: bf16과 blockwise fp8** (`formats.py`). 이 모듈은 둘의 차이를 모른다 — 아는
것은 "포맷이 K를 어디서 자를 수 있는지 말해준다"뿐이고(`k_align`, `check_rows`),
gather와 패딩의 형식별 구현은 포맷이 갖는다. dense 대상에 mxfp4 체크포인트가 없다는
사용자 확인(2026-08-31)에 따라 그 구현은 두지 않았다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from sglang.srt.layers.prism.geometry import PAIR_GROUP, PlanError, Tier
from sglang.srt.layers.prism.kernels import (
    cold_pack_tile_rows,
    resolve_cpu_kernel,
    resolve_gpu_kernel,
)
from sglang.srt.layers.prism.linear.calib import LinearCalibShard, LinearCalibTables
from sglang.srt.layers.prism.linear.formats import BF16, LinearStoreFormat, store_format
from sglang.srt.layers.prism.linear.plan import (
    LinearPlan,
    ProjPart,
    ProjPlan,
    check_dims,
)
from sglang.srt.layers.prism.numa import alloc_pinned_on_node
from sglang.srt.layers.prism.store import IDX_DTYPE, MAX_K, is_row_run


@dataclass
class LinearTierShard:
    """GPU 티어(hot/warm) 하나의 스토어 — `[k_tier, N]` K-major.

    hot과 warm이 같은 타입인 이유는 계약 ①이 둘의 계산 계약을 같다고 못박기
    때문이다 — 다른 것은 `w_flat`이 device냐 pinned냐 하나뿐이다.

    MoE `TierShard`와 달리 `row_off`가 없다. 그 테이블은 "expert마다 k가 다르다"를
    표현하려고 있었고, dense에는 expert가 없다.
    """

    w_flat: torch.Tensor        # [k_tier, N] — bf16, 또는 fp8 코드 u8. hot=device / warm=pinned
    k_index: torch.Tensor       # uint16 [k_tier] (device — 커널이 읽는다)
    contiguous: bool            # k_index가 단위 stride 구간 하나인가
    # 스토어 포맷 (formats.py). 절단 단위와 배율의 유무를 이 객체가 정한다.
    fmt: LinearStoreFormat = BF16
    # fp8만: fp32 scale_inv [k_tier/128, N/128] — w_flat과 같은 거처.
    s_flat: Optional[torch.Tensor] = None
    # contiguous일 때 그 구간의 시작 행. 커널이 gather를 포인터 오프셋으로 대체할
    # 수 있는지의 판정이라 host에서 로드 시 1회 계산해 둔다 — 스토어가 device로 간
    # 뒤에 인덱스 값을 읽는 것은 곧 동기화다.
    k_start: Optional[int] = None
    # sparse 티어의 점수 재료 — `k_index`와 **같은 순서**로 gather된 것.
    # plan에 sparsity가 없거나 이 조각이 `sparse: false`면 None이고, 그때
    # `tiers.py`가 dense 커널을 부른다 (spec=None).
    calib: Optional[LinearCalibShard] = None

    @property
    def k_rows(self) -> int:
        return int(self.w_flat.shape[0])

    @property
    def n(self) -> int:
        return int(self.w_flat.shape[1])

    def store_args(self) -> tuple:
        """커널 래퍼에 넘기는 스토어 텐서들 — 포맷이 정한다."""
        return self.fmt.store_args(self)


@dataclass
class LinearColdShard:
    """cold 스토어 — `[N, k_pad]` ckpt 방향, 커널 타일 배수까지 0 패딩.

    `real_rows`가 패딩 전 행 수다. 0 weight는 dense 계산에서 무해하지만 kt가
    tail 처리를 알아야 하므로 같이 나른다. MoE `ColdShard`에서는 `[E]` 텐서였고
    여기서는 스칼라다 — "E=1 퇴화 MOEConfig"로의 번역은 cold backend 어댑터가
    한 곳에서 하고, 이 구조체는 dense의 사실만 말한다.
    """

    w_flat: torch.Tensor        # [N, k_pad] — bf16, 또는 fp8 코드 u8 (contiguous)
    k_index: torch.Tensor       # uint16 [k_pad] — 패딩 항목은 0을 가리킨다
    real_rows: int              # 패딩 전 행 수
    contiguous: bool
    fmt: LinearStoreFormat = BF16
    # fp8만: fp32 scale_inv [N/128, k_pad/128] — 패딩 블록은 1.0.
    s_flat: Optional[torch.Tensor] = None
    # sparse 점수 재료. `k_index`(패딩 포함)와 같은 순서이고 패딩 구간은 0이다 —
    # weight도 0이라 수치 기여가 없고 kt가 tail 비트를 끈다 (`real_rows`).
    calib: Optional[LinearCalibShard] = None

    @property
    def k_pad(self) -> int:
        return int(self.w_flat.shape[1])

    @property
    def n(self) -> int:
        return int(self.w_flat.shape[0])


@dataclass
class PreparedPart:
    """한 N축 조각의 세 티어. 분할이 없으면 조각이 하나(`name=None`)다.

    셋 다 None인 경우는 없다 — plan의 밴드가 [0, k)를 덮으므로(`validate_static`)
    적어도 하나는 행을 갖는다.
    """

    name: Optional[str]      # 조각 이름 — calib 테이블 이름이기도 하다
    n_start: int
    n_end: int
    hot: Optional[LinearTierShard]
    warm: Optional[LinearTierShard]
    cold: Optional[LinearColdShard]
    # `[ng]` fp32 threshold 곡선 — 밴드와 무관해 티어별이 아니라 조각별이다.
    # 소비자가 `[E, ng]`로 복원한다 (GPU는 E=1, kt cold는 unit 축으로 스택).
    thr: Optional[torch.Tensor] = None
    # 이 조각의 예산. plan에서 온 그대로 나른다 — 티어마다 같고, 소비자(GPU spec /
    # kt config)가 각자의 어휘로 옮긴다.
    sparsity_p: Optional[float] = None
    sparsity_lambda: Optional[float] = None

    @property
    def n(self) -> int:
        return self.n_end - self.n_start

    def tier(self, t: Tier) -> Optional[object]:
        return {Tier.HOT: self.hot, Tier.WARM: self.warm, Tier.COLD: self.cold}[t]

    def rows(self, t: Tier) -> int:
        s = self.tier(t)
        if s is None:
            return 0
        return s.real_rows if isinstance(s, LinearColdShard) else s.k_rows


@dataclass
class PreparedLinear:
    """dense Stage 2의 유일한 산출물이자 weight lifetime owner (계약 ③).

    `parts`는 weight 행 순서대로다 (gate가 앞). `mlp.gate_up_proj`처럼 sparsity가
    두 절반을 따로 캘리브하는 linear는 조각이 둘이고, 나머지는 하나다 — 그 이유는
    plan.py docstring 참조.
    """

    name: str
    layer_idx: int
    k: int
    n: int
    parts: Tuple[PreparedPart, ...]
    fmt: LinearStoreFormat = BF16

    @property
    def split(self) -> bool:
        return len(self.parts) > 1

    @property
    def sole(self) -> "PreparedPart":
        """분할이 없는 linear의 유일한 조각.

        분할된 것에 부르면 즉사한다 — "조각이 하나겠지"라는 가정이 gate만 보고 up을
        빠뜨리는 형태로 조용히 틀리는 것을 막는다.
        """
        if len(self.parts) != 1:
            raise ValueError(
                f"{self.name}: split into {len(self.parts)} parts "
                f"({[p.name for p in self.parts]}) — pick one with .part(name)"
            )
        return self.parts[0]

    def part(self, name: Optional[str]) -> PreparedPart:
        for p in self.parts:
            if p.name == name:
                return p
        raise KeyError(f"{self.name}: no part {name!r}")

    def rows(self, t: Tier) -> int:
        """조각들의 K행 합. 분할이 있으면 조각마다 밴딩이 다를 수 있다."""
        return sum(p.rows(t) for p in self.parts)


# ---------------------------------------------------------------------------
# 밴드 → K행
# ---------------------------------------------------------------------------


def tier_rows(part: ProjPart, tier: Tier) -> List[int]:
    """이 티어가 소유하는 K행 번호 (밴드 순서대로 이어 붙인다).

    티어당 밴드가 여러 개여도 된다 — 인덱스 표현에 "티어당 단일 밴드" 제약은 없다.
    """
    rows: List[int] = []
    for b in part.bands:
        if b.tier is tier:
            rows.extend(range(b.start, b.end))
    return rows


def _index(rows: List[int], where: str) -> Optional[Tuple[torch.Tensor, bool, Optional[int]]]:
    """행 목록 → (uint16 인덱스, contiguous, k_start). 비면 None.

    None을 돌려주는 이유는 MoE와 같다: "이 티어는 여기 없다"를 길이 0 텐서가 아니라
    **부재**로 표현해야 스토어(`PreparedLinear.hot = None`)와 어휘가 맞는다.
    """
    if not rows:
        return None
    t = torch.as_tensor(rows, dtype=torch.int64)
    hi = int(t.max())
    if hi > MAX_K:
        raise PlanError(
            f"{where}: K row index {hi} exceeds {MAX_K} — "
            f"uint16 인덱스로 표현 불가 (dtype을 올려야 한다)"
        )
    # 페어 검증 (계약 ① 정렬 규칙). plan의 밴드 정렬(ROW_GROUP)이 이미 보장하지만,
    # 로드마다 다시 본다 — 쪼개진 페어는 VNNI skip 단위를 깨고 sparse 점수를
    # 재구성 불가능하게 만드는데, 어느 것도 예외를 내지 않는다.
    n = int(t.numel())
    if n % PAIR_GROUP:
        raise PlanError(
            f"{where}: row count {n} is not a multiple of PAIR_GROUP={PAIR_GROUP}"
        )
    pairs = t.view(-1, PAIR_GROUP)
    if not bool(((pairs[:, 0] // PAIR_GROUP) == (pairs[:, 1] // PAIR_GROUP)).all()):
        bad = int(
            ((pairs[:, 0] // PAIR_GROUP) != (pairs[:, 1] // PAIR_GROUP)).nonzero()[0][0]
        )
        raise PlanError(
            f"{where}: masking pair split at position {bad * PAIR_GROUP} "
            f"({int(pairs[bad, 0])}, {int(pairs[bad, 1])}) — 한 페어의 두 채널은 "
            f"같은 티어에서 인접해야 한다"
        )
    run = is_row_run(t)
    return t.to(IDX_DTYPE), run, (int(t[0]) if run else None)


# ---------------------------------------------------------------------------
# 절단
# ---------------------------------------------------------------------------


def prepare_linear_weights(
    layer_idx: int,
    name: str,
    weight: torch.Tensor,
    plan: LinearPlan,
    *,
    scale: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    warm_node: Optional[int] = None,
    pin_memory: bool = True,
    calib: Optional[LinearCalibTables] = None,
) -> PreparedLinear:
    """한 linear의 full weight를 Plan대로 절단·배치한다.

    `create_weights` 이후, 로더가 채운 weight에 대해 호출된다. 반환 후 호출자는
    `weight`/`scale` 참조를 놓아야 한다 (full 텐서 소멸 계약).

    weight는 **CPU** `[N, K]`여야 한다. sglang 로더가 파라미터를 CUDA로 옮겨둔
    상태일 수 있는데(`device_loading_context`), cold 주입은 C++가 host memcpy로
    읽으므로 device 포인터를 넘기면 segfault다 (MoE에서 실제로 겪은 함정 —
    `moe/prism/method.py`의 주석 참조).

    plan이 `halves`를 선언했으면 **N축으로 먼저 쪼갠 뒤** 조각마다 K-split한다.
    조각 경계가 실제 layer의 `output_partition_sizes`와 맞는지는 호출자가
    `plan.check_partition`으로 확인한다 — 여기서는 weight만 보므로 알 수 없다.

    scale은 fp8 스토어의 `weight_scale_inv` `[N/128, K/128]` fp32다. 포맷과 짝이
    맞아야 한다 (bf16에 주거나 fp8에 안 주면 즉사).

    device는 HOT 밴드가 있을 때만 필요하다. warm_node는 warm pinned store가
    상주해야 하는 NUMA 노드이고(계약 ③), 값을 정하는 것은 조립 지점의 몫이다.
    pin_memory=False는 CUDA 없는 테스트용 탈출구다.

    calib은 plan에 sparsity가 있을 때 **필수**다. 없이 부르면 즉사한다 — 조용히
    dense로 절단하면 마스킹이 사라진 채 sparse 벤치 결론이 나온다 (calib.py
    docstring의 무증상 실패와 같은 종류). 점수 재료는 weight와 같은 gather 순서로
    같은 자리에서 만들어야 하므로 이 함수가 맡는다: 인덱스가 여기서만 존재한다.
    """
    if plan.sparsity is not None and calib is None:
        raise PlanError(
            f"layer {layer_idx} proj '{name}': plan has sparsity but no calib "
            f"tables were passed — 마스킹이 조용히 사라진다"
        )
    pp = plan.get(layer_idx, name)
    if pp is None:
        raise PlanError(f"layer {layer_idx} proj '{name}' is not in the plan")
    where = f"layer {layer_idx} proj '{name}'"

    if weight.dim() != 2:
        raise PlanError(f"{where}: weight must be 2-D [N, K], got {tuple(weight.shape)}")
    n, k = int(weight.shape[0]), int(weight.shape[1])
    check_dims(pp, k, n, where)
    for t, what in ((weight, "weight"), (scale, "scale")):
        if t is not None and t.device.type != "cpu":
            raise PlanError(
                f"{where}: {what} must be on CPU (got {t.device}) — cold 주입이 "
                f"host memcpy로 읽는다"
            )

    # 커널 쌍은 **proj별**이다 — 한 모델 안에서 형식이 갈린다 (DSV4의 wo_a는
    # 나머지가 fp8이어도 bf16이다; plan.py docstring 참조).
    resolve_gpu_kernel(pp.kernels.gpu_warm)
    resolve_cpu_kernel(pp.kernels.cpu_cold)
    fmt = store_format(pp.kernels.gpu_warm, pp.kernels.cpu_cold)
    fmt.check(weight, scale, k, n, where)

    if pp.has_tier(Tier.HOT) and device is None:
        raise PlanError(
            f"{where}: plan has HOT rows but no device was given — "
            f"hot store는 VRAM 상주라 배치 device가 로더의 입력이어야 한다"
        )
    idx_device = device if device is not None else torch.device("cpu")
    tile = cold_pack_tile_rows(pp.kernels.cpu_cold)

    parts = tuple(
        _prepare_part(part, weight, scale, fmt, where, idx_device, device,
                      warm_node, pin_memory, tile, layer_idx, calib)
        for part in pp.parts
    )
    return PreparedLinear(
        name=name, layer_idx=layer_idx, k=k, n=n, parts=parts, fmt=fmt
    )


def _prepare_part(part: ProjPart, weight, scale, fmt: LinearStoreFormat, where: str,
                  idx_device, device, warm_node, pin_memory: bool,
                  tile: int, layer_idx: int = 0,
                  calib: Optional[LinearCalibTables] = None) -> PreparedPart:
    """조각 하나: N축 슬라이스 → 티어별 K-split.

    N 슬라이스가 먼저인 이유는 조각마다 밴딩이 다를 수 있어서다 — 합쳐 두면 두
    절반이 같은 마스크를 강요받는다 (자산이 gate/up을 따로 캘리브한 것과 어긋난다).
    """
    sub = where if part.name is None else f"{where} [{part.name}]"
    w = weight if part.name is None else weight[part.n_start : part.n_end]
    s = scale
    if s is not None and part.name is not None:
        block = getattr(fmt, "BLOCK", 1)
        if part.n_start % block or part.n_end % block:
            raise PlanError(
                f"{sub}: part rows [{part.n_start}, {part.n_end}) are not "
                f"{block}-aligned — 배율 블록을 쪼갤 수 없다"
            )
        s = scale[part.n_start // block : part.n_end // block]

    def rows_of(tier: Tier):
        built = _index(tier_rows(part, tier), f"{sub} {tier.value}")
        if built is not None:
            fmt.check_rows(built[0], f"{sub} {tier.value}")
        return built

    # 이 조각을 마스킹하는가 — plan에 sparsity가 있고, 조각이 끄지 않았고,
    # calib 키를 말했을 때만이다. 셋 중 하나라도 없으면 dense로 돈다.
    masking = calib is not None and part.sparse and part.calib is not None
    # 함수 안에서 가져온다: `tiers`가 이 모듈을 import하므로 모듈 레벨은 순환이다.
    # 집합을 여기 복제하면 "어느 티어가 마스킹하는가"의 정의점이 둘이 된다.
    from sglang.srt.layers.prism.linear.tiers import SPARSE_TIERS

    def gpu_shard(tier: Tier, place) -> Optional[LinearTierShard]:
        built = rows_of(tier)
        if built is None:
            return None
        rows, contiguous, k_start = built
        w_flat, s_flat = fmt.gather(w, s, rows, contiguous, k_start)
        # 점수 재료는 SPARSE_TIERS(=WARM)에만 싣는다. hot은 마스킹하지 않으므로
        # (`tiers.SPARSE_TIERS`) a/c를 VRAM에 올리는 것이 순손실이다.
        # 점수 재료는 **device 상주**다 — warm의 weight가 pinned host여도 점수는
        # GPU 커널이 읽는다 (MoE `_slab_sparse_spec`과 같은 거처). `k_index`와 같은
        # device에 둔다: 둘은 항상 같은 순서로 함께 읽힌다.
        cal = None
        if masking and tier in SPARSE_TIERS:
            cal = calib.gather(layer_idx, part.calib, rows, where=sub)
            cal = LinearCalibShard(wn=cal.wn.to(idx_device),
                                   pair_dot=cal.pair_dot.to(idx_device))
        return LinearTierShard(
            w_flat=place(w_flat),
            k_index=rows.to(idx_device),
            contiguous=contiguous,
            fmt=fmt,
            s_flat=None if s_flat is None else place(s_flat),
            k_start=k_start,
            calib=cal,
        )

    def place_warm(t: torch.Tensor) -> torch.Tensor:
        if not pin_memory:
            return t.contiguous()
        store = alloc_pinned_on_node(
            tuple(t.shape), t.dtype, warm_node, f"{sub} warm store")
        store.copy_(t)
        return store

    hot = gpu_shard(Tier.HOT, lambda t: t.to(device, non_blocking=False))
    warm = gpu_shard(Tier.WARM, place_warm)

    cold = None
    built = rows_of(Tier.COLD)
    if built is not None:
        rows, contiguous, _ = built
        w_flat, s_flat, idx, real = fmt.cold_flat(w, s, rows, tile)
        # cold의 점수 재료는 **패딩된** k_index 순서다 — packed 타일의 행 순서가
        # 그것이고, 마스크 비트가 그 순서를 따른다. host에 남긴다: kt가 host
        # memcpy로 읽으므로 device 포인터를 넘기면 segfault다.
        cal = None
        if masking:
            cal = calib.gather(layer_idx, part.calib, idx, real_rows=real, where=sub)
        cold = LinearColdShard(
            w_flat=w_flat, k_index=idx, real_rows=real, contiguous=contiguous,
            fmt=fmt, s_flat=s_flat, calib=cal)

    thr = None
    if masking:
        thr = calib.thr(layer_idx, part.calib)
        if part.sparsity_p is None or part.sparsity_lambda is None:
            raise PlanError(
                f"{sub}: plan has sparsity and this part is masked but carries no "
                f"p/lambda — plan.py가 조각마다 요구한다"
            )

    return PreparedPart(name=part.name, n_start=part.n_start, n_end=part.n_end,
                        hot=hot, warm=warm, cold=cold, thr=thr,
                        sparsity_p=part.sparsity_p,
                        sparsity_lambda=part.sparsity_lambda)

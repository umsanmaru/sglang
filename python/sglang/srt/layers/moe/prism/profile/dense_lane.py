"""dense 레인(`srt/layers/prism/linear`)용 프로파일 래퍼 — weight `[k, n]` 하나가 티어별로 몇 µs냐.

MoE 프로파일러(`profile/__init__.py`)는 어휘가 expert/top_k다. dense 레인은 expert가
없고 **슬롯**(같은 모양의 서로 다른 weight — 실모델이면 그 선형층을 가진 층 수)만
있으므로, 이 모듈이 그 퇴화형(top_k = 1, expert 축 = 슬롯)을 감춘다. 세 티어가
같은 결과 타입(`DenseTier`)을 돌려주므로 harness가 셋을 같은 표에 넣을 수 있다.

    from sglang.srt.layers.moe.prism.profile import DenseShape, dense_hot, dense_warm, dense_cold

    s = DenseShape(k=5120, n=8192, slots=8)
    dense_hot(s, dtype="fp8", kernel_type="pt", device=0).us     # 30.14
    dense_warm(s, 0.9, dtype="bf16", device=0).us                # 171.5
    dense_cold(s, 0.9, dtype="bf16").us                          # 80.9
    dense_cold(s, None, dtype="fp8", kernel_type="pt", m=64).us_per_token   # 209.2

**백엔드 축은 둘이다.**
  `dtype`        스토어이자 백엔드 묶음 — GPU 진입점 계열·K 정렬·스토어 모양을 함께 정한다.
  `kernel_type`  **fp8에만** 있는 커널 변종. 다른 dtype은 변종이 하나뿐이라 인자를 받지 않는다.
      k2 (기본) `kt_tile_k2_fp8b128` — 128×128 블록 배율, 마스크는 VNNI 페어 단위
      k1        `kt_tile_k1_fp8b128` — 같은 배율, 마스크가 **k 행 단위** (score k1).
                warm도 per-k 진입점(`*_sparsek1`)으로 간다 — 그때 입력 x는 1이 아니라
                레벨 값이라 마스킹이 빠지면 결과가 달라진다
      pt        `kt_tile_k2_fp8pt`   — 배율이 선형층당 fp32 하나 (per-tensor). 스토어가
                다르므로 K 정렬이 128이 아니라 **32**로 풀린다.

**슬롯이 왜 필요한가.** `slots=1`이면 iteration마다 같은 weight를 읽어 GPU L2/CPU L3에
남고 실제보다 최대 30% 빠르게 나온다. 풀을 두면 회전이 생겨 그 착시가 사라진다.
실모델 개수를 주는 것이 가장 정확하고, 최소한 `store_mb`가 L2를 넉넉히 넘어야 한다.

**티어별로 무엇을 조절할 수 있나** (사용자 계약):
  hot   — shape (+ dtype/kernel_type). 마스킹이 없으므로 sparsity 인자가 없다
  warm  — shape + sparsity. decode 전용이라 토큰은 1로 고정한다 (executor의 masking 조건)
  cold  — shape + sparsity + `m`. `sparsity=None`이 kt **dense 경로**이고 `m > 1`은 그것만 된다

**바이트 회계.** 세 티어 모두 `weight_bytes`는 "[k, n] weight 하나"로 정규화한다 —
티어끼리 GB/s를 비교하려면 분모가 같아야 한다. hot의 `m > 1`은 worklist 항목이 m개라
커널이 같은 weight를 m번 훑지만(대개 L2 히트) 메모리에서 오는 것은 한 벌이므로,
원본 회계는 `raw` 안에 그대로 남긴다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from sglang.srt.layers.moe.prism.profile.common import (
    PAIR_GROUP,
    Timing,
    emit,
    env_stamp,
    gbps,
    store_of,
)

TIERS = ("hot", "warm", "cold")

# fp8 커널 변종 → (스토어 dtype, kt cold 커널). pt는 배율 기하가 달라 **스토어가 바뀐다**.
KERNEL_TYPES = {
    "k2": ("fp8", "kt_tile_k2_fp8b128"),
    "k1": ("fp8", "kt_tile_k1_fp8b128"),
    "pt": ("fp8pt", "kt_tile_k2_fp8pt"),
}
# 변종을 가진 dtype. 나머지(bf16/mxfp4)는 kernel_type 인자 자체를 거부한다.
KERNEL_TYPE_DTYPE = "fp8"


def _resolve(dtype, kernel_type: Optional[str]):
    """(dtype, kernel_type) → (Store, cold kt 커널 또는 None=스토어 기본).

    `kernel_type`은 fp8 전용이다 — bf16/mxfp4는 커널이 하나뿐이라 고를 것이 없고,
    `fp8pt`는 그 자체가 pt 변종이라 다시 변종을 받지 않는다.
    """
    store = store_of(dtype)
    if kernel_type is None:
        return store, None
    if store.name != KERNEL_TYPE_DTYPE:
        raise ValueError(
            f"kernel_type은 dtype={KERNEL_TYPE_DTYPE!r}에만 있다 (변종 {sorted(KERNEL_TYPES)}). "
            f"dtype={store.name!r}은 커널이 하나뿐이라 인자를 받지 않는다"
            + (" — pt를 원하면 dtype='fp8', kernel_type='pt'로 준다"
               if store.name == "fp8pt" else ""))
    if kernel_type not in KERNEL_TYPES:
        raise ValueError(
            f"kernel_type은 {sorted(KERNEL_TYPES)} 중 하나여야 한다: {kernel_type!r}")
    name, kernel = KERNEL_TYPES[kernel_type]
    return store_of(name), kernel


def dense_backends() -> tuple:
    """고를 수 있는 (dtype, kernel_type) 조합 — harness가 그대로 순회해 스윕을 만들 수 있게 행으로 준다.

        for b in dense_backends():
            if b["warm"]: dense_warm(shape, 0.9, dtype=b["dtype"], kernel_type=b["kernel_type"])
    """
    combos = [("bf16", None), ("mxfp4", None),
              *(("fp8", kt) for kt in KERNEL_TYPES)]
    rows = []
    for dtype, kt in combos:
        store, kernel = _resolve(dtype, kt)
        row = {"dtype": dtype, "kernel_type": kt, "store": store.name,
               "k_align": store.k_align, "elem_bytes": store.elem_bytes,
               "cold_kernel": kernel or store.cpu_kernel,
               "has_vec": store.has_vec,
               "warm": True}
        try:
            row["hot_gpu"] = store.fmt.gemv(pinned=False, sparse=False).__name__
            row["warm_gpu"] = _warm_entry(store, per_k=kt == "k1")
        except ValueError as e:                    # GPU 진입점이 없는 스토어 (cold 전용)
            row["hot_gpu"] = row["warm_gpu"] = None
            row["gpu"] = f"unavailable: {e}"
        rows.append(row)
    return tuple(rows)


@dataclass(frozen=True)
class DenseShape:
    """dense 레인 weight 하나. `slots`는 같은 모양의 weight 개수(회전 풀)다."""

    k: int                 # weight 행 수 = 입력 차원 (GEMV의 K축)
    n: int                 # weight 열 수 = 출력 차원
    slots: int = 8         # 회전 풀 크기 (실모델이면 그 shape을 가진 층/part 수)

    def __post_init__(self):
        if self.k <= 0 or self.n <= 0:
            raise ValueError(f"k, n은 양수여야 한다: k={self.k}, n={self.n}")
        if self.slots < 1:
            raise ValueError(f"slots는 1 이상이어야 한다: {self.slots}")

    @classmethod
    def from_moe(cls, shape, proj: str = "gate", *, slots: Optional[int] = None):
        """MoE `Shape` → dense weight 하나. **축이 proj마다 뒤집히므로 명시로 받는다.**

            gate/up  weight [hidden, inter]  → k=hidden, n=inter
            down     weight [inter, hidden]  → k=inter,  n=hidden

        `slots`를 안 주면 `shape.experts`를 회전 풀로 쓴다 (top_k는 dense에서 1이라 버린다).

            DenseShape.from_moe(Shape(128, 8, 2048, 768), "down")   # k=768, n=2048, slots=128
        """
        if proj not in ("gate", "up", "down"):
            raise ValueError(f"proj must be gate|up|down, got {proj!r}")
        k, n = ((shape.inter, shape.hidden) if proj == "down"
                else (shape.hidden, shape.inter))
        return cls(k=k, n=n, slots=slots if slots is not None else shape.experts)

    def as_dict(self) -> dict:
        return {"k": self.k, "n": self.n, "slots": self.slots}


@dataclass(frozen=True)
class DenseTier:
    """한 (shape, 티어, sparsity) 조합의 결과. 세 티어가 같은 타입을 돌려준다."""

    tier: str                       # hot | warm | cold
    shape: DenseShape
    dtype: str                      # 요청한 dtype
    kernel_type: Optional[str]      # fp8 변종 (k1|k2|pt) 또는 None
    store: str                      # 실제 스토어 (pt는 fp8pt로 갈린다)
    sparsity: Optional[float]       # 요청값. None = 마스킹 없음
    masked: bool                    # 마스크 커널을 탔는가 (cold의 sparsity=None은 False)
    keep_frac: float                # 실현된 살아있는 행 비율 (마스킹 없으면 1.0)
    tokens: int                     # 한 호출의 토큰 수 M
    timing: Timing
    weight_bytes: int               # [k, n] weight 한 벌의 스토어 바이트 (티어 비교의 분모)
    store_mb: float                 # 회전 풀 전체 (slots벌) 크기
    backend: str                    # 실제로 탄 커널/진입점 이름
    raw: dict = field(default_factory=dict)   # 원본 리포트 (회계·NUMA 등)

    @property
    def us(self) -> float:
        return self.timing.us

    @property
    def us_per_token(self) -> float:
        return self.timing.us / self.tokens

    @property
    def kept_bytes(self) -> int:
        return int(self.weight_bytes * self.keep_frac)

    @property
    def gbps(self) -> float:
        """실효 대역폭 — 마스킹 후 실제로 읽은 바이트 기준."""
        return gbps(self.kept_bytes, self.timing.us)

    def as_dict(self) -> dict:
        d = dict(self.timing.as_dict())
        d.update(tier=self.tier, dtype=self.dtype, kernel_type=self.kernel_type,
                 store=self.store, **self.shape.as_dict(),
                 sparsity=self.sparsity, masked=self.masked,
                 keep_frac=round(self.keep_frac, 4), tokens=self.tokens,
                 us_per_token=round(self.us_per_token, 3),
                 weight_bytes=self.weight_bytes, kept_bytes=self.kept_bytes,
                 gbps=self.gbps, store_mb=self.store_mb, backend=self.backend)
        if self.raw:
            d["raw"] = self.raw
        return d


# ── 검증 ─────────────────────────────────────────────────────────────
def _as_dense_shape(shape) -> DenseShape:
    """MoE `Shape`을 그대로 받으면 어느 proj의 weight인지 알 수 없다 — 조용히 고르지 않는다."""
    if isinstance(shape, DenseShape):
        return shape
    if hasattr(shape, "experts") and hasattr(shape, "hidden"):
        raise TypeError(
            "MoE Shape은 proj마다 weight 축이 뒤집혀서 그대로 못 받는다 — 변환을 명시해라: "
            f"DenseShape.from_moe(shape, 'gate')  # k={shape.hidden}, n={shape.inter} 또는 "
            f"DenseShape.from_moe(shape, 'down')  # k={shape.inter}, n={shape.hidden}")
    raise TypeError(f"shape은 DenseShape이어야 한다, got {type(shape).__name__}")


def _check(shape: DenseShape, store, kernel, *, tier: str):
    """하드웨어를 건드리기 전에 정렬을 본다 — 스토어를 만든 뒤 죽으면 느리고 메시지가 멀다."""
    align = max(PAIR_GROUP, store.k_align)
    if shape.k % align:
        raise ValueError(
            f"{store.name}: k는 {align}의 배수여야 한다 (배율 블록이 원본 행 블록에 걸려 "
            f"있다). k={shape.k}")
    if tier == "cold":
        from sglang.srt.layers.moe.prism.profile.warm_cold import N_ALIGN

        na = N_ALIGN.get(kernel or store.cpu_kernel)
        if na and shape.n % na:
            raise ValueError(
                f"{kernel or store.cpu_kernel}: cold의 N(={shape.n})은 {na}의 배수여야 한다 "
                f"(노드 N shard가 커널 stride를 전제한다)")


def _warm_entry(store, *, per_k: bool) -> str:
    """warm이 실제로 부르는 진입점 이름. per-k는 jit 래퍼가 `sparse`를 `sparsek1`로 바꾼다."""
    name = store.fmt.gemv(pinned=True, sparse=True).__name__
    return name.replace("sparse", "sparsek1") if per_k else name


def _store_mb(store, shape: DenseShape) -> float:
    return round(store.store_bytes(shape.slots, shape.k, shape.n) / 1e6, 1)


# ── hot: shape만 ──────────────────────────────────────────────────────
def dense_hot(shape: DenseShape, *, dtype: str = "bf16",
              kernel_type: Optional[str] = None, device=0, m: int = 1,
              reps: int = 100, replays: int = 20, vec: int = 0,
              seed: int = 0) -> DenseTier:
    """GPU 상주 weight의 dense GEMV. 마스킹이 없으므로 sparsity 인자가 없다.

    `m`은 한 호출의 토큰 수다 (decode 1, prefill 청크면 그 값).
    hot은 마스크를 안 쓰므로 `kernel_type` k1과 k2는 **같은 커널**이다 (둘의 차이는
    cold 쪽 마스크 단위와 slab 순열이고, hot의 행우선 device 스토어에는 없다).
    """
    from sglang.srt.layers.moe.prism.profile.hot import dense_gemv

    shape = _as_dense_shape(shape)
    store, _ = _resolve(dtype, kernel_type)
    _check(shape, store, None, tier="hot")
    if m < 1:
        raise ValueError(f"m은 1 이상이어야 한다: {m}")
    r = dense_gemv(shape.k, shape.n, m=m, vec=vec, reps=reps, replays=replays,
                   device=device, experts=shape.slots, topk=1, seed=seed,
                   dtype=store)
    return DenseTier(
        tier="hot", shape=shape, dtype=dtype, kernel_type=kernel_type,
        store=store.name, sparsity=None, masked=False,
        keep_frac=1.0, tokens=m, timing=r.timing,
        weight_bytes=store.store_bytes(1, shape.k, shape.n),
        store_mb=r.w_store_mb,
        backend=store.fmt.gemv(pinned=False, sparse=False).__name__,
        raw={"bytes_per_launch": r.w_bytes_per_launch, "gbps_worklist": r.gbps},
    )


# ── warm: shape + sparsity (decode 전용) ──────────────────────────────
def dense_warm(shape: DenseShape, sparsity: float, *, dtype: str = "bf16",
               kernel_type: Optional[str] = None, device=0, reps: int = 100,
               replays: int = 20, mask_pattern: str = "random",
               warm_node: Optional[int] = None, seed: int = 0) -> DenseTier:
    """pinned host weight를 GPU가 UVA로 제자리 읽는 sparse GEMV.

    토큰은 1로 고정이다 — 마스킹은 decode(M=1)에서만 성립한다 (executor 조건).
    `sparsity=0.0`이 "전부 살리는 sparse 경로"이고, 마스크 커널 자체를 빼는 경로는
    이 래퍼에 없다.
    """
    from sglang.srt.layers.moe.prism.profile.warm_cold import warm_sparse_gemv

    shape = _as_dense_shape(shape)
    store, _ = _resolve(dtype, kernel_type)
    _check(shape, store, None, tier="warm")
    if sparsity is None:
        raise ValueError(
            "warm은 sparsity=None(마스크 없는 경로)을 지원하지 않는다 — 전부 살리려면 "
            "sparsity=0.0을 주면 sparse 커널이 모든 행을 읽는다")
    if not 0.0 <= sparsity <= 1.0:
        raise ValueError(f"sparsity는 [0, 1]이어야 한다: {sparsity}")
    r = warm_sparse_gemv(shape.k, shape.n, sparsity, m=1, reps=reps,
                         replays=replays, device=device, mask_pattern=mask_pattern,
                         seed=seed, warm_node=warm_node, dtype=store,
                         experts=shape.slots, topk=1, per_k=kernel_type == "k1")
    return DenseTier(
        tier="warm", shape=shape, dtype=dtype, kernel_type=kernel_type,
        store=store.name, sparsity=sparsity, masked=True,
        keep_frac=r.keep_frac, tokens=1, timing=r.timing,
        weight_bytes=store.store_bytes(1, shape.k, shape.n),
        store_mb=_store_mb(store, shape),
        backend=_warm_entry(store, per_k=kernel_type == "k1"),
        raw={"dense_bytes": r.dense_bytes, "gbps_dense": r.gbps_dense},
    )


# ── cold: shape + sparsity + m ────────────────────────────────────────
def dense_cold(shape: DenseShape, sparsity: Optional[float], *, dtype: str = "bf16",
               kernel_type: Optional[str] = None, m: int = 1,
               threads: Optional[int] = None,
               numa_map: Optional[Sequence[int]] = None, numa_split: float = 0.5,
               iters: int = 100, replays: int = 8, mask_pattern: str = "random",
               seed: int = 0) -> DenseTier:
    """CPU(kt) weight의 GEMV. CUDA를 쓰지 않는다.

    `sparsity=None`이 kt **dense 경로**다 — 테이블을 설치하지 않으므로 마스크 빌드도
    plan 인코딩도 없고, dense 레인의 cold가 실제로 부르는 것이 이 경로다.
    `sparsity=0.0`(전부 살리는 sparse 경로)과 다른 코드 경로다.
    `m > 1`(prefill 청크)은 `sparsity=None`에서만 된다 — sparse는 decode 전용이다.
    """
    from sglang.srt.layers.moe.prism.profile.cold_cpu import cold_sparse_gemv

    shape = _as_dense_shape(shape)
    store, kernel = _resolve(dtype, kernel_type)
    _check(shape, store, kernel, tier="cold")
    if m < 1:
        raise ValueError(f"m은 1 이상이어야 한다: {m}")
    if m > 1 and sparsity is not None:
        raise ValueError(
            "sparse cold는 decode 전용이다 (kt가 sparsity 테이블 + qlen != 1을 거절한다) — "
            f"m={m}이면 sparsity=None을 줘라")
    if sparsity is not None and not 0.0 <= sparsity <= 1.0:
        raise ValueError(f"sparsity는 [0, 1] 또는 None이어야 한다: {sparsity}")
    # proj="down"이면 K축 = inter = k, N = hidden = n — weight 하나짜리 진입점이다
    # (gateup은 K를 공유하는 GEMV 2개라 dense 레인의 part 하나와 맞지 않는다).
    r = cold_sparse_gemv(shape.k, shape.n, sparsity, iters=iters, replays=replays,
                         mask_pattern=mask_pattern, numa_split=numa_split,
                         threads=threads, cpu_kernel=kernel, dtype=store,
                         seed=seed, numa_map=numa_map, experts=shape.slots,
                         topk=1, m=m, proj="down")
    return DenseTier(
        tier="cold", shape=shape, dtype=dtype, kernel_type=kernel_type,
        store=store.name, sparsity=sparsity,
        masked=sparsity is not None, keep_frac=r.keep_frac, tokens=m,
        timing=r.timing, weight_bytes=store.store_bytes(1, shape.k, shape.n),
        store_mb=_store_mb(store, shape),
        backend=kernel or store.cpu_kernel,
        raw={"dense_bytes": r.dense_bytes, "numa_split": r.numa_split,
             "node_rows": list(r.node_rows or ())},
    )


# ── 스윕 ──────────────────────────────────────────────────────────────
def dense_sweep(shapes: Sequence[DenseShape],
                sparsities: Sequence[Optional[float]] = (0.0, 0.5, 0.9), *,
                tiers: Sequence[str] = TIERS, dtype: str = "bf16",
                kernel_type: Optional[str] = None, device=0,
                hot_m: int = 1, cold_m: int = 1, out: Optional[str] = None,
                quiet: bool = True, **kw) -> dict:
    """(shape × 티어 × sparsity) 표. hot은 sparsity를 돌지 않는다 (shape당 1행).

    `**kw`는 티어별 함수가 받는 것만 골라 넘긴다 (threads/numa_map/reps/seed…).
    실패한 조합은 죽이지 않고 `error` 필드로 남긴다 — 긴 스윕이 한 칸 때문에
    통째로 날아가면 안 된다.
    """
    bad = [t for t in tiers if t not in TIERS]
    if bad:
        raise ValueError(f"tiers는 {TIERS} 중에서: {bad}")
    common = {"dtype": dtype, "kernel_type": kernel_type}
    hot_kw = {k: v for k, v in kw.items()
              if k in ("reps", "replays", "vec", "seed")}
    warm_kw = {k: v for k, v in kw.items()
               if k in ("reps", "replays", "mask_pattern", "warm_node", "seed")}
    cold_kw = {k: v for k, v in kw.items()
               if k in ("threads", "numa_map", "numa_split", "iters", "replays",
                        "mask_pattern", "seed")}
    shapes = [_as_dense_shape(sh) for sh in shapes]
    rows = []
    for shape in shapes:
        for tier in tiers:
            todo = [None] if tier == "hot" else list(sparsities)
            for s in todo:
                try:
                    if tier == "hot":
                        r = dense_hot(shape, device=device, m=hot_m, **common, **hot_kw)
                    elif tier == "warm":
                        r = dense_warm(shape, s, device=device, **common, **warm_kw)
                    else:
                        r = dense_cold(shape, s, m=cold_m, **common, **cold_kw)
                    rows.append(r.as_dict())
                except (ValueError, RuntimeError) as e:
                    rows.append({"tier": tier, **shape.as_dict(), **common,
                                 "sparsity": s, "error": f"{type(e).__name__}: {e}"})
    payload = {
        "bench": "dense_lane_sweep",
        "params": {**common, "tiers": list(tiers), "sparsities": list(sparsities),
                   "hot_m": hot_m, "cold_m": cold_m, "device": device, **kw},
        "shapes": [s.as_dict() for s in shapes],
        "results": rows,
        # cold만 도는 스윕은 CUDA를 초기화하지 않는다 (GPU가 남에게 점유돼 있어도 돌아야 한다).
        "env": env_stamp(device if ({"hot", "warm"} & set(tiers)) else None),
    }
    emit(payload, out, quiet=quiet)
    return payload

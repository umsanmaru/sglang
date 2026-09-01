"""프로파일 API의 공통 부품 — shape 어휘, 타이머, sparsity 합성, 리포트 헬퍼.

이 패키지는 **Plan을 읽지 않는다.** 프로파일러는 "이 치수에서 이 연산이 몇
µs냐"만 묻고 그 답이 Plan을 만드는 입력이 되므로, 역방향 의존을 만들면 안 된다
(그래서 `plan.py`/`weights.py`를 import하지 않고 스토어 형태만 흉내낸다).

**라이브러리 규약**: 여기서는 `SystemExit`을 던지지 않는다. 잘못된 입력은
`ValueError`다 — 남의 프로그램에 심었을 때 프로세스를 죽이지 않기 위해서고,
CLI 껍데기가 그것을 잡아 `SystemExit`으로 바꾼다.

sparsity 합성: 커널(GPU/CPU 양쪽)이 스스로 threshold를 계산하고 페어 점수와
비교하므로 (계약 ①의 k2wl2), 원하는 마스크를 얻으려면 그 입력을 거꾸로 짜야 한다:

    imp²[j] = a[2j]·x0² + a[2j+1]·x1² + 2·c[j]·x0·x1
    keep[j] = imp[j] >= thr[e, round(s/grid)]

x ≡ 1, c ≡ 0, a[2j] = a[2j+1] ∈ {1, 0}, thr 곡선을 상수로 채우면 imp²[j] ∈
{2, 0}이고 thr² = 0.25이므로 keep[j] = (a[2j] == 1)이 된다. 곡선이 상수라 격자
인덱스(→ s → 라우터 가중)가 무엇이든 조회값이 같다 — 즉 **라우터 가중과 무관하게**
마스크가 결정되고, GPU와 CPU가 같은 마스크를 본다.
"""

from __future__ import annotations

import contextlib
import json
import os
import platform
import socket
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import torch

# kt packed 저장의 K축 타일 (kernels.cold_pack_tile_rows의 값). 티어 행 수를 이
# 배수로 잡으면 로더의 타일 올림(real_rows)이 필요 없어 패딩 없는 구성만 다룬다.
K_STEP = 32
PAIR_GROUP = 2

# sparsity 합성 상수 (위 docstring의 역산). thr 곡선이 상수이므로 예산
# 스칼라(p/lam/pmax/grid/ng/renorm_it)는 마스크에 영향을 주지 않지만, kt와
# GPU 커널 양쪽이 유효 범위를 검증하므로 실 plan과 같은 값을 쓴다.
THR_CONST = 0.5
NG, GRID, PMAX, RENORM_IT = 201, 0.005, 0.9, 3
SPARSITY_P, SPARSITY_LAM = 0.5, 4.0

PROJS = ("gate", "up", "down")


# ─── shape 어휘 ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Shape:
    """MoE 한 레이어의 치수. proj가 K축과 N을 결정한다."""

    experts: int
    topk: int
    hidden: int
    inter: int

    def __post_init__(self) -> None:
        if self.topk > 16:
            raise ValueError(f"top_k <= 16 (커널의 per-thread slot 예산), got {self.topk}")
        for name in ("experts", "topk", "hidden", "inter"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    def k_axis(self, proj: str) -> int:
        """이 proj의 contraction 축 전체 길이 (티어가 여기서 행을 나눠 갖는다)."""
        self._check(proj)
        return self.inter if proj == "down" else self.hidden

    def n_cols(self, proj: str) -> int:
        self._check(proj)
        return self.hidden if proj == "down" else self.inter

    def x_row_is_pair(self, proj: str) -> bool:
        """down의 x는 expert별 act라 행이 pair (m, j)다 (executor와 같은 규약)."""
        self._check(proj)
        return proj == "down"

    @staticmethod
    def _check(proj: str) -> None:
        if proj not in PROJS:
            raise ValueError(f"unknown proj {proj!r} (expected one of {PROJS})")

    def replace(self, **kw) -> "Shape":
        """치수 하나만 바꾼 사본 — E 스윕처럼 한 인자만 훑을 때 쓴다."""
        return Shape(**{**self.as_dict(), **kw})

    def as_dict(self) -> dict:
        return {"experts": self.experts, "topk": self.topk,
                "hidden": self.hidden, "inter": self.inter}


# ─── 스토어 dtype (= 백엔드 선택) ──────────────────────────────────────────
@dataclass(frozen=True)
class Store:
    """프로파일이 재는 **스토어 dtype**. 이름 하나가 백엔드 전부를 정한다 (계약 ①의
    "커널 키가 함의한다"를 프로파일 쪽에서 그대로 쓴 것):

      fmt         GPU 진입점 (formats.StoreFormat — hot/warm의 dense·sparse·융합)
      cpu_kernel  cold 백엔드 (kt 클래스 키; 이 dtype의 cold 스토어를 소비할 수 있는 것)
      k_align     티어 K 행 정렬 (bf16 2 / mxfp4 32 / fp8 128)
      has_vec     bf16 커널만 `vec`(W 로드 폭) 인자를 받는다

    합성 스토어도 여기서 만든다 — 프로파일은 실제 체크포인트를 읽지 않고 커널이
    받아들이는 **형태**만 흉내내면 되고, 그 형태가 dtype마다 다르기 때문이다.
    """

    name: str
    cpu_kernel: str
    k_align: int
    elem_bytes: float          # weight 원소 하나의 바이트 (배율 제외)
    has_vec: bool = False

    @property
    def fmt(self):
        from sglang.srt.layers.moe.prism.formats import FORMATS

        return FORMATS[self.name]

    @property
    def cpu_kernels(self) -> tuple:
        return self.fmt.cold_kernels

    def rows_step(self, base: int = K_STEP) -> int:
        """티어 행 수를 반올림할 단위 — 커널 정렬과 K_STEP 중 큰 쪽."""
        return max(base, self.k_align)

    # ── 합성 스토어 ──────────────────────────────────────────────────────
    def _codes_scales(self, rows: int, n: int, seed: int):
        """(codes, scales) CPU 텐서. rows = 이 스토어의 총 k 행 수."""
        g = torch.Generator().manual_seed(seed)
        if self.name == "mxfp4":
            nib = torch.randint(0, 16, (rows, n), generator=g, dtype=torch.int64)
            codes = (nib[0::2] | (nib[1::2] << 4)).to(torch.uint8)      # 행 = k-페어
            scales = torch.full((rows // 32, n), 127, dtype=torch.uint8)  # 2^0
            return codes.contiguous(), scales.contiguous()
        if self.name == "fp8":
            # 지수 1..13, 가수·부호 임의 — denormal(e=0)·NaN(0x7F/0xFF)은 만들지 않는다
            # (커널·CPU 포팅이 공유하는 인코더 전제).
            e = torch.randint(1, 14, (rows, n), generator=g)
            m = torch.randint(0, 8, (rows, n), generator=g)
            sg = torch.randint(0, 2, (rows, n), generator=g)
            codes = ((sg << 7) | (e << 3) | m).to(torch.uint8)
            scales = torch.ones(rows // 128, max(1, n // 128), dtype=torch.float32)
            return codes.contiguous(), scales.contiguous()
        w = torch.empty(rows, n, dtype=torch.bfloat16)
        w.normal_(0, 0.02, generator=g)
        return (w, None)

    def gpu_store(self, experts: int, k_rows, n: int, *, device=None,
                  node: Optional[int] = None, seed: int = 0) -> tuple:
        """hot(device) 또는 warm(pinned) 스토어 인자 — `fmt.store_args`와 같은 순서.

        `k_rows`는 스칼라(균일) 또는 expert당 하나의 시퀀스다 — 스토어가
        [Σₑ k(e), N]이므로 가변 행 수는 총합만 바뀌는 것이고, 어디서 어디까지가
        어느 expert인지는 `row_off`가 말한다 (`tier_indices`가 같이 만든다).

        `node`를 주면 pinned + NUMA 바인딩(warm), 아니면 device 상주(hot)다."""
        rows = per_expert(k_rows, experts, "k_rows")
        for ke in rows:
            self.check_geometry(int(ke), n)
        parts = self._codes_scales(sum(int(k) for k in rows), n, seed)
        out = []
        for t in parts:
            if t is None:
                continue
            if node is not None:
                from sglang.srt.layers.moe.prism.numa import alloc_pinned_on_node

                dst = alloc_pinned_on_node(tuple(t.shape), t.dtype, node,
                                           f"{self.name} warm store")
                dst.copy_(t)
                out.append(dst)
            else:
                out.append(t.to(device) if device is not None else t)
        return tuple(out)

    def cold_store(self, experts: int, n: int, k, *, seed: int = 0) -> tuple:
        """kt 주입용 (w_flat, scale_flat|None) — expert 블록이 ckpt 방향 [N, k(e)]인 1-D flat
        (weights.py `cold_flat`과 같은 레이아웃).

        `k`가 시퀀스면 expert마다 행 수가 다른 형태다. kt는 이 형태를 정식으로
        받는다 — 검증이 shape가 아니라 `N × Σₑ k(e)` 원소 수 기준이다
        (`experts_partial._k_total`).
        """
        rows = [int(x) for x in per_expert(k, experts, "k")]
        if len(set(rows)) > 1:
            parts = [self.cold_store(1, n, ke, seed=seed + e)
                     for e, ke in enumerate(rows)]
            w = torch.cat([p[0] for p in parts]).contiguous()
            s = (None if parts[0][1] is None
                 else torch.cat([p[1] for p in parts]).contiguous())
            return w, s
        k = rows[0]
        self.check_geometry(k, n)
        g = torch.Generator().manual_seed(seed)
        if self.name == "bf16":
            t = torch.empty(experts * n * k, dtype=torch.bfloat16)
            t.normal_(0, 0.02, generator=g)
            return t.contiguous(), None
        if self.name == "mxfp4":
            nib = torch.randint(0, 16, (experts * n, k), generator=g, dtype=torch.int64)
            codes = (nib[:, 0::2] | (nib[:, 1::2] << 4)).to(torch.uint8)
            # kt는 bf16 배율(2^e)을 받아 자기 형식으로 바꾼다 — 1.0으로 채운다.
            scales = torch.ones(experts * n * (k // 32), dtype=torch.bfloat16)
            return codes.reshape(-1).contiguous(), scales
        if self.name == "fp8":
            e = torch.randint(1, 14, (experts * n, k), generator=g)
            m = torch.randint(0, 8, (experts * n, k), generator=g)
            sg = torch.randint(0, 2, (experts * n, k), generator=g)
            codes = ((sg << 7) | (e << 3) | m).to(torch.uint8)
            scales = torch.ones(experts * (n // 128) * (k // 128), dtype=torch.float32)
            return codes.reshape(-1).contiguous(), scales
        raise ValueError(f"unknown store dtype {self.name!r}")

    # ── 검증 / 회계 ──────────────────────────────────────────────────────
    def check_geometry(self, k_rows: int, n: int) -> None:
        if k_rows % self.k_align:
            raise ValueError(f"{self.name}: k rows {k_rows} must be a multiple of "
                             f"{self.k_align} (배율 블록이 원본 행 블록에 걸려 있다)")
        if self.name == "fp8" and n % 128:
            raise ValueError(f"fp8: n {n} must be a multiple of 128 (배율 블록의 N축)")

    def store_bytes(self, experts: int, k_rows, n: int) -> int:
        """스토어 전체 바이트 (코드 + 배율). `k_rows`는 스칼라 또는 expert별 시퀀스."""
        rows = [int(k) for k in per_expert(k_rows, experts, "k_rows")]
        total = sum(rows)
        base = total * n * self.elem_bytes
        if self.name == "mxfp4":
            base += sum(k // 32 for k in rows) * n
        elif self.name == "fp8":
            base += sum(k // 128 for k in rows) * max(1, n // 128) * 4
        return int(base)

    def call(self, fn, args: tuple, *, vec: int = 0) -> None:
        """진입점 호출 — bf16 커널만 받는 `vec` 인자를 여기서 흡수한다."""
        if self.has_vec:
            fn(*args, vec)
        else:
            fn(*args)

    # ── 참조값 (check용 dequant) ─────────────────────────────────────────
    def dequant(self, parts: Sequence[torch.Tensor], k_rows: int, n: int) -> torch.Tensor:
        """스토어 인자 → fp32 [rows, n] (x ≡ 1 레퍼런스가 행을 더할 수 있게)."""
        if self.name == "bf16":
            return parts[0].float().cpu()
        codes, scales = parts[0].cpu(), parts[1].cpu()
        if self.name == "mxfp4":
            from sglang.srt.layers.moe.prism.profile.common import _FP4_TABLE

            low = (codes & 0xF).long()
            high = (codes >> 4).long()
            vals = torch.stack([_FP4_TABLE[low], _FP4_TABLE[high]], dim=1).reshape(-1, n)
            sc = torch.ldexp(torch.ones_like(scales, dtype=torch.float32),
                             scales.int() - 127).repeat_interleave(32, dim=0)
            return vals * sc
        vals = _e4m3_table()[codes.long()]
        sc = scales.float().repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
        return vals * sc[: vals.shape[0], : vals.shape[1]]


_FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
)


def _e4m3_table() -> torch.Tensor:
    """e4m3 바이트 256개 → fp32 (커널 `prism_fp8.cuh`와 같은 비트 산술)."""
    b = torch.arange(256, dtype=torch.int32)
    mag = (b & 0x7F) << 4
    bits = ((b & 0x80) << 8) | torch.where(mag != 0, mag + 0x3C00, torch.zeros_like(mag))
    return (bits << 16).view(torch.float32).clone()


STORES = {
    "bf16": Store(name="bf16", cpu_kernel="kt_tile_k2_bf16", k_align=2,
                  elem_bytes=2.0, has_vec=True),
    "mxfp4": Store(name="mxfp4", cpu_kernel="kt_tile_k2_mxfp4", k_align=32,
                   elem_bytes=0.5),
    "fp8": Store(name="fp8", cpu_kernel="kt_tile_k2_fp8b128", k_align=128,
                 elem_bytes=1.0),
}


def store_of(dtype) -> Store:
    """dtype 이름 → Store. 이미 Store면 그대로 (호출부가 둘 다 받게)."""
    if isinstance(dtype, Store):
        return dtype
    try:
        return STORES[dtype]
    except KeyError:
        raise ValueError(f"unknown store dtype {dtype!r} (known: {sorted(STORES)})") from None


def split_rows(k_axis: int, frac: float, *, step: int = K_STEP) -> int:
    """K축에서 비율 `frac`에 해당하는 행 수를 `step` 배수로 만든다.

    0과 1은 정확히 보존한다 (frac=0 → 0행, frac=1 → 전체). 그 사이에서는
    반올림하되 최소 한 타일은 준다 — "티어가 있는데 행이 0"은 프로파일 입력으로
    의미가 없다.
    """
    if not 0.0 <= frac <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {frac}")
    if frac == 0.0:
        return 0
    if frac == 1.0:
        return k_axis
    rows = int(round(k_axis * frac / step)) * step
    return max(step, min(k_axis, rows))


def per_expert(value, experts: int, name: str) -> tuple:
    """스칼라 → E개 반복, 시퀀스 → 길이 검증 후 그대로.

    "expert마다 다르게"를 받는 인자는 전부 이걸 통과한다 — 균일 구성이 특수한
    경우일 뿐 별도 경로가 아니게 하려는 것이다.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (value,) * experts
    vals = tuple(value)
    if len(vals) != experts:
        raise ValueError(f"{name}: expert마다 하나씩 필요하다 — {experts}개 기대, {len(vals)}개 받음")
    return vals


def spread_values(mean: float, experts: int, *, spread: float,
                  seed: int = 0, lo: float = 0.0, hi: float = 1.0) -> tuple:
    """평균이 **정확히** `mean`인 expert별 값 — [mean-spread, mean+spread]에서 뽑는다.

    대칭 쌍(antithetic)으로 뽑는다: 짝수 번째에 `mean-d`, 홀수 번째에 `mean+d`를
    주면 쌍마다 평균이 보존되므로 시드·E와 무관하게 총평균이 흔들리지 않는다.
    (독립 표본이면 E가 작을 때 표본평균이 목표에서 벗어나 "평균 0.5를 줬는데
    실현값은 0.47"이 되고, 그러면 모델 대비 오차에 그 편차가 섞인다.)
    E가 홀수면 마지막 하나가 정확히 `mean`이다.
    """
    if spread < 0:
        raise ValueError(f"spread must be >= 0, got {spread}")
    if not lo <= mean <= hi:
        raise ValueError(f"mean {mean} out of [{lo}, {hi}]")
    if mean - spread < lo or mean + spread > hi:
        raise ValueError(f"mean {mean} ± {spread} escapes [{lo}, {hi}]")
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(experts // 2):
        d = float(torch.rand(1, generator=g)) * spread
        out += [mean - d, mean + d]
    if experts % 2:
        out.append(mean)
    # 쌍을 섞는다 — 안 섞으면 짝수 expert가 항상 평균 이하, 홀수가 항상 평균
    # 이상이 되어 두 번 부른 결과(행 수와 sparsity)가 구조적으로 **역상관**한다.
    # 실측으로 드러난 문제다: 행 많은 expert에 늘 높은 sparsity가 붙어 페어 가중
    # keep이 0.5 요청에 0.466으로 나왔다. 순열은 평균을 보존한다.
    perm = torch.randperm(experts, generator=g).tolist()
    return tuple(out[i] for i in perm)


def split_rows_varied(k_axis: int, frac: float, experts: int, *,
                      spread: float = 0.0, step: int = K_STEP,
                      seed: int = 0) -> tuple:
    """expert마다 다른 티어 행 수 [E] — 평균 비율이 `frac`, 각각 `step`의 배수.

    실제 plan의 티어 경계는 expert마다 다르다 (중요도 곡선이 expert마다 다르므로).
    `spread=0`이면 전부 같은 값이라 `split_rows`와 같다.

    **반올림이 평균을 흔든다**: 비율은 정확히 대칭으로 뽑지만 각 값을 `step`
    배수로 반올림하므로 실현 평균은 요청과 조금 다르다. 호출부는 실현값을
    리포트에 남겨야 한다 (`sum(rows) / (experts * k_axis)`).
    """
    fracs = spread_values(frac, experts, spread=spread, seed=seed)
    return tuple(split_rows(k_axis, f, step=step) for f in fracs)


def tier_indices(k_axis: int, rows, experts: int, *, skip=0,
                 shuffle: bool = False, seed: int = 0) -> tuple:
    """expert별 K-인덱스를 이어붙인 (k_index [Σₑ k(e)] int32, row_off [E+1] int32).

    `rows`와 `skip`은 스칼라(균일) 또는 expert당 하나의 시퀀스다. expert마다
    **다른** 순열을 쓴다 (`seed + e`) — 실제 plan에서 어느 행이 어느 티어에
    가는지는 expert마다 다르고, 같은 인덱스를 돌려 쓰면 gather 패턴이 실제보다
    규칙적이 된다.

    세 티어를 서로소로 만들려면 같은 `seed`로 세 번 부르고 `skip`을 누적해서
    준다 — expert e의 순열 하나에서 앞을 hot, 그 다음을 warm, 그 다음을 cold가
    갖는다 (`skip`이 expert마다 다른 이유: 앞 티어의 행 수가 expert마다 다르다).
    """
    per = per_expert(rows, experts, "rows")
    skips = per_expert(skip, experts, "skip")
    parts = [tier_index(k_axis, int(k), skip=int(s), shuffle=shuffle, seed=seed + e)
             for e, (k, s) in enumerate(zip(per, skips))]
    off = torch.zeros(experts + 1, dtype=torch.int32)
    off[1:] = torch.tensor(per, dtype=torch.int32).cumsum(0)
    return torch.cat(parts).contiguous(), off.contiguous()


def tier_index(k_axis: int, k_rows: int, *, skip: int = 0,
               shuffle: bool = False, seed: int = 0) -> torch.Tensor:
    """이 티어가 소유하는 K축 행 번호 [k_rows] int32.

    실제 plan의 티어 멤버십은 중요도 순으로 뽑힌 **흩어진 행**이므로 (계약 ①의
    가변 per-expert 인덱스), 고정 시드 순열에서 `skip` 이후 `k_rows`개를 취한다.
    저장 순서는 오름차순 — 로더가 그렇게 굽고, 그 순서가 gather 지역성을
    결정한다. `shuffle=True`는 정렬하지 않은 최악 경우다.
    """
    if k_rows > k_axis - skip:
        raise ValueError(f"tier rows {k_rows} + skip {skip} exceed axis {k_axis}")
    g = torch.Generator().manual_seed(seed)
    rows = torch.randperm(k_axis, generator=g)[skip: skip + k_rows]
    if not shuffle:
        rows = rows.sort().values
    return rows.to(torch.int32).contiguous()


# ─── sparsity 합성 ─────────────────────────────────────────────────────────
def sparse_tables(experts: int, k_rows, sparsity, *,
                  pattern: str = "random", seed: int = 0,
                  ng: int = NG, thr: float = THR_CONST):
    """요청 sparsity를 정확히 실현하는 (a, c, thr_tab, 실현 keep 비율).

    a: fp32 [Σₑ k(e)] — wn². c: fp32 [Σₑ k(e)/2] — 0. thr_tab: fp32 [E, ng].
    모두 weight 스토어와 같은 오프셋(expert 블록 이어붙인 flat)이다.

    `k_rows`와 `sparsity`는 스칼라(균일) 또는 expert당 하나의 시퀀스다 — 실제
    plan은 둘 다 expert마다 다르다(티어 경계는 중요도 곡선이, sparsity는 라우터
    분포가 정한다). thr 곡선이 상수이므로 expert마다 다른 keep 비율을 줘도 마스크
    역산은 그대로 성립한다 (`keep[j] = a[2j] == 1`).

    pattern:
      random — 페어를 시드 고정 랜덤으로 죽인다 (실제 마스크의 산포에 가깝다).
      block  — 앞쪽 페어만 살린다 (kt의 16-페어 워드 스킵이 최대로 먹는 최선 경우).

    반환하는 keep 비율은 **페어 수로 가중한 실현 평균**이다 — expert마다 행 수와
    sparsity가 다르면 단순 평균은 바이트 회계와 어긋난다.
    """
    rows = per_expert(k_rows, experts, "k_rows")
    sps = per_expert(sparsity, experts, "sparsity")
    if pattern not in ("random", "block"):
        raise ValueError(f"unknown mask pattern {pattern!r} (random|block)")
    a_parts, kept, total = [], 0, 0
    for e, (ke, sp) in enumerate(zip(rows, sps)):
        ke = int(ke)
        if not 0.0 <= sp <= 1.0:
            raise ValueError(f"sparsity[{e}] must be in [0, 1], got {sp}")
        if ke % PAIR_GROUP:
            raise ValueError(f"tier rows must be even (pair group), got {ke} at expert {e}")
        npairs = ke // PAIR_GROUP
        keep_n = int(round(npairs * (1.0 - sp)))
        if pattern == "block":
            sel = torch.arange(keep_n)
        else:
            g = torch.Generator().manual_seed(seed + e)
            sel = torch.randperm(npairs, generator=g)[:keep_n]
        pair = torch.zeros(npairs, dtype=torch.float32)
        pair[sel] = 1.0
        a_parts.append(pair.repeat_interleave(PAIR_GROUP))
        kept += keep_n
        total += npairs
    return (
        torch.cat(a_parts).contiguous(),
        torch.zeros(total, dtype=torch.float32),
        torch.full((experts, ng), thr, dtype=torch.float32).contiguous(),
        kept / total if total else 1.0,
    )


# ─── 타이머 ────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Timing:
    """iteration당 µs의 표본 요약. median을 대표값으로 쓴다 — 공유 머신의
    간헐적 간섭이 mean을 오염시키므로."""

    us: float
    min_us: float
    max_us: float
    p90_us: float
    replays: int

    @classmethod
    def of(cls, samples: Sequence[float]) -> "Timing":
        if not samples:
            raise ValueError("no samples")
        s = sorted(samples)
        return cls(
            us=round(statistics.median(s), 3),
            min_us=round(s[0], 3),
            max_us=round(s[-1], 3),
            p90_us=round(s[min(len(s) - 1, int(0.9 * len(s)))], 3),
            replays=len(s),
        )

    def as_dict(self) -> dict:
        return {"us": self.us, "min_us": self.min_us, "max_us": self.max_us,
                "p90_us": self.p90_us, "replays": self.replays}


@dataclass(frozen=True)
class SparseGemv:
    """[k_rows, n_cols] weight 하나의 **sparse** GEMV 결과.

    dense와 달리 "몇 바이트를 읽었나"가 두 개다 — 마스킹 전(dense_bytes)과 실제로
    읽은 양(kept_bytes). GB/s를 dense 바이트로 나누면 대역폭이 과대평가되고 kept로
    나누면 실효값이 나오므로 둘 다 노출한다 (한쪽만 두면 반드시 오독된다).
    """

    where: str            # "warm" (pinned/UVA) | "cold" (CPU/kt)
    k_rows: int
    n_cols: int
    sparsity: float       # 요청값
    keep_frac: float      # 실현값 (합성이 정확하므로 요청과 거의 같다)
    dense_bytes: int
    timing: Timing
    # cold 전용. NUMA N 분할은 요청값이 커널의 N 정렬로 **반올림**되므로
    # (tile_k2는 노드당 256의 배수) 실현된 행 수를 같이 남긴다 — 저장된 JSON이
    # 어느 분할에서 나온 값인지 스스로 말해야 한다.
    numa_split: Optional[float] = None
    node_rows: Optional[tuple] = None

    @property
    def us(self) -> float:
        return self.timing.us

    @property
    def kept_bytes(self) -> int:
        return int(self.dense_bytes * self.keep_frac)

    @property
    def gbps(self) -> float:
        """실효 대역폭 — 실제로 읽은 바이트 기준."""
        return gbps(self.kept_bytes, self.timing.us)

    @property
    def gbps_dense(self) -> float:
        """마스킹 전 바이트 기준. dense 구성과 직접 비교할 때만 의미가 있다."""
        return gbps(self.dense_bytes, self.timing.us)

    def as_dict(self) -> dict:
        d = dict(self.timing.as_dict())
        d.update(where=self.where, k_rows=self.k_rows, n_cols=self.n_cols,
                 sparsity=self.sparsity, keep_frac=round(self.keep_frac, 4),
                 dense_bytes=self.dense_bytes, kept_bytes=self.kept_bytes,
                 gbps=self.gbps, gbps_dense=self.gbps_dense)
        if self.numa_split is not None:
            d["numa_split"] = self.numa_split
            d["node_rows"] = list(self.node_rows or ())
            d["numa_split_realized"] = (
                round(self.node_rows[0] / sum(self.node_rows), 4)
                if self.node_rows else None)
        return d


@contextlib.contextmanager
def nvtx(name: str):
    """NVTX 구간 — nsys 타임라인에서 어느 변형/어느 단계인지 구분하는 유일한 표시.

    표시가 없으면 리포트의 변형들이 이름 없이 섞여서, 어느 커널이 어느 측정에
    속하는지 순서로 추측해야 한다. CUDA가 없으면(cold 단독 경로) no-op이다.
    """
    if not torch.cuda.is_available():
        yield
        return
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def capture(launch: Callable[[int], None], reps: int, *, warmup: int = 3,
            error_mode: str = "thread_local") -> "torch.cuda.CUDAGraph":
    """`launch(i)`를 i=0..reps-1로 한 그래프에 캡처한다.

    error_mode는 thread_local이 기본이다: 캡처 중 cold의 host node(kt의
    cudaLaunchHostFunc)와 그 뒤의 host 측 할당이 global 모드에서 불필요하게
    잡히는 것을 피한다 — sglang의 CudaGraphRunner도 같은 이유로 이 모드다.
    """
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for i in range(warmup):
            launch(i % reps)  # reps < warmup이면 iteration 자원을 돌려 쓴다
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, capture_error_mode=error_mode):
        for i in range(reps):
            launch(i)
    return graph


def replay_timing(graph: "torch.cuda.CUDAGraph", reps: int, *,
                  replays: int = 20, warmup: int = 3) -> Timing:
    """graph replay 시간 / reps = launch당 µs."""
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    per = []
    for _ in range(replays):
        beg = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        beg.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        per.append(beg.elapsed_time(end) * 1e3 / reps)
    return Timing.of(per)


def graph_timing(launch: Callable[[int], None], reps: int, *,
                 replays: int = 20, error_mode: str = "thread_local") -> Timing:
    """커널 `reps`개를 그래프 하나에 담아 replay/reps를 잰다.

    launch당 고정비(커널 launch ~2 µs)를 graph가 지우므로 남는 것이 커널 자체의
    시간이다. **같은 launch를 reps번 반복하면 L2-hot 시간**이 나오므로, 호출자가
    `launch(i)`의 i로 iteration마다 다른 expert를 태워야 한다 (실측 차이 30%).
    """
    graph = capture(launch, reps, error_mode=error_mode)
    try:
        return replay_timing(graph, reps, replays=replays)
    finally:
        # 그래프를 살려두면 캡처한 pool이 다음 캡처의 할당과 겹친다.
        del graph
        torch.cuda.synchronize()


def host_timing(step: Callable[[int], None], reps: int, *, replays: int = 20,
                sync_cuda: bool = False, warmup_rounds: int = 1) -> Timing:
    """host 루프 타이머 — CUDA graph에 담을 수 없는 경로(cold 단독, eager 교차
    검증)용. sync_cuda면 라운드 끝에서 GPU까지 기다린다 (겹침 측정)."""
    for _ in range(warmup_rounds):
        for i in range(reps):
            step(i)
    if sync_cuda:
        torch.cuda.synchronize()
    per = []
    for _ in range(replays):
        t0 = time.perf_counter()
        for i in range(reps):
            step(i)
        if sync_cuda:
            torch.cuda.synchronize()
        per.append((time.perf_counter() - t0) / reps * 1e6)
    return Timing.of(per)


# ─── 리포트 ────────────────────────────────────────────────────────────────
def env_stamp(device: Optional[torch.device] = None) -> dict:
    """숫자를 나중에 해석할 수 있게 하는 최소 정보. 공유 머신이라 GPU 점유량도
    같이 남긴다 (남이 쓰는 중이면 절대값이 흔들린다)."""
    out = {
        "host": socket.gethostname(),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    if torch.cuda.is_available() and device is not None:
        idx = torch.device(device).index or 0
        props = torch.cuda.get_device_properties(idx)
        free, total = torch.cuda.mem_get_info(idx)
        out["gpu"] = {
            "index": idx, "name": props.name,
            "sm": f"{props.major}.{props.minor}",
            "multi_processor_count": props.multi_processor_count,
            "total_mem_gb": round(props.total_memory / 1e9, 1),
            "mem_used_gb": round((total - free) / 1e9, 1),
        }
    keep = ("CUDA_VISIBLE_DEVICES", "SGLANG_PRISM_CPUINFER_THREADS",
            "SGLANG_PRISM_NUMA_MAP", "OMP_NUM_THREADS")
    out["env"] = {k: os.environ[k] for k in keep if k in os.environ}
    return out


def numa_nodes() -> int:
    """NUMA 노드 수. `numa.py`를 쓰지 않는 이유는 이 모듈이 CUDA 없이도 import
    되어야 하기 때문이다 (cold 단독 경로)."""
    try:
        return len([d for d in os.listdir("/sys/devices/system/node")
                    if d.startswith("node") and d[4:].isdigit()]) or 1
    except OSError:
        return 1


def default_cpuinfer_threads() -> int:
    """method.py와 같은 관례: 물리 코어 − 2. 과다구독은 submit/sync 고정비를
    폭증시킨다 (실측: 물리 16코어에 60스레드 → sync 회당 1.85 ms)."""
    env = os.environ.get("SGLANG_PRISM_CPUINFER_THREADS")
    if env:
        return int(env)
    return max(2, (os.cpu_count() or 4) // 2 - 2)


def gbps(nbytes: float, micros: float) -> float:
    return round(nbytes / (micros * 1e-6) / 1e9, 1)


def select_device(index) -> torch.device:
    if not torch.cuda.is_available():
        raise ValueError("CUDA required")
    dev = torch.device(index if isinstance(index, str) else f"cuda:{int(index)}")
    if (dev.index or 0) >= torch.cuda.device_count():
        raise ValueError(f"device {dev} >= {torch.cuda.device_count()} devices")
    torch.cuda.set_device(dev)
    return dev


def emit(payload: dict, out: Optional[str] = None, *, quiet: bool = False) -> str:
    """JSON 직렬화 + (선택) 파일 기록. CLI 껍데기가 쓰는 헬퍼."""
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if not quiet:
        print(text)
    if out:
        Path(out).write_text(text + "\n")
        if not quiet:
            print(f"\n-> {out}", flush=True)
    return text

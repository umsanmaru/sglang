"""hot 티어 dense GEMV의 실행시간 (프로파일 ①).

재는 것은 **hot 티어가 실제로 부르는 그 커널**이다: `gemv_worklist_indexed`
(tiers.py `ResidentTier` → device 상주 W, 인덱스 경로). 밴드/bmm 변형은 폐기
됐으므로 (kernels.py의 registry 주석) 재현할 대상이 아니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from sglang.srt.layers.moe.prism.profile.common import (
    Shape,
    Timing,
    env_stamp,
    gbps,
    graph_timing,
    nvtx,
    select_device,
    split_rows,
    tier_index,
)


@dataclass(frozen=True)
class ProjGemv:
    """한 (proj, k_rows) 조합의 결과."""

    proj: str
    k_rows: int
    n_cols: int
    k_axis: int
    m: int
    timing: Timing
    w_bytes_per_launch: int
    w_store_mb: float

    @property
    def us(self) -> float:
        return self.timing.us

    @property
    def gbps(self) -> float:
        return gbps(self.w_bytes_per_launch, self.timing.us)

    def as_dict(self) -> dict:
        d = dict(self.timing.as_dict())
        d.update(proj=self.proj, k_rows=self.k_rows, n_cols=self.n_cols,
                 k_axis=self.k_axis, m=self.m,
                 w_bytes_per_launch=self.w_bytes_per_launch,
                 w_store_mb=self.w_store_mb, gbps=self.gbps)
        return d


@dataclass(frozen=True)
class HotGemvReport:
    shape: Shape
    params: dict
    results: tuple
    env: dict

    def __getitem__(self, proj: str) -> ProjGemv:
        for r in self.results:
            if r.proj == proj:
                return r
        raise KeyError(proj)

    def us(self, proj: str) -> float:
        return self[proj].us

    @property
    def layer_gemv_us(self) -> Optional[float]:
        """한 레이어의 GPU dense GEMV 총합 (gate + up + down, up == gate 치수).
        gate와 down을 다 재지 않았으면 None."""
        try:
            return round(2 * self.us("gate") + self.us("down"), 3)
        except KeyError:
            return None

    def as_dict(self) -> dict:
        d = {
            "bench": "gpu_dense_gemv",
            "kernel": "gemv_worklist_indexed (hot / device-resident W)",
            "shape": self.shape.as_dict(),
            "params": self.params,
            "results": [r.as_dict() for r in self.results],
            "env": self.env,
        }
        if self.layer_gemv_us is not None:
            d["layer_gemv_us"] = self.layer_gemv_us
        return d


def _ids(shape: Shape, m: int, reps: int, device, seed: int) -> list:
    """iteration마다 다른 (token, expert) 배정. 토큰 안에서는 중복 없는 top_k
    (라우터의 성질) — reps×m×top_k가 E를 넘으면 자연히 전 expert를 훑는다.

    이게 없으면 같은 expert를 reps번 반복해 W가 L2에 남아 실제 decode보다
    낙관적인 시간이 나온다 (실측: reps=1 12.6 µs vs reps=100 16.5 µs).
    """
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(reps):
        rows = [torch.randperm(shape.experts, generator=g)[: shape.topk]
                for _ in range(m)]
        out.append(torch.stack(rows).to(torch.int32).to(device))
    return out


def measure_proj(shape: Shape, proj: str, k_rows: int, *, m: int = 1,
                 vec: int = 0, reps: int = 100, replays: int = 20,
                 device=0, shuffle_index: bool = False, seed: int = 0,
                 label: Optional[str] = None) -> ProjGemv:
    """한 (proj, k_rows) 조합의 launch당 µs.

    스토어·인덱스·출력 레이아웃은 weights.py의 hot shard와 같다:
    w_flat [Σₑ k(e), N] device, row_off [E+1] int32 device,
    k_index uint16 device, out3d [M, top_k, N].
    """
    from sglang.jit_kernel.prism_gemv import gemv_worklist_indexed

    dev = select_device(device) if not isinstance(device, torch.device) else device
    E, topk = shape.experts, shape.topk
    n = shape.n_cols(proj)
    axis = shape.k_axis(proj)
    x_row_is_pair = shape.x_row_is_pair(proj)
    if k_rows <= 0 or k_rows > axis:
        raise ValueError(f"k_rows {k_rows} out of range for axis {axis}")

    w_flat = torch.empty(E * k_rows, n, dtype=torch.bfloat16, device=dev)
    w_flat.normal_(0, 0.02)
    row_off = (torch.arange(E + 1, dtype=torch.int32) * k_rows).to(dev)
    rows = tier_index(axis, k_rows, shuffle=shuffle_index, seed=seed)
    kidx = rows.to(torch.uint16).repeat(E).contiguous().to(dev)
    x_rows = m * topk if x_row_is_pair else m
    x = torch.empty(x_rows, axis, dtype=torch.bfloat16, device=dev)
    x.normal_(0, 1.0)
    out = torch.zeros(m, topk, n, dtype=torch.bfloat16, device=dev)
    ids = _ids(shape, m, reps, dev, seed + 1)

    def launch(i: int) -> None:
        gemv_worklist_indexed(x, ids[i], w_flat, row_off, kidx, out, 0,
                              x_row_is_pair, torch.cuda.current_stream(), vec)

    try:
        with nvtx(f"hot/{label or proj}"):
            timing = graph_timing(launch, reps, replays=replays)
    finally:
        del w_flat, x, out, ids
        torch.cuda.empty_cache()

    return ProjGemv(
        proj=label or proj, k_rows=k_rows, n_cols=n, k_axis=axis, m=m,
        timing=timing, w_bytes_per_launch=m * topk * k_rows * n * 2,
        w_store_mb=round(E * k_rows * n * 2 / 1e6, 1),
    )


def hot_dense_gemv(shape: Shape, *, hot_frac: float = 1.0,
                   projs: Sequence[str] = ("gate", "down"),
                   k_rows: Optional[int] = None, n_cols: Optional[int] = None,
                   m: int = 1, vec: int = 0, reps: int = 100, replays: int = 20,
                   device=0, shuffle_index: bool = False,
                   seed: int = 0) -> HotGemvReport:
    """hot dense GEMV를 proj별로 잰다.

    `k_rows`/`n_cols`를 함께 주면 raw 모드다 — proj 치수를 무시하고 그 weight
    shape 하나만 잰다 (`proj` 라벨은 "raw").

    warmup_jit()을 먼저 부르는 이유: 캡처 안에서 JIT 컴파일이 일어나면 안 된다.
    """
    from sglang.jit_kernel.prism_gemv import warmup_jit

    dev = select_device(device)
    warmup_jit()

    params = {"hot_frac": hot_frac, "m": m, "vec": vec, "reps": reps,
              "replays": replays, "shuffle_index": shuffle_index, "seed": seed}
    results = []
    if k_rows or n_cols:
        if not (k_rows and n_cols):
            raise ValueError("raw 모드는 k_rows와 n_cols를 함께 준다")
        raw = Shape(experts=shape.experts, topk=shape.topk,
                    hidden=k_rows, inter=n_cols)
        params.update(raw_k=k_rows, raw_n=n_cols)
        results.append(measure_proj(
            raw, "gate", k_rows, m=m, vec=vec, reps=reps, replays=replays,
            device=dev, shuffle_index=shuffle_index, seed=seed, label="raw"))
    else:
        for proj in projs:
            rows = split_rows(shape.k_axis(proj), hot_frac)
            if rows == 0:
                raise ValueError("hot_frac 0은 잴 것이 없다")
            results.append(measure_proj(
                shape, proj, rows, m=m, vec=vec, reps=reps, replays=replays,
                device=dev, shuffle_index=shuffle_index, seed=seed))

    return HotGemvReport(shape=shape, params=params, results=tuple(results),
                         env=env_stamp(dev))


def dense_gemv(k: int, n: int, *, m: int = 1, vec: int = 0, reps: int = 100,
               replays: int = 20, device=0, experts: int = 1, topk: int = 1,
               seed: int = 0) -> ProjGemv:
    """weight shape [k, n] 하나의 dense GEMV — expert/top_k/hot_frac을 1로 접은 형태.

        dense_gemv(768, 512, device=1).us     # 이 shape의 launch당 µs

    MoE 어휘 없이 "이 치수의 GEMV가 몇 µs냐"만 묻고 싶을 때 쓴다. hot 티어와 같은
    커널·같은 호출 규약이고, 갈리는 것은 worklist가 (1 토큰 × 1 expert)로
    퇴화한다는 것뿐이다.

    **주의 — 이 형태는 커널의 상한을 재는 것이지 hot 티어의 시간이 아니다.**
    E=1이면 스토어가 `k·n·2` B뿐이라 GPU L2에 들어앉을 수 있고, expert가 하나라
    iteration마다 다른 expert를 태우는 회전이 불가능하다. 그래서 반복 launch가
    L2 히트로 떨어져 실제 decode보다 낙관적인 값이 나온다 (실측: 768×512에서
    회전 있음 16.5 µs vs 없음 12.6 µs, 30% 차이). 실제 hot 티어 비용을 원하면
    `hot_dense_gemv(shape, ...)`에 실제 `experts`를 주면 된다 — 그쪽은 회전을
    자동으로 한다. `experts`/`topk` 인자로 이 helper에서도 풀을 키울 수 있다.
    """
    return measure_proj(
        Shape(experts=experts, topk=topk, hidden=k, inter=n), "gate", k,
        m=m, vec=vec, reps=reps, replays=replays, device=device,
        shuffle_index=False, seed=seed, label="dense")

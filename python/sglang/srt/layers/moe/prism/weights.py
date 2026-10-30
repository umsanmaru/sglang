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
- 세 티어 모두 인덱스다. 티어당 다중 밴드, expert 간 기하 불일치(가변 k),
  gate ≠ up — 전부 지원된다. 밴드 시절의 NotImplementedError 세 개가 사라졌다.
- cold만 **타일 올림**을 받는다: plan/자산이 지키는 정렬은 페어(%2)뿐인데
  packed 저장은 커널 타일의 배수를 요구하므로(`kernels.cold_pack_tile_rows` —
  커널 키가 함의하는 값), 로더가 올리고 0 열을 채우며 `real_rows`가 패딩 전
  행 수를 나른다. hot/warm은 GEMV 루프라 정렬 요구가 없다.
- 남은 전환기 물건은 GPU 티어의 밴드 호환 속성뿐이다: `weights`/`k_offset`이
  균일·연속을 요구하며 아니면 즉사한다 — 조용히 틀린 값을 주느니 죽는 편이
  낫고, 그 raise가 켜지는 날이 그 속성을 지울 날이다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import torch

from sglang.srt.layers.moe.prism.calib import CalibShard, CalibTables
from sglang.srt.layers.moe.prism.formats import BF16, FullWeights, ProjSource, StoreFormat
from sglang.srt.layers.moe.prism.index import (
    IDX_DTYPE,
    LayerIndex,
    TierIndex,
    from_bands,
    validate_layer,
)
from sglang.srt.layers.moe.prism.numa import alloc_pinned_on_node
from sglang.srt.layers.moe.prism.plan import (
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

    w_flat: torch.Tensor       # bf16 [Σₑ k[e], N] (또는 mxfp4 codes u8 [Σₑ k[e]/2, N]) — hot=device / warm=pinned
    row_off: torch.Tensor      # int32 [E+1] (device — 커널이 읽는다) — 항상 **k 단위**
    k_index: torch.Tensor      # uint16 [Σₑ k[e]] (device)
    contiguous: bool
    # 스토어 포맷 (formats.py). 커널 진입점·정렬·스토어 인자 형태를 이 객체가 정한다.
    fmt: StoreFormat = BF16
    # mxfp4만: E8M0 배율 u8 [Σₑ k[e]/32, N] — w_flat과 같은 거처.
    s_flat: Optional[torch.Tensor] = None
    # 밴드 퇴화형일 때만 채워진다 (전환기 호환용, 로드 시 host에서 계산).
    uniform_k: Optional[int] = None
    band_start: Optional[int] = None
    calib: Optional[CalibShard] = None
    # expert당 최대 K 행 (host에서 로드 시 1회 계산) — grouped W-resident 커널의 smem 예산.
    k_max: int = 0

    @property
    def num_experts(self) -> int:
        return int(self.row_off.numel()) - 1

    @property
    def total_rows(self) -> int:
        return int(self.row_off[-1])

    def store_args(self) -> tuple:
        """커널 래퍼에 넘기는 스토어 텐서들 — 포맷이 정한다 (bf16: (w_flat,), mxfp4: (codes, scales))."""
        return self.fmt.store_args(self)

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
            k_max=k,
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
class ColdShard:
    """한 proj의 cold 스토어 — expert 블록 [N, k_pad(e)]를 이어 붙인 flat.

    ckpt 방향(N-major)을 유지하는 이유는 소비자인 kt pack(`from_mat`)이 그 방향의
    `[N, k]`를 읽기 때문이고, flat인 이유는 hot/warm과 같다: expert마다 k가 다르면
    `[E, N, k]` 적층이 성립하지 않는다.

    **패딩**: plan/자산이 지키는 정렬은 페어(%2)뿐인데 packed 저장은 커널 타일의
    배수를 요구한다 (`kernels.cold_pack_tile_rows` — 커널 키가 함의하는 값).
    로더가 `k_real`을 타일 경계까지 올리고 그만큼의 열을 0으로 채운다. 0 weight는
    dense 경로에서 무해하고, sparse 경로의 tail 비트는 kt가 `real_rows`로 끈다.
    """

    w_flat: torch.Tensor       # bf16 [Σₑ N·k_pad(e)] (expert 블록 [N, k_pad]) / mxfp4: u8 nibble [N, k_pad/2]
    row_off: torch.Tensor      # int32 [E+1] — **패딩된** 행 기준 (k 단위)
    k_index: torch.Tensor      # uint16 [Σₑ k_pad(e)] — 패딩 항목은 0을 가리킨다
    real_rows: torch.Tensor    # int32 [E] — 패딩 전 행 수
    calib: Optional[CalibShard] = None
    fmt: StoreFormat = BF16
    s_flat: Optional[torch.Tensor] = None   # mxfp4: bf16 배율 블록 [N, k_pad/32] flat

    @property
    def num_experts(self) -> int:
        return int(self.row_off.numel()) - 1

    @property
    def total_rows(self) -> int:
        return int(self.row_off[-1])


@dataclass
class PendingColdTensors:
    """cold backend 접속 전까지의 임시 소유자. backend가 붙으면 이 텐서들은
    C++로 pack되어 넘어가고 PreparedWeights.cold는 핸들로 대체된다."""

    gate: Optional[ColdShard]
    up: Optional[ColdShard]
    down: Optional[ColdShard]

    def band(self, proj: Proj) -> Optional[ColdShard]:
        return {Proj.GATE: self.gate, Proj.UP: self.up, Proj.DOWN: self.down}[proj]


@dataclass
class PreparedWeights:
    """Stage 2의 유일한 산출물이자 weight lifetime owner (계약 ③)."""

    hot: Optional[HotStore]  # VRAM 상주 밴드 (없으면 세 proj 모두 None)
    warm: WarmStore
    cold: PendingColdTensors  # S5 이후: ColdHandle
    # warm을 kt 포맷 slab(pinned)으로 두는 모드(2026-08-27): warm 행을 cold와 같은
    # ckpt-방향 flat(타일 올림)으로 잘라 kt 인스턴스에 주입한다. 이때 `warm`의 shard는
    # 전부 None이다 — GPU는 kt slab을 packed GEMV/grouped(cold-layout)로 읽는다.
    warm_kt: Optional[PendingColdTensors] = None
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


def _cold_flat(
    src: torch.Tensor, ti: TierIndex, real_rows, tile: int
) -> tuple:
    """cold 스토어 [Σₑ N·k_pad(e)]와 패딩된 (row_off, k_index)를 만든다.

    expert 블록은 ckpt 방향 `[N, k_pad(e)]`이고 패딩 열은 0이다. 패딩 인덱스는
    아무 열이어도 되지만(weight가 0) 0을 넣는다 — kt가 축 범위를 검증한다.
    """
    E, N, _ = src.shape
    pad = [((int(r) + tile - 1) // tile) * tile for r in real_rows]
    off = torch.zeros(E + 1, dtype=torch.int32)
    for e, kp in enumerate(pad):
        off[e + 1] = int(off[e]) + kp
    total = int(off[-1])
    flat = torch.zeros(total * N, dtype=src.dtype)
    idx = torch.zeros(total, dtype=torch.int64)
    for e in range(E):
        kr, kp, o0 = int(real_rows[e]), pad[e], int(off[e])
        if kr == 0:
            continue
        rows = ti.for_expert(e)[:kr].to(torch.int64)
        idx[o0 : o0 + kr] = rows
        blk = flat[o0 * N : (o0 + kp) * N].view(N, kp)
        blk[:, :kr] = src[e].index_select(1, rows)
    return (
        flat.contiguous(),
        off,
        idx.to(IDX_DTYPE),
        torch.tensor([int(r) for r in real_rows], dtype=torch.int32),
    )


def _build_shard(
    src: ProjSource,
    ti: TierIndex,
    *,
    fmt: StoreFormat,
    idx_device: torch.device,
    place,
    calib: Optional[CalibShard],
) -> TierShard:
    """티어 스토어 하나. `fmt.gather`가 포맷대로 flat을 만들고 `place`가 최종 거처로 옮긴다
    (hot=device / warm=pinned). row_off·k_index는 **항상 커널이 읽는 device**로 간다."""
    uniform_k, band_start = _uniform_band(ti)
    w_flat, s_flat = fmt.gather(src, ti, uniform_k, band_start)
    ro = ti.row_off
    k_max = int((ro[1:] - ro[:-1]).max()) if ti.num_experts > 0 else 0
    return TierShard(
        w_flat=place(w_flat),
        row_off=ti.row_off.to(idx_device),
        k_index=ti.idx.to(idx_device),
        contiguous=ti.contiguous,
        fmt=fmt,
        s_flat=None if s_flat is None else place(s_flat),
        uniform_k=uniform_k,
        band_start=band_start,
        calib=calib,
        k_max=k_max,
    )


def _by_proj(shards: Mapping[Proj, object]) -> dict:
    """proj 키 dict → gate/up/down 키워드. 세 스토어 타입이 같은 세 필드를 받는다."""
    return {"gate": shards[Proj.GATE], "up": shards[Proj.UP], "down": shards[Proj.DOWN]}


@dataclass(frozen=True)
class _CalibGather:
    """점수 재료(wn²·pair_dot)를 weight와 **같은 인덱스·같은 오프셋**으로 모으는 창구.

    티어마다 절단이 다르고 cold/warm-kt는 타일 올림까지 받으므로, "어느 인덱스로
    모으느냐"가 티어별로 갈린다. dense plan(calib 없음)이면 항상 None을 낸다.
    """

    tables: Optional[CalibTables]
    layer_idx: int
    proj: Proj
    where: str

    def for_tier(self, ti: TierIndex, tier_name: str,
                 real_rows=None) -> Optional[CalibShard]:
        if self.tables is None:
            return None
        return self.tables.gather_index(
            self.layer_idx, self.proj, ti, real_rows, f"{self.where} {tier_name}")


def _hot_placer(device):
    """hot 스토어의 거처 — VRAM (계약 ③)."""
    def place(t: torch.Tensor) -> torch.Tensor:
        return t.to(device, non_blocking=False)
    return place


def _warm_placer(pin_memory: bool, warm_node: Optional[int], where: str):
    """warm 스토어의 거처 — `warm_node`에 바인딩된 pinned 메모리. GPU가 UVA로 제자리
    읽으므로 어느 소켓에 앉느냐가 곧 PCIe 경로다. pin_memory=False는 CUDA 없는
    테스트용 탈출구다."""
    def place(t: torch.Tensor) -> torch.Tensor:
        if not pin_memory:
            return t.contiguous()
        store = alloc_pinned_on_node(
            tuple(t.shape), t.dtype, warm_node, f"{where} warm store")
        store.copy_(t)
        return store
    return place


def _resolve_store_format(plan: Plan, fmt: Optional[StoreFormat]) -> StoreFormat:
    """스토어 포맷 — 주어지지 않으면 plan의 gpu_warm 커널 키가 함의하는 것 (계약 ①)."""
    if fmt is not None:
        return fmt
    from sglang.srt.layers.moe.prism.kernels import gpu_store_format

    return gpu_store_format(plan.kernels.gpu_warm)


def _check_calib_pairing(layer_idx: int, plan: Plan,
                         calib: Optional[CalibTables], dims) -> None:
    """plan.sparsity와 calib은 all-or-nothing이다 — 한쪽만 있으면 마스킹이 조용히
    사라지거나 테이블이 통째로 버려지므로 즉사한다."""
    if (plan.sparsity is None) != (calib is None):
        raise PlanError(
            f"layer {layer_idx}: plan.sparsity and calib must both be present "
            f"or both absent (sparsity={plan.sparsity is not None}, "
            f"calib={calib is not None})"
        )
    if calib is not None:
        calib.check_dims(dims, plan.sparsity)


def _index_device(layer_index: LayerIndex, layer_idx: int,
                  device: Optional[torch.device]) -> torch.device:
    """인덱스(row_off·k_index)가 갈 곳 = 커널이 읽는 곳. HOT 행이 있으면 device는
    **로더의 필수 입력**이다 (계약 ③: hot 스토어는 VRAM 상주라 배치 device를
    로더가 스스로 정할 수 없다). hot 없는 plan은 CPU로 남아 CUDA 없이도 돈다."""
    if any(layer_index.get(p, Tier.HOT) is not None for p in Proj) and device is None:
        raise PlanError(
            f"layer {layer_idx}: plan has HOT rows but no device was given — "
            f"hot store는 VRAM 상주라 배치 device가 로더의 입력이어야 한다"
        )
    return device if device is not None else torch.device("cpu")


def _prepare_warm_kt_shard(src: ProjSource, ti: TierIndex, *, fmt: StoreFormat,
                           num_experts: int, tile_rows: int, where: str,
                           calib: _CalibGather) -> ColdShard:
    """warm을 **kt 포맷**으로 절단한다 — cold와 같은 절단(타일 올림, ckpt 방향)이고
    저장(pack)은 kt가 한다. 그래서 산출물도 cold와 같은 ColdShard다."""
    real = [ti.k_rows(e) for e in range(num_experts)]
    flat, off, idx, real_t = _cold_flat(
        fmt.cold_source(src, f"{where} warm-kt"), ti, real, tile_rows)
    padded = TierIndex(row_off=off, idx=idx, contiguous=ti.contiguous)
    return ColdShard(
        w_flat=flat, row_off=off, k_index=idx, real_rows=real_t,
        calib=calib.for_tier(padded, "warm", real_rows=real),
    )


def _prepare_cold_shard(src: ProjSource, ti: TierIndex, *, fmt: StoreFormat,
                        num_experts: int, tile_rows: int, where: str,
                        calib: _CalibGather) -> ColdShard:
    """cold 스토어. 타일 올림은 커널 키가 함의하는 값이고(계약 ①), 패딩 열은 0이다."""
    real = [ti.k_rows(e) for e in range(num_experts)]
    fmt.check_index(ti, f"{where} cold")
    flat, s_flat, off, idx, real_t = fmt.cold_flat(src, ti, real, tile_rows)
    idx = idx.to(IDX_DTYPE)
    padded = TierIndex(row_off=off, idx=idx, contiguous=ti.contiguous)
    return ColdShard(
        w_flat=flat, row_off=off, k_index=idx, real_rows=real_t,
        calib=calib.for_tier(padded, "cold", real_rows=real), fmt=fmt, s_flat=s_flat,
    )


def _warm_kt_tensors(enabled: bool, shards: dict, layer_idx: int):
    """warm-kt 인스턴스 입력. kt partial 인스턴스는 세 proj를 모두 요구하므로(cold와
    같은 제약) warm이 일부 proj에만 있는 plan은 이 모드로 실행할 수 없다 — 조용히
    row-major로 떨어지지 않고 즉사한다."""
    if not enabled or all(v is None for v in shards.values()):
        return None
    if any(v is None for v in shards.values()):
        missing = [p.value for p, v in shards.items() if v is None]
        raise PlanError(
            f"layer {layer_idx}: warm-kt needs warm rows on all projections "
            f"(missing {missing}) — kt partial instances carry gate/up/down together")
    return PendingColdTensors(gate=shards[Proj.GATE], up=shards[Proj.UP], down=shards[Proj.DOWN])


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
    cold_tile_rows: int = 32,
    warm_kt: bool = False,
    fmt: Optional[StoreFormat] = None,
    w13_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
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
    fmt는 스토어 포맷(formats.py). None이면 plan의 gpu_warm 커널 키가 함의하는 포맷
    (kernels.gpu_store_format). mxfp4면 w13_scale/w2_scale([E, N, K/32])이 필수다.
    """
    dims = plan.dims
    fmt = _resolve_store_format(plan, fmt)
    full = FullWeights(w13=w13, w2=w2, w13_scale=w13_scale, w2_scale=w2_scale)
    _check_calib_pairing(layer_idx, plan, calib, dims)
    fmt.check_full_shapes(full, dims, f"layer {layer_idx}")

    # 티어 멤버십을 인덱스로 확정하고 **순열·페어를 검증한다** (계약 ①).
    # 밴드 검증이 plan.py에서 사라진 자리를 여기가 메운다 — 로드마다 돈다.
    layer_index = from_bands(plan, layer_idx)
    validate_layer(layer_index, dims, layer_idx)
    idx_device = _index_device(layer_index, layer_idx, device)
    place_hot = _hot_placer(device)

    hot_shards: dict[Proj, Optional[TierShard]] = {}
    warm_shards: dict[Proj, Optional[TierShard]] = {}
    warm_kt_shards: dict[Proj, Optional[ColdShard]] = {}
    cold_shards: dict[Proj, Optional[ColdShard]] = {}

    for proj in Proj:
        where = f"layer {layer_idx} {proj.value}"
        src = fmt.proj_source(full, dims.intermediate_size, proj)  # [E, N, K…]
        gather = _CalibGather(calib, layer_idx, proj, where)
        for tier in (Tier.HOT, Tier.WARM):
            ti_ = layer_index.get(proj, tier)
            if ti_ is not None:
                fmt.check_index(ti_, f"{where} {tier.value}")

        hot_ti = layer_index.get(proj, Tier.HOT)
        hot_shards[proj] = None if hot_ti is None else _build_shard(
            src, hot_ti, fmt=fmt, idx_device=idx_device, place=place_hot,
            calib=gather.for_tier(hot_ti, "hot"),
        )

        # warm은 두 갈래다: 기본은 GPU가 읽는 row-major pinned 스토어, warm_kt면
        # cold와 같은 kt 포맷(그래서 산출물 타입도 cold와 같다).
        warm_ti = layer_index.get(proj, Tier.WARM)
        if warm_kt:
            warm_shards[proj] = None
            warm_kt_shards[proj] = None if warm_ti is None else _prepare_warm_kt_shard(
                src, warm_ti, fmt=fmt, num_experts=dims.num_experts,
                tile_rows=cold_tile_rows, where=where, calib=gather,
            )
        else:
            warm_kt_shards[proj] = None
            warm_shards[proj] = None if warm_ti is None else _build_shard(
                src, warm_ti, fmt=fmt, idx_device=idx_device,
                place=_warm_placer(pin_memory, warm_node, where),
                calib=gather.for_tier(warm_ti, "warm"),
            )

        cold_ti = layer_index.get(proj, Tier.COLD)
        cold_shards[proj] = None if cold_ti is None else _prepare_cold_shard(
            src, cold_ti, fmt=fmt, num_experts=dims.num_experts,
            tile_rows=cold_tile_rows, where=where, calib=gather,
        )

    return PreparedWeights(
        hot=HotStore(**_by_proj(hot_shards)),
        warm=WarmStore(**_by_proj(warm_shards)),
        cold=PendingColdTensors(**_by_proj(cold_shards)),
        thr=(
            None if calib is None
            else {proj: calib.thr(layer_idx, proj) for proj in Proj}
        ),
        warm_kt=_warm_kt_tensors(warm_kt, warm_kt_shards, layer_idx),
    )

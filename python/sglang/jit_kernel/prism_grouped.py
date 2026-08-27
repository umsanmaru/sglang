"""Prism grouped GEMM (prefill 형태) — `prism_grouped.cuh`의 Python 진입점.

worklist GEMV(`prism_gemv.py`)가 pair마다 W를 다시 읽는 decode 형태라면, 이
커널은 pair를 expert로 묶어 W 타일을 한 번 읽고 그 expert의 모든 토큰에 곱한다.
호출 규약은 worklist와 같다 (current stream에 launch만, 출력은 호출자 소유,
rejoin 레이아웃 out[m, j, off:off+N]). 다른 것은 `topk_ids` 대신 **그룹핑
텐서 셋**(pair_sorted / pair_off / tile_off — `prism.grouping.build_grouping`)을
받는다는 것이다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# 커널의 토큰 타일 크기 (kBM). grouping.py가 tile_off를 만들 때 같은 값을 써야
# 하므로 여기서 한 번만 정의한다 — 어긋나면 블록이 남의 pair를 계산한다.
TILE_M = 128

# PCIe(pinned/UVA·cold slab) launch의 **블록 수 상한**. PCIe 바운드 커널은 SM을
# 다 채울 이유가 없고, 비워둔 SM에 hot 커널이 동시에 올라가야 hot∥warm 스트림
# 분리가 실제로 겹친다. 2026-08-27 H100 실측 (35B h50/w50 gateup, M=2048): 128
# 블록으로도 PCIe 51 GB/s 포화 유지 + hot 2.64 ms가 **완전히** 겹침(합 13.1 →
# 10.5 ms); 256블록이면 겹침 1.8, 512는 0.9, 무제한은 0. 0 = 상한 없음.
WARM_MAX_BLOCKS = 128

# W-resident 커널(pair 수와 무관하게 expert당 W를 1회만 읽음)을 PCIe 경로에 쓸 것인가.
# wres_k_max>0 이면 그 커널로 간다 — 값은 스토어의 expert당 최대 K 행(32 배수).
WRES_PCIE = True


def _scratch(out3d, wres_k_max):
    """K-chunk 누산용 fp32 scratch [P, out_row]. wres가 아니면 빈 텐서(커널이 무시)."""
    if not wres_k_max:
        return torch.empty(0, dtype=torch.float32, device=out3d.device)
    m, k, w_row = out3d.shape
    return torch.empty(m * k, w_row, dtype=torch.float32, device=out3d.device)


@cache_once
def _jit_prism_grouped_module() -> Module:
    return load_jit(
        "prism_grouped",
        cuda_files=["moe/prism_grouped.cuh"],
        cuda_wrappers=[
            ("grouped_gemm_indexed", "grouped_gemm_indexed"),
            ("grouped_gemm_indexed_pinned", "grouped_gemm_indexed_pinned"),
            ("grouped_gemm_indexed_gateup", "grouped_gemm_indexed_gateup"),
            ("grouped_gemm_indexed_pinned_gateup",
             "grouped_gemm_indexed_pinned_gateup"),
            ("grouped_gemm_cold", "grouped_gemm_cold"),
            ("grouped_gemm_cold_gateup", "grouped_gemm_cold_gateup"),
            ("prism_wait_flag", "prism_wait_flag"),
        ],
    )


def warmup_jit() -> None:
    """모듈을 강제 컴파일한다 (startup에서 — 첫 prefill이 JIT을 기다리지 않게)."""
    _jit_prism_grouped_module()


def grouped_gemm_indexed(x2d, grouping, w_flat, row_off, kidx, out3d,
                         out_col_offset, x_row_is_pair, stream, max_blocks=0, wres_k_max=0) -> None:
    """out3d[m, j, off:off+N] = x_row · W[topk[m,j]] — device W (HOT), expert 그룹 GEMM.

    `grouping`은 `build_grouping(topk_ids, E)`의 결과다. x_row 규약은 worklist와
    같다: gateup은 hidden [M, K](False), down은 act.view(M·top_k, inter)(True).
    """
    module = _jit_prism_grouped_module()
    with torch.cuda.stream(stream):
        module.grouped_gemm_indexed(
            x2d, grouping.pair_sorted, grouping.pair_off, grouping.tile_off,
            w_flat, row_off, kidx, out3d, int(out_col_offset), int(bool(x_row_is_pair)),
            int(max_blocks), int(wres_k_max), _scratch(out3d, wres_k_max))


def grouped_gemm_indexed_pinned(x2d, grouping, w_flat, row_off, kidx, out3d,
                         out_col_offset, x_row_is_pair, stream, max_blocks=0, wres_k_max=0) -> None:
    """위의 쌍둥이 — W가 pinned CPU(UVA 제자리 읽기, WARM). expert당 W를 **한 번만**
    PCIe로 읽는다 — worklist의 pair당 재읽기(중복도 M·k/E)가 사라진 자리다."""
    module = _jit_prism_grouped_module()
    with torch.cuda.stream(stream):
        module.grouped_gemm_indexed_pinned(
            x2d, grouping.pair_sorted, grouping.pair_off, grouping.tile_off,
            w_flat, row_off, kidx, out3d, int(out_col_offset), int(bool(x_row_is_pair)),
            int(max_blocks), int(wres_k_max), _scratch(out3d, wres_k_max))


def grouped_gemm_indexed_gateup(x2d, grouping, w_gate, row_off_gate, kidx_gate,
                                w_up, row_off_up, kidx_up, out3d,
                                out_col_gate, out_col_up, x_row_is_pair,
                                stream, max_blocks=0, wres_k_max=0) -> None:
    """gate와 up을 한 launch로 (device W). blockIdx.z가 슬롯을 고른다."""
    module = _jit_prism_grouped_module()
    with torch.cuda.stream(stream):
        module.grouped_gemm_indexed_gateup(
            x2d, grouping.pair_sorted, grouping.pair_off, grouping.tile_off,
            w_gate, row_off_gate, kidx_gate, w_up, row_off_up, kidx_up, out3d,
            int(out_col_gate), int(out_col_up), int(bool(x_row_is_pair)),
            int(max_blocks), int(wres_k_max), _scratch(out3d, wres_k_max))


def grouped_gemm_indexed_pinned_gateup(x2d, grouping, w_gate, row_off_gate,
                                       kidx_gate, w_up, row_off_up, kidx_up,
                                       out3d, out_col_gate, out_col_up,
                                       x_row_is_pair, stream, max_blocks=0, wres_k_max=0) -> None:
    """gate+up 융합의 warm 짝 (pinned W)."""
    module = _jit_prism_grouped_module()
    with torch.cuda.stream(stream):
        module.grouped_gemm_indexed_pinned_gateup(
            x2d, grouping.pair_sorted, grouping.pair_off, grouping.tile_off,
            w_gate, row_off_gate, kidx_gate, w_up, row_off_up, kidx_up, out3d,
            int(out_col_gate), int(out_col_up), int(bool(x_row_is_pair)),
            int(max_blocks), int(wres_k_max), _scratch(out3d, wres_k_max))


def grouped_gemm_cold(x2d, grouping, cold, out3d, out_col_offset, x_row_is_pair,
                      stream, max_blocks=0, wres_k_max=0) -> None:
    """kt AMX packed slab(cudaHostRegister된 host 메모리)을 **제자리에서** 읽는
    grouped GEMM. `cold`는 `ColdSlab`(weights.py) — slab 텐서 뷰, blk_off, 타일
    올림된 row_off/kidx, 노드 N shard 시작, n_block/k_block을 든다.
    출력 열은 `out_col_offset + cold.n_start`부터 이 노드의 n열이다."""
    module = _jit_prism_grouped_module()
    with torch.cuda.stream(stream):
        module.grouped_gemm_cold(
            x2d, grouping.pair_sorted, grouping.pair_off, grouping.tile_off,
            cold.slab, cold.blk_off, cold.row_off, cold.k_index, out3d,
            int(out_col_offset) + int(cold.n_start), int(bool(x_row_is_pair)),
            int(max_blocks), int(cold.n_block), int(cold.k_block), int(cold.n), int(wres_k_max),
            _scratch(out3d, wres_k_max))


def grouped_gemm_cold_gateup(x2d, grouping, gate, up, out3d, out_col_gate,
                             out_col_up, x_row_is_pair, stream, max_blocks=0, wres_k_max=0) -> None:
    """cold gate+up 한 launch (같은 노드의 두 slab). 두 slab의 n/n_start/블록
    상수는 같아야 한다 — 같은 kt 인스턴스가 만든 것이므로 그렇다."""
    if (gate.n_start, gate.n_block, gate.k_block) != (up.n_start, up.n_block, up.k_block):
        raise ValueError("cold gateup fusion requires gate/up slabs of the same node instance")
    module = _jit_prism_grouped_module()
    with torch.cuda.stream(stream):
        module.grouped_gemm_cold_gateup(
            x2d, grouping.pair_sorted, grouping.pair_off, grouping.tile_off,
            gate.slab, gate.blk_off, gate.row_off, gate.k_index,
            up.slab, up.blk_off, up.row_off, up.k_index, out3d,
            int(out_col_gate) + int(gate.n_start), int(out_col_up) + int(up.n_start),
            int(bool(x_row_is_pair)), int(max_blocks), int(gate.n_block), int(gate.k_block),
            int(gate.n), int(wres_k_max), _scratch(out3d, wres_k_max))


def wait_flag(flag_pinned: torch.Tensor, target: int, stream) -> None:
    """현재 stream이 pinned int32 플래그가 target 이상이 될 때까지 기다린다 (1-thread 커널).
    cold_async: kt의 signal 태스크가 cold 완료 후 플래그를 쓴다."""
    module = _jit_prism_grouped_module()
    with torch.cuda.stream(stream):
        module.prism_wait_flag(flag_pinned, int(target))

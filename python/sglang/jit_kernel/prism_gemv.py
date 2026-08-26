from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_prism_gemv_module() -> Module:
    return load_jit(
        "prism_gemv",
        cuda_files=["moe/prism_gemv.cuh"],
        cuda_wrappers=[
            ("gemv_worklist", "gemv_worklist"),
            ("gemv_worklist_pinned", "gemv_worklist_pinned"),
            ("gemv_worklist_indexed", "gemv_worklist_indexed"),
            ("gemv_worklist_indexed_gateup", "gemv_worklist_indexed_gateup"),
            ("gemv_worklist_indexed_pinned_sparse_gateup",
             "gemv_worklist_indexed_pinned_sparse_gateup"),
            ("gemv_worklist_indexed_pinned", "gemv_worklist_indexed_pinned"),
            ("gemv_worklist_indexed_sparse", "gemv_worklist_indexed_sparse"),
            ("gemv_worklist_indexed_pinned_sparse",
             "gemv_worklist_indexed_pinned_sparse"),
        ],
    )


def warmup_jit() -> None:
    """모듈을 강제 컴파일한다 (public 래퍼 — `_jit_prism_gemv_module`은 private).

    호출자(method.py)가 startup 경로에서 이걸 불러 lazy JIT을 캡처 워밍업
    이전으로 앞당긴다 — 캡처 순서 의존 제거."""
    _jit_prism_gemv_module()


def gemv_worklist(x2d, topk_ids, weights, out3d, k_offset, out_col_offset,
                  x_row_is_pair, stream) -> None:
    """out3d[m, j, off:off+N] = x_row · W[topk_ids[m,j]] — device-resident W (HOT).

    누산 fp32/출력 bf16 (계약 ⑤). stream 규약은 이 JIT 모듈 공통: 커널은
    호출 시점의 current stream에 오르므로 래퍼가 stream 컨텍스트를 강제한다.
    x_row = pair(m*top_k+j) if x_row_is_pair else m — gateup은 hidden[M,K]
    (False), down은 act.view(M*top_k, inter) (True).
    """
    module = _jit_prism_gemv_module()
    with torch.cuda.stream(stream):
        module.gemv_worklist(x2d, topk_ids, weights, out3d,
                             int(k_offset), int(out_col_offset), int(bool(x_row_is_pair)))


def gemv_worklist_pinned(x2d, topk_ids, weights, out3d, k_offset, out_col_offset,
                         x_row_is_pair, stream) -> None:
    """gemv_worklist의 쌍둥이 — W가 pinned CPU(UVA 직접 읽기, WARM). 배치 내
    중복 expert는 PCIe 재전송이다 (설계 결정 — spec §1.2)."""
    module = _jit_prism_gemv_module()
    with torch.cuda.stream(stream):
        module.gemv_worklist_pinned(x2d, topk_ids, weights, out3d,
                                    int(k_offset), int(out_col_offset), int(bool(x_row_is_pair)))


def gemv_worklist_indexed(x2d, topk_ids, w_flat, row_off, kidx, out3d,
                          out_col_offset, x_row_is_pair, stream, vec=0) -> None:
    """인덱스 변형 — 티어 멤버십이 밴드가 아니라 **가변 per-expert 인덱스**인 경우.

    `gemv_worklist`와 다른 것은 두 가지뿐이다:
      - W가 flat `[Σₑ k[e], N]`이고 expert 구간을 `row_off[e] ..= row_off[e+1]`이 준다
      - activation 열을 `kidx`가 준다 (`x[row, kidx[o0 + r]]`)
    `k_offset`/`k_rows` 인자가 사라진 자리가 그 둘이다. grid는 같다 — k는 루프라
    expert마다 길이가 달라도 launch 모양이 안 변한다.

    row_off(int32 [E+1])와 kidx(uint16 [Σₑ k[e]])는 **항상 device 상주**여야
    한다. W만 pinned일 수 있다 (아래 쌍둥이).

    vec: W 로드 폭 (열/스레드). 0 = 자동(정렬이 허용하는 최대), 1/4/8 강제.
    강제는 벤치·디버그용이다.

    연속 인덱스(밴드 퇴화형)에서는 읽는 원소도 누산 순서도 `gemv_worklist`와
    같으므로 **비트일치**한다 — 그것이 전환기의 합격 기준이다.
    """
    module = _jit_prism_gemv_module()
    with torch.cuda.stream(stream):
        module.gemv_worklist_indexed(x2d, topk_ids, w_flat, row_off, kidx, out3d,
                                     int(out_col_offset), int(bool(x_row_is_pair)),
                                     int(vec))


def gemv_worklist_indexed_pinned(x2d, topk_ids, w_flat, row_off, kidx, out3d,
                                 out_col_offset, x_row_is_pair, stream, vec=0) -> None:
    """gemv_worklist_indexed의 쌍둥이 — W가 pinned CPU(UVA 직접 읽기, WARM)."""
    module = _jit_prism_gemv_module()
    with torch.cuda.stream(stream):
        module.gemv_worklist_indexed_pinned(x2d, topk_ids, w_flat, row_off, kidx,
                                            out3d, int(out_col_offset),
                                            int(bool(x_row_is_pair)), int(vec))


def _sparse_call(fn, x2d, topk_ids, topk_weights, w_flat, row_off, kidx, out3d,
                 sp, out_col_offset, x_row_is_pair, stream, vec) -> None:
    """sparse 두 래퍼의 공통 본체 — 갈리는 것은 FFI 심볼 하나뿐이다.

    `topk_weights`가 `sp`에 들어가지 않는 이유: sp는 로드 타임에 한 번 만들어
    영구 보관하는 물건이고(a/c/thr는 device 상주 상수), 라우터 가중은 스텝마다
    갈린다. 섞으면 스텝마다 spec을 복제하거나 공유 객체를 변조해야 한다."""
    with torch.cuda.stream(stream):
        fn(x2d, topk_ids, w_flat, row_off, kidx, out3d,
           sp.a, sp.c, sp.thr, topk_weights,
           int(out_col_offset), int(bool(x_row_is_pair)), int(vec),
           float(sp.p), float(sp.lam), float(sp.pmax), float(sp.grid),
           int(sp.ng), int(sp.renorm_it))


def gemv_worklist_indexed_sparse(x2d, topk_ids, topk_weights, w_flat, row_off,
                                 kidx, out3d, sp, out_col_offset,
                                 x_row_is_pair, stream, vec=0) -> None:
    """gemv_worklist_indexed의 sparse 변형 — 죽은 페어의 W 로드를 발행하지 않는다.

    `sp`는 점수 재료와 예산을 담은 객체다 (a=wn², c=pair_dot, thr 곡선,
    topk_weights, p/lam/pmax/grid/ng/renorm_it). threshold는 **커널이 직접
    계산한다** — CPU가 계산한 값을 받으면 스텝마다 device sync가 생겨 CUDA
    graph가 깨지고, 어차피 같은 순수 함수라 양쪽이 독립 계산해도 같은 값이다.

    누산 순서는 dense와 같다 (마스크는 W 로드를 건너뛸 뿐 루프 모양을 바꾸지
    않는다) — 그래서 전부 keep인 경우 dense와 **비트일치**한다.
    """
    module = _jit_prism_gemv_module()
    _sparse_call(module.gemv_worklist_indexed_sparse, x2d, topk_ids,
                 topk_weights, w_flat, row_off, kidx, out3d, sp,
                 out_col_offset, x_row_is_pair, stream, vec)


def gemv_worklist_indexed_pinned_sparse(x2d, topk_ids, topk_weights, w_flat,
                                        row_off, kidx, out3d, sp,
                                        out_col_offset, x_row_is_pair, stream,
                                        vec=0) -> None:
    """위의 쌍둥이 — W가 pinned CPU(WARM). 건너뛴 로드가 그대로 PCIe 절약이다."""
    module = _jit_prism_gemv_module()
    _sparse_call(module.gemv_worklist_indexed_pinned_sparse, x2d, topk_ids,
                 topk_weights, w_flat, row_off, kidx, out3d, sp,
                 out_col_offset, x_row_is_pair, stream, vec)


def gemv_worklist_indexed_gateup(x2d, topk_ids, w_gate, row_off_gate, kidx_gate,
                                 w_up, row_off_up, kidx_up, out3d,
                                 out_col_gate, out_col_up, x_row_is_pair,
                                 stream, vec=0) -> None:
    """gate와 up을 **한 커널로** 발행한다 (device 상주 W, dense).

    두 proj는 x·topk_ids·출력 버퍼를 공유하고 W·인덱스·출력 열 구간만 다르므로,
    `blockIdx.z`가 그 셋을 고른다. 얻는 것은 grid.z로 블록이 2배가 되는 것이다 —
    bs=1의 이 커널은 블록에 굶어 있어(96블록 / 114 SM) 그것이 곧 성능이다.
    출력 원소당 누산 순서가 불변이라 두 번 launch한 것과 비트일치한다.
    """
    module = _jit_prism_gemv_module()
    with torch.cuda.stream(stream):
        module.gemv_worklist_indexed_gateup(
            x2d, topk_ids, w_gate, row_off_gate, kidx_gate,
            w_up, row_off_up, kidx_up, out3d,
            int(out_col_gate), int(out_col_up), int(bool(x_row_is_pair)),
            int(vec))


def gemv_worklist_indexed_pinned_sparse_gateup(
        x2d, topk_ids, topk_weights, w_gate, row_off_gate, kidx_gate,
        w_up, row_off_up, kidx_up, out3d, sp_gate, sp_up,
        out_col_gate, out_col_up, x_row_is_pair, stream, vec=0) -> None:
    """warm의 gate+up 융합 (pinned W, sparse).

    두 proj가 x·topk_ids·라우터 가중·출력 버퍼를 공유하고 점수 재료만 갈린다.
    예산 스칼라(pmax/grid/ng/renorm_it)는 plan 단위라 슬롯 0의 값을 쓰며, 두
    spec이 다르면 C++가 아니라 여기서 거절한다 — 공유 전제를 호출부에서 지킨다.
    """
    for f in ("pmax", "grid", "ng", "renorm_it"):
        if getattr(sp_gate, f) != getattr(sp_up, f):
            raise ValueError(f"gateup fusion requires a shared sparsity budget; "
                             f"{f} differs ({getattr(sp_gate, f)} vs {getattr(sp_up, f)})")
    module = _jit_prism_gemv_module()
    with torch.cuda.stream(stream):
        module.gemv_worklist_indexed_pinned_sparse_gateup(
            x2d, topk_ids, w_gate, row_off_gate, kidx_gate,
            w_up, row_off_up, kidx_up, out3d,
            sp_gate.a, sp_gate.c, sp_gate.thr,
            sp_up.a, sp_up.c, sp_up.thr, topk_weights,
            int(out_col_gate), int(out_col_up), int(bool(x_row_is_pair)), int(vec),
            float(sp_gate.p), float(sp_gate.lam), float(sp_up.p), float(sp_up.lam),
            float(sp_gate.pmax), float(sp_gate.grid),
            int(sp_gate.ng), int(sp_gate.renorm_it))

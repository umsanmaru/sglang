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

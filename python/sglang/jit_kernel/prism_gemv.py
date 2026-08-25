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

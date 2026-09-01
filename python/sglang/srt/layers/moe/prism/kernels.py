"""(이행 shim) 커널 이름표는 `layers/prism/kernels.py`로 승격됐다 (2026-08-31).

이름·정렬·타일·slab 레이아웃은 expert 축과 무관해 dense와 공유한다. 여기 남는
실체는 하나뿐이다: **태그 → StoreFormat 객체** 해석. 포맷 객체는 파라미터 이름과
full 텐서 인출을 들고 있고 그것이 MoE(w13/w2)와 dense(weight)에서 갈리므로,
공유는 태그까지이고 객체는 각자다.

기존 import 경로를 살려두는 것은 42개 파일을 건드리지 않기 위해서다 — 새 코드는
`sglang.srt.layers.prism.kernels`를 직접 쓸 것.
"""

from sglang.srt.layers.prism.kernels import (  # noqa: F401
    KernelError,
    cold_n_align,
    cold_pack_tile_rows,
    cold_slab_layout,
    gpu_store_format_tag,
    known_cpu_kernels,
    known_gpu_kernels,
    resolve_cpu_kernel,
    resolve_gpu_kernel,
)


def gpu_store_format(name: str):
    """GPU 커널 키 → MoE `StoreFormat` 객체 (이름 검증 겸).

    런타임 분기는 이 객체의 메서드다. dense 오프로드는 같은 태그를 자기
    포맷 표로 해석한다 (`layers/prism/linear/`).
    """
    from sglang.srt.layers.moe.prism.formats import FORMATS

    return FORMATS[gpu_store_format_tag(name)]

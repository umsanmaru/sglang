"""Compatibility facade and process-global storage-format instances.

Implementations live in store_formats; historical class/function imports and
pickle module names remain available here. No loading algorithm or ownership
policy changes are part of this extraction.
"""
from .store_formats.base import FullWeights, ProjSource, StoreFormat
from .store_formats.bf16 import Bf16Format
from .store_formats.mxfp4 import Mxfp4Format
from .store_formats.fp8 import Fp8Format
from .store_formats.loading import (
    _host,
    _check_shapes,
    _proj_view,
    _e8m0_bytes,
    _half_index,
    _gather_rows,
    _cold_flat_rows,
    _tensor_view,
    _padded_kindex,
)

# Preserve historical serialized class names and one registry instance per format.
for _class in (FullWeights, ProjSource, StoreFormat, Bf16Format, Mxfp4Format, Fp8Format):
    _class.__module__ = __name__
del _class

BF16 = Bf16Format()
MXFP4 = Mxfp4Format()
FP8 = Fp8Format()
FORMATS = {BF16.name: BF16, MXFP4.name: MXFP4, FP8.name: FP8}

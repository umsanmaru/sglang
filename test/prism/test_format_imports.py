"""Legacy imports, singleton registry and serialization survive module moves."""
import pickle

import torch

from sglang.srt.layers.moe.prism import formats as F
from sglang.srt.layers.moe.prism.store_formats.base import FullWeights, StoreFormat
from sglang.srt.layers.moe.prism.store_formats.bf16 import Bf16Format
from sglang.srt.layers.moe.prism.store_formats.fp8 import Fp8Format
from sglang.srt.layers.moe.prism.store_formats.mxfp4 import Mxfp4Format


def test_legacy_classes_and_registry_identity():
    assert F.FullWeights is FullWeights and F.StoreFormat is StoreFormat
    for cls, obj in [(Bf16Format,F.BF16),(Mxfp4Format,F.MXFP4),(Fp8Format,F.FP8)]:
        assert isinstance(obj,cls)
        assert F.FORMATS[obj.name] is obj
        assert cls.__module__ == F.__name__
        assert type(pickle.loads(pickle.dumps(obj))) is cls


def test_checkpoint_views_and_old_pickle_names():
    a=torch.arange(16,dtype=torch.float32).reshape(1,4,4)
    weights=FullWeights(a,a)
    assert weights.w13 is a and weights.w2 is a
    restored=pickle.loads(pickle.dumps(weights))
    assert type(restored) is F.FullWeights
    assert restored.w13 is restored.w2
    torch.testing.assert_close(restored.w13,a)

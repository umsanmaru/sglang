"""Warm KT packing must not keep the Python source weights resident."""
from types import SimpleNamespace as NS
import weakref

import pytest
import torch

from sglang.srt.layers.moe.prism import executor as module


@pytest.mark.parametrize('build_fails', [False, True])
def test_warm_source_released_after_tier_construction(monkeypatch, build_fails):
    sources = [torch.ones(32) for _ in range(3)]
    refs = [weakref.ref(t) for t in sources]
    calib = [object() for _ in range(3)]
    pending = NS(**{name: NS(w_flat=t, calib=c)
                    for name, t, c in zip(('gate', 'up', 'down'), sources, calib)})
    prepared = NS(warm_kt=pending)
    del sources, pending
    view = object()
    executor = object.__new__(module.PrismExecutor)
    executor._plan = NS(expert=lambda *_: NS(proj=lambda _: NS(has_tier=lambda _: False)))
    executor._cold = NS(warm_view=lambda _: view)
    executor._layers = {}
    executor._tiers = {}
    executor._cold_gpu_min_m = None
    executor._layer_has_cold = {}
    executor._layer_cold_gpu = {}
    executor._layer_warm_kt = {}
    executor._mask_stats = None

    def build(*args, warm_kt, warm_kt_calib, **kwargs):
        assert all(ref() is not None for ref in refs)
        assert warm_kt is view
        assert list(warm_kt_calib.values()) == calib
        if build_fails:
            raise RuntimeError('tier construction failed')
        return NS(view=warm_kt, calib=warm_kt_calib)

    monkeypatch.setattr(module, 'build_layer_tiers', build)
    if build_fails:
        with pytest.raises(RuntimeError, match='tier construction failed'):
            executor.register_layer(0, prepared)
        assert prepared.warm_kt is not None
    else:
        executor.register_layer(0, prepared)
        assert prepared.warm_kt is None
        assert all(ref() is None for ref in refs)
        assert executor._tiers[0].view is view
        assert list(executor._tiers[0].calib.values()) == calib
        assert executor._layer_warm_kt[0]

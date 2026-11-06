"""CPU lifecycle contracts for the context extraction and legacy adapter."""
from types import SimpleNamespace

import pytest

from sglang.srt.layers.moe.prism import method as M
from sglang.srt.layers.moe.prism import kernels
from sglang.srt.layers.moe.prism.runtime.context import PrismRuntime


def make_runtime(monkeypatch):
    fmt = SimpleNamespace()
    monkeypatch.setattr(kernels, 'gpu_store_format', lambda key: fmt)
    plan = SimpleNamespace(kernels=SimpleNamespace(gpu_warm='test'), experts=[])
    return M._PrismRuntime(plan), fmt


def test_constructor_is_lazy_and_retains_legacy_type(monkeypatch):
    runtime, fmt = make_runtime(monkeypatch)
    assert isinstance(runtime, PrismRuntime)
    assert runtime.fmt is fmt
    assert runtime._resources is runtime._cold is runtime._executor is None
    assert runtime.activation is runtime.routed_scaling_factor is None
    assert not runtime.has_cold and not runtime.has_warm


def test_cached_objects_skip_all_initialization(monkeypatch):
    runtime, _ = make_runtime(monkeypatch)
    executor, backend = object(), object()
    runtime._executor, runtime._cold = executor, backend
    monkeypatch.setattr(M, '_CONFIG', None)
    assert runtime.executor(None) is executor
    assert runtime.cold() is backend


def test_live_bindings_and_capture_function_resolution(monkeypatch):
    runtime, _ = make_runtime(monkeypatch)
    replacement = object()
    monkeypatch.setattr(M, '_CONFIG', replacement)
    assert runtime._bindings.config() is replacement
    capture = lambda: True
    monkeypatch.setattr(M, '_sglang_capture_mode', capture)
    assert runtime._bindings.capture_mode_fn() is capture
    monkeypatch.setattr(M, '_cold_gpu_min_m', lambda: 17)
    assert runtime._bindings.cold_gpu_min_m() == 17
    monkeypatch.setattr(M, '_hybrid_local_node', lambda device: device + 1)
    assert runtime._bindings.hybrid_local_node(2) == 3


def test_invalid_constructor_config_fails_at_original_boundary(monkeypatch):
    monkeypatch.setenv('SGLANG_PRISM_MAX_TOKENS', 'invalid')
    with pytest.raises(ValueError):
        make_runtime(monkeypatch)

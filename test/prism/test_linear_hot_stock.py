"""Hot-only plans retain the native fused linear and loader; mixed plans do not."""
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.prism.geometry import PlanError
from sglang.srt.layers.prism.linear import method
from sglang.srt.layers.prism.linear.plan import parse_plan, validate_static


def runtime(monkeypatch, hot=True):
    bands = [[0, 256, 'hot']] if hot else [[0, 128, 'hot'], [128, 256, 'warm']]
    plan = parse_plan(dict(schema_version=1, model_id='test',
        dims=dict(num_layers=2, dtype='bfloat16'),
        kernels=dict(gpu_warm='gemv_worklist', cpu_cold='kt_tile_k2_bf16'),
        projs={'mlp.gate_up_proj':dict(k=256,n=512,parts=[
            dict(name='gate',n=256,bands=[[0,256,'hot']]),
            dict(name='up',n=256,bands=bands)])}))
    validate_static(plan)
    rt = method._LinearRuntime(plan)
    monkeypatch.setattr(method, '_RUNTIME', rt)
    monkeypatch.setenv('SGLANG_PRISM_LINEAR_HOT_STOCK', '1')
    return rt


@pytest.mark.parametrize('enabled,hot,dtype,expected', [
    (True, True, torch.bfloat16, True),
    (False, True, torch.bfloat16, False),
    (True, False, torch.bfloat16, False),
    (True, True, torch.float16, False)])
def test_native_only_for_complete_bf16_hot(monkeypatch, enabled, hot, dtype, expected):
    rt = runtime(monkeypatch, hot)
    monkeypatch.setenv('SGLANG_PRISM_LINEAR_HOT_STOCK', str(int(enabled)))
    inner = type('UnquantizedLinearMethod', (), {})()
    layer = SimpleNamespace(params_dtype=dtype,input_size=256,output_size=512)
    got = method._factory(layer, inner, dict(layer_idx=1,name='mlp.gate_up_proj'))
    assert (got is inner) == expected
    assert rt._stock_registered == ({(1,'mlp.gate_up_proj')} if expected else set())


def test_stock_still_checks_plan_dimensions(monkeypatch):
    runtime(monkeypatch)
    inner = type('UnquantizedLinearMethod', (), {})()
    layer = SimpleNamespace(params_dtype=torch.bfloat16,input_size=128,output_size=512)
    with pytest.raises(PlanError):
        method._factory(layer, inner, dict(layer_idx=1,name='mlp.gate_up_proj'))


def test_stock_and_split_share_coverage(monkeypatch, caplog):
    rt = runtime(monkeypatch)
    rt._stock_registered.add((1,'mlp.gate_up_proj'))
    rt._executor = SimpleNamespace(registered=lambda:frozenset({(0,'mlp.gate_up_proj')}))
    with caplog.at_level('INFO'):
        rt.check_coverage()
    assert 'stock hot-only projections: 1' in caplog.text
    assert 'mlp.gate_up_proj×2' in caplog.text


def test_all_stock_coverage_without_executor(monkeypatch, caplog):
    rt = runtime(monkeypatch)
    rt._stock_registered.add((0,'mlp.gate_up_proj'))
    with caplog.at_level('INFO'):
        rt.check_coverage()
    assert rt._checked
    assert 'stock hot-only projections: 1' in caplog.text

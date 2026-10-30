"""Loading gather equivalence, cache invalidation, corruption and size limits."""
import json
from dataclasses import replace

import pytest
import torch

from sglang.srt.layers.moe.prism.index import TierIndex
from sglang.srt.layers.moe.prism.loading_gather import gather_padded_columns
from sglang.srt.layers.moe.prism.loading_cache import load_plan_cached, load_layer_index


def reference(src, ti, real, tile, div=1, value=0, minimum=None):
    pad = [max((r + tile - 1) // tile * tile, 0 if minimum is None else minimum[e])
           for e, r in enumerate(real)]
    offsets = [0]
    blocks, indices = [], []
    for e, (r, k) in enumerate(zip(real, pad)):
        rows = ti.for_expert(e)[:r // div].long()
        b = torch.full((src.shape[1], k // div), value, dtype=src.dtype)
        b[:, :r // div] = src[e].index_select(1, rows)
        blocks.append(b.flatten())
        ix = torch.zeros(k, dtype=torch.int64)
        if div == 1:
            ix[:r] = rows
        indices.append(ix)
        offsets.append(offsets[-1] + k)
    return (torch.cat(blocks), torch.tensor(offsets, dtype=torch.int32),
            torch.cat(indices), torch.tensor(real, dtype=torch.int32))


@pytest.mark.parametrize('div,dtype,value', [(1, torch.bfloat16, 0), (2, torch.uint8, 136),
                                           (32, torch.float32, 0)])
@pytest.mark.parametrize('workspace', [1, 4096, 1 << 24])
def test_variable_gather(div, dtype, value, workspace):
    torch.manual_seed(9)
    real = [0, 32, 96, 64, 32, 128, 0]
    src = torch.randint(0, 127, (7, 64, 128 // div)).to(dtype)
    # A noncontiguous outer expert stride, as in the gate view of w13.
    source = torch.empty(7, 128, 128 // div, dtype=dtype)
    source[:, :64].copy_(src);src = source[:, :64]
    rows = [torch.randperm(128 // div)[:r // div] for r in real]
    ti = TierIndex.from_rows(rows)
    minimum = [32, 0, 0, 0, 0, 0, 0]  # one-sided empty warm GU sentinel
    actual = gather_padded_columns(src, ti, real, 32, div=div, pad_value=value,
                                   minimum_rows=minimum, workspace_bytes=workspace)
    expected = reference(src, ti, real, 32, div, value, minimum)
    for a, b in zip(actual, expected):
        assert torch.equal(a, b)


def test_tail_padding_and_all_zero():
    src = torch.randn(4, 32, 64, dtype=torch.bfloat16)
    real = [2, 34, 0, 62]
    ti = TierIndex.from_rows([torch.arange(r) for r in real])
    for a, b in zip(gather_padded_columns(src, ti, real, 32), reference(src, ti, real, 32)):
        assert torch.equal(a, b)
    zero = TierIndex(torch.zeros(5, dtype=torch.int32), torch.empty(0, dtype=torch.uint16), True)
    result = gather_padded_columns(src, zero, [0] * 4, 32)
    assert result[0].numel() == result[2].numel() == 0


def plan_file(tmp_path):
    raw = {'schema_version': 1, 'model_id': 'cache-test',
           'dims': {'hidden_size': 64, 'intermediate_size': 64, 'num_experts': 2,
                    'num_layers': 1, 'top_k': 1, 'dtype': 'bfloat16'},
           'kernels': {'gpu_warm': 'gemv_worklist', 'cpu_cold': 'kt_amx_bf16'},
           'default': {p: {'bands': [[0, 64, 'warm']]} for p in ['gate', 'up', 'down']}}
    path = tmp_path / 'plan.json';path.write_text(json.dumps(raw))
    return path, raw


def test_plan_and_index_cache(tmp_path, monkeypatch):
    from sglang.srt.layers.moe.prism import plan as module, index as index_module
    monkeypatch.setenv('SGLANG_PRISM_LOAD_CACHE_DIR', str(tmp_path / 'cache'))
    path, raw = plan_file(tmp_path)
    first = load_plan_cached(path)
    indices = load_layer_index(first, 0)
    parse = module.parse_plan
    def fail(*args):
        raise AssertionError('cache missed')
    monkeypatch.setattr(module, 'parse_plan', fail)
    monkeypatch.setattr(index_module, 'from_bands', fail)
    second = load_plan_cached(path)
    assert first == second
    for key, ti in indices.tiers.items():
        other = load_layer_index(second, 0).tiers[key]
        assert (other is None) == (ti is None)
        if ti is not None:
            assert torch.equal(ti.idx, other.idx)
    monkeypatch.setattr(module, 'parse_plan', parse)
    raw['model_id'] = 'changed';path.write_text(json.dumps(raw))
    changed = load_plan_cached(path)
    assert changed.model_id == 'changed'
    assert changed._loading_cache_key != first._loading_cache_key
    for cache in (tmp_path / 'cache').rglob('*.plan'):
        cache.write_bytes(b'broken')
    assert load_plan_cached(path).model_id == 'changed'


def test_weight_cache_uses_actual_bytes(tmp_path, monkeypatch):
    from sglang.srt.layers.moe.prism import loading_gather as module
    monkeypatch.setenv('SGLANG_PRISM_WEIGHT_CACHE_DIR', str(tmp_path))
    src = torch.randn(2, 32, 64, dtype=torch.bfloat16)
    ti = TierIndex.from_rows([torch.arange(32), torch.arange(32)])
    first = gather_padded_columns(src, ti, [32, 32], 32)
    compute = module._gather_padded_columns
    def fail(*args, **kwargs):
        raise AssertionError('weight cache missed')
    monkeypatch.setattr(module, '_gather_padded_columns', fail)
    second = gather_padded_columns(src, ti, [32, 32], 32)
    assert torch.equal(first[0], second[0])
    monkeypatch.setattr(module, '_gather_padded_columns', compute)
    src[0, 0, 0] += 1
    third = gather_padded_columns(src, ti, [32, 32], 32)
    assert not torch.equal(first[0], third[0])
    assert len(list(tmp_path.rglob('*.pt'))) == 2
    # Cold bytes may be the same while the K-index order changes.
    shuffled = replace(ti, idx=ti.idx.long().flip(0).to(torch.uint16))
    fourth = gather_padded_columns(src, shuffled, [32, 32], 32)
    assert not torch.equal(third[0], fourth[0])


def test_weight_cache_budget_and_corruption(tmp_path, monkeypatch):
    monkeypatch.setenv('SGLANG_PRISM_WEIGHT_CACHE_DIR', str(tmp_path))
    monkeypatch.setenv('SGLANG_PRISM_WEIGHT_CACHE_MAX_GB', str(18000 / (1 << 30)))
    src = torch.ones(2, 32, 64, dtype=torch.bfloat16)
    ti = TierIndex.from_rows([torch.arange(32)] * 2)
    gather_padded_columns(src, ti, [32, 32], 32)
    src.add_(1)
    expected = gather_padded_columns(src, ti, [32, 32], 32)
    files = list(tmp_path.rglob('*.pt'))
    assert len(files) == 1
    files[0].write_bytes(b'broken')
    actual = gather_padded_columns(src, ti, [32, 32], 32)
    assert torch.equal(actual[0], expected[0])


def test_contiguous_bands_with_different_starts():
    from sglang.srt.layers.moe.prism.formats import _cold_flat_rows
    src = torch.arange(3 * 32 * 64).reshape(3, 32, 64).to(torch.bfloat16)
    ti = TierIndex.from_rows([torch.arange(0, 32), torch.arange(32, 64), torch.arange(16, 48)])
    assert ti.contiguous
    for actual, expected in zip(_cold_flat_rows(src, ti, [32] * 3, 32),
                                reference(src, ti, [32] * 3, 32)):
        assert torch.equal(actual, expected)

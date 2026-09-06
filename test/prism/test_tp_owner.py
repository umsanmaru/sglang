"""MoE-TP에서의 prism 배치 — owner(rank 0)만 expert를 갖고 나머지 rank는 0을 낸다.

GPU 2장에 dense(attention/shared expert/latent proj)를 TP로, routed expert는 CPU(+owner GPU 티어)에
두는 K3 구도(2026-09-06). e2e는 GPU가 둘 필요해 이 박스에서 못 돌고(H100 박스에서 DSV4 TP=2로 검증),
여기서는 get_parallel을 흉내내 method의 결정과 로더 필터를 본다. FusedMoE.weight_loader hook
(prism_tp_mode "skip"/"full")은 e2e 몫이다.
"""
import json
import os
from types import SimpleNamespace

import pytest
import torch

import sglang.srt.runtime_context as rc
from sglang.srt.layers.moe.prism import method as M

DIMS = {"hidden_size": 256, "intermediate_size": 128, "num_layers": 1,
        "num_experts": 8, "top_k": 2, "dtype": "bfloat16"}


def _plan_file(tmp_path):
    raw = {
        "schema_version": 1, "model_id": "test/tp", "dims": dict(DIMS),
        "kernels": {"gpu_warm": "gemv_worklist", "cpu_cold": "kt_amx_bf16"},
        "default": {"gate": {"bands": [[0, 256, "hot"]], "cold_shards": []},
                    "up": {"bands": [[0, 256, "hot"]], "cold_shards": []},
                    "down": {"bands": [[0, 128, "hot"]], "cold_shards": []}},
    }
    p = tmp_path / "plan.json"
    p.write_text(json.dumps(raw))
    return str(p)


@pytest.fixture
def fresh_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv(M._ENV_PLAN, _plan_file(tmp_path))
    monkeypatch.setattr(M, "_RUNTIME", None)
    yield
    monkeypatch.setattr(M, "_RUNTIME", None)


def _parallel(monkeypatch, rank, size, ep=1):
    monkeypatch.setattr(rc, "get_parallel",
                        lambda: SimpleNamespace(moe_tp_rank=rank, moe_tp_size=size, moe_ep_size=ep))


class _GpuMethod:
    load_up_proj_weight_first = False


def _layer(tp_size):
    return SimpleNamespace(moe_tp_size=tp_size, moe_tp_rank=0, layer_id=0,
                           register_parameter=lambda *a, **k: None, _params={})


def test_coords_default_without_distributed(monkeypatch):
    def boom():
        raise RuntimeError("not initialized")
    monkeypatch.setattr(rc, "get_parallel", boom)
    assert M.moe_tp_coords() == (0, 1, 1)
    assert M.is_tp_owner()


def test_tp1_is_plain(monkeypatch):
    _parallel(monkeypatch, 0, 1)
    m = M.PrismMoEMethod(_GpuMethod(), layer_id=3)
    assert m.is_owner and m.prism_tp_mode is None


def test_owner_rank_loads_full_rows(monkeypatch, fresh_runtime):
    _parallel(monkeypatch, 0, 2)
    m = M.PrismMoEMethod(_GpuMethod(), layer_id=0)
    assert m.is_owner and m.prism_tp_mode == "full"
    registered = {}
    layer = SimpleNamespace(moe_tp_size=2, moe_tp_rank=0, layer_id=0,
                            register_parameter=lambda n, p: registered.__setitem__(n, p))
    # FusedMoE는 intermediate/moe_tp_size 를 넘긴다 — prism은 whole row(inter_full)로 만든다
    m.create_weights(layer, DIMS["num_experts"], DIMS["hidden_size"],
                     DIMS["intermediate_size"] // 2, torch.bfloat16, weight_loader=None)
    assert tuple(registered["w13_weight"].shape) == (8, 2 * 128, 256)
    assert tuple(registered["w2_weight"].shape) == (8, 256, 128)
    assert registered["w13_weight"].device.type == "cpu"


def test_nonowner_rank_holds_nothing_and_emits_zeros(monkeypatch, fresh_runtime):
    _parallel(monkeypatch, 1, 2)
    m = M.PrismMoEMethod(_GpuMethod(), layer_id=0)
    assert not m.is_owner and m.prism_tp_mode == "skip"
    registered = {}
    layer = SimpleNamespace(moe_tp_size=2, moe_tp_rank=1, layer_id=0,
                            register_parameter=lambda n, p: registered.__setitem__(n, p))
    m.create_weights(layer, 8, 256, 64, torch.bfloat16, weight_loader=None)
    assert registered == {}
    m.process_weights_after_loading(layer)   # no-op, must not touch runtime/executor
    assert M._RUNTIME is None or M._RUNTIME._executor is None
    x = torch.randn(3, 256, dtype=torch.bfloat16)
    topk = SimpleNamespace(topk_ids=torch.zeros(3, 2, dtype=torch.int64),
                           topk_weights=torch.ones(3, 2))
    out = m.apply(layer, SimpleNamespace(hidden_states=x, topk_output=topk))
    assert out.hidden_states.shape == x.shape and out.hidden_states.dtype == x.dtype
    assert torch.count_nonzero(out.hidden_states) == 0


def test_ep_is_rejected(monkeypatch):
    _parallel(monkeypatch, 0, 1, ep=2)
    with pytest.raises(NotImplementedError):
        M.PrismMoEMethod(_GpuMethod(), layer_id=0)


def test_checkpoint_filter_skips_experts_on_nonowner_only(monkeypatch, tmp_path):
    from sglang.srt.model_loader import weight_utils as W

    monkeypatch.setenv("SGLANG_PRISM_PLAN", "x.json")
    _parallel(monkeypatch, 1, 2)
    skip = W._prism_skip_tensor_fn()
    assert skip is not None
    assert skip("model.layers.5.mlp.experts.17.w1.weight_packed")
    assert skip("model.layers.5.mlp.experts.17.w2.weight_scale")
    assert not skip("model.layers.5.mlp.shared_experts.gate_proj.weight")   # 공유 expert는 TP 몫
    assert not skip("model.layers.5.mlp.routed_expert_down_proj.weight")     # latent proj는 dense
    assert not skip("model.layers.5.self_attn.q_proj.weight")
    _parallel(monkeypatch, 0, 2)
    assert W._prism_skip_tensor_fn() is None       # owner는 전부 읽는다
    _parallel(monkeypatch, 0, 1)
    assert W._prism_skip_tensor_fn() is None       # TP=1
    monkeypatch.delenv("SGLANG_PRISM_PLAN")
    _parallel(monkeypatch, 1, 2)
    assert W._prism_skip_tensor_fn() is None       # prism 꺼짐

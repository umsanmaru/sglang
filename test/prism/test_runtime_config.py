"""CPU-only contracts for the extracted MoE environment settings."""
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sglang.srt.layers.moe.prism.runtime.config import PrismEnvConfig


def config(values=None, cores=32):
    return PrismEnvConfig({"SGLANG_PRISM_" + k: v for k, v in (values or {}).items()},
                          cpu_count=lambda: cores)


def test_defaults_and_required_plan():
    c = config()
    assert not c.enabled
    with pytest.raises(KeyError, match="SGLANG_PRISM_PLAN"):
        _ = c.plan_path
    assert c.max_tokens == 4096
    assert c.load_threads() == 16
    assert c.cpuinfer_threads == 14
    assert c.split_streams and not c.cold_stream
    assert not c.cold_async and not c.cold_split and not c.mask_stats_nan_probe
    assert c.mask_stats_path is None and c.mask_stats_every == 16
    assert c.cold_hybrid_frac is None and not c.hybrid_mask
    assert c.warm_cpu_min_m is None
    assert c.warm_cpu_max_m("bf16", 1536) == 1536
    assert c.warm_cpu_max_m("fp8", 1536) is None


@pytest.mark.parametrize("raw,enabled,split", [("1", True, True), ("0", False, False),
                                              ("", False, True), ("true", False, True)])
def test_legacy_boolean_conventions(raw, enabled, split):
    c = config({k: raw for k in ("COLD_STREAM", "COLD_ASYNC", "COLD_SPLIT",
                                 "MASK_STATS_NANPROBE", "SPLIT_STREAMS")})
    assert c.cold_stream == c.cold_async == c.cold_split == c.mask_stats_nan_probe == enabled
    assert c.split_streams == split


@pytest.mark.parametrize("cores,load,cpu", [(None, 2, 2), (0, 2, 2), (1, 1, 2),
                                          (8, 4, 2), (32, 16, 14)])
def test_cpu_count_fallback(cores, load, cpu):
    c = config(cores=cores)
    assert c.load_threads() == load
    assert c.cpuinfer_threads == cpu


def test_only_loader_threads_are_clamped():
    c = config({"LOAD_THREADS": "-1", "CPUINFER_THREADS": "-1", "MAX_TOKENS": "0"})
    assert c.load_threads() == 1
    assert c.cpuinfer_threads == -1
    assert c.max_tokens == 0


@pytest.mark.parametrize("raw,expected", [("0", None), ("-1", None), ("64", 64)])
def test_explicit_cold_threshold_does_not_resolve_default(raw, expected):
    c = config({"COLD_GPU_MIN_M": raw, "GROUPED_MIN_M": "invalid"})
    def forbidden(_):
        pytest.fail("explicit threshold must not initialize the runtime default")
    assert c.cold_gpu_min_m(forbidden) == expected


def test_default_resolution_is_lazy_and_preserves_raw_grouped_value():
    c = config({"GROUPED_MIN_M": "32"})
    seen = []
    assert c.cold_gpu_min_m(lambda raw: seen.append(raw) or int(raw)) == 32
    assert seen == ["32"]


@pytest.mark.parametrize("key", ["MAX_TOKENS", "CPUINFER_THREADS", "LOAD_THREADS",
                                 "MASK_STATS_EVERY", "COLD_GPU_MIN_M"])
def test_empty_integer_is_not_an_unset_value(key):
    c = config({key: ""})
    readers = {"MAX_TOKENS": lambda: c.max_tokens,
               "CPUINFER_THREADS": lambda: c.cpuinfer_threads,
               "LOAD_THREADS": c.load_threads,
               "MASK_STATS_EVERY": lambda: c.mask_stats_every,
               "COLD_GPU_MIN_M": lambda: c.cold_gpu_min_m(lambda _: 1536)}
    with pytest.raises(ValueError):
        readers[key]()


def test_optional_settings_and_conditional_validation():
    c = config({"WARM_CPU_MIN_M": "", "COLD_HYBRID_FRAC": "", "NUMA_MAP": "",
                "MASK_STATS_EVERY": "bad"})
    assert c.warm_cpu_min_m is None and c.cold_hybrid_frac is None
    assert c.hybrid_local_node(2) == 2
    assert c.enabled is False  # unrelated malformed option stays dormant
    c = config({"WARM_CPU_MIN_M": "0", "WARM_CPU_MAX_M": "bad"})
    assert c.warm_cpu_min_m == 0
    assert c.warm_cpu_max_m("bf16", 1536) is None
    assert config({"WARM_CPU_MAX_M": "bad"}).warm_cpu_max_m("fp8", 1536) is None


def test_numa_and_hybrid_values():
    c = config({"NUMA_MAP": "3,1", "COLD_HYBRID_FRAC": "0.25,0.75"})
    assert c.parse_numa_map(c.numa_map_raw) == [3, 1]
    assert c.hybrid_local_node(1) == 1
    assert c.hybrid_local_node(9) == 0
    assert c.cold_hybrid_frac == (0.25, 0.75)
    c = config({"COLD_HYBRID_FRAC": "bad"})
    assert c.hybrid_mask
    with pytest.raises(ValueError):
        _ = c.cold_hybrid_frac


@pytest.mark.parametrize("name,cold,has_cold,all_warm,expected", [
    ("bf16", 1536, True, True, True), ("fp8", 16, True, True, False),
    ("bf16", None, True, True, False), ("bf16", 1536, False, True, False),
    ("bf16", 1536, True, False, False),
])
def test_packed_warm_defaults(name, cold, has_cold, all_warm, expected):
    assert config().warm_kt_enabled(name, lambda: cold, has_cold=has_cold,
                                    all_proj_warm=all_warm) == expected


def test_warm_override_keeps_legacy_short_circuit_and_error_behavior():
    def bad():
        raise ValueError("invalid cold threshold")
    assert config({"WARM_KT": "1"}).warm_kt_enabled("fp8", bad)
    with pytest.raises(ValueError, match="invalid cold"):
        config({"WARM_KT": "0"}).warm_kt_enabled("bf16", bad)


def test_reads_remain_live_not_a_constructor_snapshot(monkeypatch):
    env = {"SGLANG_PRISM_MAX_TOKENS": "8"}
    c = PrismEnvConfig(env)
    assert c.max_tokens == 8
    env["SGLANG_PRISM_MAX_TOKENS"] = "16"
    assert c.max_tokens == 16
    monkeypatch.setenv("SGLANG_PRISM_MAX_TOKENS", "32")
    live = PrismEnvConfig()
    monkeypatch.setenv("SGLANG_PRISM_MAX_TOKENS", "64")
    assert live.max_tokens == 64


def test_config_module_can_load_without_sglang_or_accelerators():
    # Normal dotted import may load torch through upstream package initializers.
    # The extracted module itself must remain independently usable with stdlib.
    path = Path(__file__).resolve().parents[2] / "python/sglang/srt/layers/moe/prism/runtime/config.py"
    subprocess.run([sys.executable, "-c", "import runpy, sys; "
                    "runpy.run_path(sys.argv[1]); "
                    "assert 'torch' not in sys.modules; assert 'kt_kernel' not in sys.modules",
                    str(path)], check=True)


def test_method_compatibility_and_override_without_runtime(monkeypatch):
    from sglang.srt.layers.moe.prism import method as m
    monkeypatch.setattr(m, "_CONFIG", config({"PLAN": "plan.json", "COLD_GPU_MIN_M": "0"}))
    monkeypatch.setattr(m, "_RUNTIME", None)
    assert m._ENV_PLAN == "SGLANG_PRISM_PLAN"
    assert m._prism_predicate(SimpleNamespace(layer_id=7), None) == {"layer_id": 7}
    assert m._cold_gpu_min_m() is None
    assert m._RUNTIME is None

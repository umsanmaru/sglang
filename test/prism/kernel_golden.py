"""Bit-exact golden gate for the Prism JIT kernels (refactor gate, K0).

Wraps every tvm-ffi module the Prism kernel wrappers load, and records each FFI call
made while the test suite runs: a hash of every tensor argument before the call and
after it. A later run in ``check`` mode replays the same suite and requires, call by
call, that identical inputs still produce identical bytes.

Enable it with an environment variable (normal test runs are unaffected)::

    PRISM_KERNEL_GOLDEN=record:<dir>   pytest test/prism   # writes <dir>/golden.json
    PRISM_KERNEL_GOLDEN=check:<dir>    pytest test/prism   # compares, fails on drift

Deselect timing tests in gate runs (``-k "not perf_smoke"``); the per-call sync and hashing
slow every kernel call.

What it pins down while active, so two runs see the same bytes:
- CUDA ``torch.empty``-family allocations are zero-filled. Kernels often write only part
  of an output buffer; garbage in the untouched part would otherwise differ between runs.
- Global RNGs are seeded per test.
Calls made during CUDA graph capture pass through unrecorded (no sync is legal there).

The report also lists, per module, the exported entry points the suite never called.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import random
from typing import Dict, List, Optional

import torch

ENV = "PRISM_KERNEL_GOLDEN"

# Modules whose `load_jit` result is wrapped. Kept as a list so the gate survives the
# planned move of these modules (only this list changes).
WRAPPER_MODULES = (
    "sglang.kernels.ops.moe.prism.gemv",
    "sglang.kernels.ops.moe.prism.gemv_fp8",
    "sglang.kernels.ops.moe.prism.gemv_mxfp4",
    "sglang.kernels.ops.moe.prism.gemv_packed",
    "sglang.kernels.ops.moe.prism.grouped",
    "sglang.kernels.ops.moe.prism.grouped_fp8",
    "sglang.kernels.ops.moe.prism.grouped_mxfp4",
    "sglang.kernels.ops.moe.prism.gather",
    "sglang.kernels.ops.moe.prism.thr_ctrl",
)


def _parse_env() -> Optional[tuple]:
    raw = os.environ.get(ENV, "")
    if not raw:
        return None
    mode, _, path = raw.partition(":")
    if mode not in ("record", "check") or not path:
        raise ValueError(f"{ENV} must be 'record:<dir>' or 'check:<dir>', got {raw!r}")
    return mode, os.path.abspath(path)


def _digest(t: torch.Tensor) -> str:
    """Hash of a tensor's logical contents (dtype, shape and bytes)."""
    c = t.detach()
    if c.device.type != "cpu":
        c = c.cpu()
    c = c.contiguous()
    h = hashlib.blake2b(digest_size=12)
    h.update(f"{c.dtype}|{tuple(c.shape)}|".encode())
    if c.numel():
        h.update(c.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _describe(args) -> List[str]:
    out = []
    for a in args:
        if isinstance(a, torch.Tensor):
            out.append("T:" + _digest(a))
        else:
            out.append("S:" + repr(a))
    return out


class _State:
    def __init__(self, mode: str, path: str):
        self.mode = mode
        self.path = path
        self.calls: Dict[str, List[dict]] = {}
        self.exports: Dict[str, List[str]] = {}
        self.called: Dict[str, set] = {}
        self.test = "<collection>"
        self.seq = 0
        self.skipped_in_capture = 0

    def record(self, module: str, fn: str, pre: List[str], post: List[str]) -> None:
        self.calls.setdefault(self.test, []).append(
            {"seq": self.seq, "fn": f"{module}.{fn}", "pre": pre, "post": post}
        )
        self.seq += 1
        self.called.setdefault(module, set()).add(fn)


_STATE: Optional[_State] = None


class _ModuleProxy:
    """Stands in for a tvm-ffi Module; every attribute call is hashed around."""

    def __init__(self, inner, name: str):
        self._inner = inner
        self._name = name

    def __getattr__(self, attr):
        fn = getattr(self._inner, attr)
        if not callable(fn):
            return fn
        name = self._name

        def call(*args):
            st = _STATE
            if st is None or torch.cuda.is_current_stream_capturing():
                if st is not None:
                    st.skipped_in_capture += 1
                return fn(*args)
            torch.cuda.synchronize()
            pre = _describe(args)
            ret = fn(*args)
            torch.cuda.synchronize()
            st.record(name, attr, pre, _describe(args))
            return ret

        return call


def _wrap_load_jit(orig):
    def load_jit(*args, **kwargs):
        mod = orig(*args, **kwargs)
        name = str(args[0]) if args else "?"
        if _STATE is not None:
            wrappers = kwargs.get("cuda_wrappers") or []
            _STATE.exports[name] = sorted(w[0] for w in wrappers)
        return _ModuleProxy(mod, name)

    return load_jit


def _zero_cuda(t):
    # Host buffers stay untouched: zeroing them here would first-touch their pages on
    # this thread's NUMA node and break Prism's warm-store node placement checks.
    return t.zero_() if isinstance(t, torch.Tensor) and t.is_cuda else t


def _zero_fill_empties() -> None:
    """Make CUDA `torch.empty`-family allocations deterministic (zeros) while the gate runs."""

    def zeros_like_empty(orig_name):
        orig = getattr(torch, orig_name)

        def f(*args, **kwargs):
            return _zero_cuda(orig(*args, **kwargs))

        f.__name__ = orig_name
        return f

    for n in ("empty", "empty_like", "empty_strided"):
        setattr(torch, n, zeros_like_empty(n))
    orig_new_empty = torch.Tensor.new_empty
    torch.Tensor.new_empty = lambda self, *a, **k: _zero_cuda(
        orig_new_empty(self, *a, **k)
    )


def install() -> bool:
    """Activate the gate if the environment asks for it. Returns True when active."""
    global _STATE
    parsed = _parse_env()
    if parsed is None:
        return False
    _STATE = _State(*parsed)
    _zero_fill_empties()
    for modname in WRAPPER_MODULES:
        m = importlib.import_module(modname)
        m.load_jit = _wrap_load_jit(m.load_jit)
    return True


# ===== pytest hooks (registered by conftest.py when the gate is active) =====


def pytest_runtest_setup(item):
    _STATE.test = item.nodeid
    _STATE.seq = 0
    random.seed(0)
    torch.manual_seed(0)
    try:
        import numpy as np

        np.random.seed(0)
    except ImportError:
        pass


def _compare(golden: Dict[str, List[dict]], now: Dict[str, List[dict]]) -> dict:
    out = {
        "kernel_drift": [],
        "input_drift": [],
        "missing_tests": [],
        "new_tests": [],
        "call_count_changed": [],
        "compared_calls": 0,
    }
    for test, gcalls in golden.items():
        ncalls = now.get(test)
        if ncalls is None:
            out["missing_tests"].append(test)
            continue
        if len(ncalls) != len(gcalls):
            out["call_count_changed"].append(f"{test}: {len(gcalls)} -> {len(ncalls)}")
        for g, n in zip(gcalls, ncalls):
            where = f"{test} #{g['seq']} {g['fn']}"
            if g["fn"] != n["fn"]:
                out["kernel_drift"].append(f"{where}: called {n['fn']} instead")
                continue
            if g["pre"] != n["pre"]:
                bad = [i for i, (a, b) in enumerate(zip(g["pre"], n["pre"])) if a != b]
                out["input_drift"].append(f"{where}: input args {bad} differ")
                continue
            out["compared_calls"] += 1
            if g["post"] != n["post"]:
                bad = [
                    i for i, (a, b) in enumerate(zip(g["post"], n["post"])) if a != b
                ]
                out["kernel_drift"].append(f"{where}: output args {bad} differ")
    out["new_tests"] = sorted(set(now) - set(golden))
    return out


def _coverage(st: _State) -> Dict[str, List[str]]:
    return {
        m: sorted(set(ex) - st.called.get(m, set()))
        for m, ex in sorted(st.exports.items())
    }


def pytest_sessionfinish(session, exitstatus):
    st = _STATE
    os.makedirs(st.path, exist_ok=True)
    total = sum(len(v) for v in st.calls.values())
    cov = _coverage(st)
    summary = {
        "mode": st.mode,
        "recorded_calls": total,
        "tests_with_calls": len(st.calls),
        "skipped_in_graph_capture": st.skipped_in_capture,
        "uncalled_exports": cov,
    }
    if st.mode == "record":
        with open(os.path.join(st.path, "golden.json"), "w") as f:
            json.dump(
                {"calls": st.calls, "exports": st.exports}, f, indent=0, sort_keys=True
            )
    else:
        with open(os.path.join(st.path, "golden.json")) as f:
            golden = json.load(f)["calls"]
        summary["compare"] = _compare(golden, st.calls)
        with open(os.path.join(st.path, "check_now.json"), "w") as f:
            json.dump(
                {"calls": st.calls, "exports": st.exports}, f, indent=0, sort_keys=True
            )
        c = summary["compare"]
        if c["kernel_drift"] or c["call_count_changed"] or c["missing_tests"]:
            session.exitstatus = 1
    with open(os.path.join(st.path, f"{st.mode}_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, sort_keys=True)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    st = _STATE
    tr = terminalreporter
    tr.section(f"prism kernel golden ({st.mode})")
    tr.write_line(
        f"recorded calls: {sum(len(v) for v in st.calls.values())} "
        f"in {len(st.calls)} tests; skipped in graph capture: {st.skipped_in_capture}"
    )
    for m, missing in _coverage(st).items():
        n = len(st.exports.get(m, []))
        tr.write_line(
            f"  {m}: {n - len(missing)}/{n} exports called"
            + (f"; never: {', '.join(missing)}" if missing else "")
        )
    if st.mode == "check":
        path = os.path.join(st.path, "check_summary.json")
        if os.path.exists(path):
            with open(path) as f:
                c = json.load(f)["compare"]
            tr.write_line(
                f"compared calls: {c['compared_calls']}, kernel drift: "
                f"{len(c['kernel_drift'])}, input drift: {len(c['input_drift'])}, "
                f"call-count changes: {len(c['call_count_changed'])}, "
                f"missing tests: {len(c['missing_tests'])}"
            )
            for line in (c["kernel_drift"] + c["call_count_changed"])[:20]:
                tr.write_line("  DRIFT " + line)

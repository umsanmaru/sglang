"""Kernel keys for the Prism GEMV and grouped GEMM entry points.

A key names one entry point: (op, store format, sparse score, gate+up fusion). Hot and warm
placements share the entry; the wrapper takes the placement as its `on_device` argument. Support
is read from the wrappers' export tables, so a key is supported exactly when its export is built.
"""

from __future__ import annotations

import functools
import importlib
from typing import Callable, Literal, Optional

import msgspec

Op = Literal["gemv", "grouped", "grouped_cold"]
Fmt = Literal["bf16", "fp8", "fp8pt", "mxfp4"]
Mode = Literal["dense", "pair", "k1", "k1w"]

OPS: tuple[str, ...] = ("gemv", "grouped", "grouped_cold")
FMTS: tuple[str, ...] = ("bf16", "fp8", "fp8pt", "mxfp4")
MODES: tuple[str, ...] = ("dense", "pair", "k1", "k1w")

# (op, fmt) -> (wrapper module, export name prefix). Cold FP8 has one export family for both
# scale layouts; the slab layout argument tells them apart.
_FAMILY: dict[tuple[str, str], tuple[str, str]] = {
    ("gemv", "bf16"): ("gemv", "gemv_worklist_indexed"),
    ("gemv", "fp8"): ("gemv_fp8", "gemv_fp8_indexed"),
    ("gemv", "fp8pt"): ("gemv_fp8", "gemv_fp8pt_indexed"),
    ("gemv", "mxfp4"): ("gemv_mxfp4", "gemv_mxfp4_indexed"),
    ("grouped", "bf16"): ("grouped", "grouped_gemm_indexed"),
    ("grouped", "fp8"): ("grouped_fp8", "grouped_fp8_indexed"),
    ("grouped", "fp8pt"): ("grouped_fp8", "grouped_fp8pt_indexed"),
    ("grouped", "mxfp4"): ("grouped_mxfp4", "grouped_mxfp4_indexed"),
    ("grouped_cold", "bf16"): ("grouped", "grouped_gemm_cold"),
    ("grouped_cold", "fp8"): ("grouped_fp8", "grouped_fp8_cold"),
    ("grouped_cold", "fp8pt"): ("grouped_fp8", "grouped_fp8_cold"),
    ("grouped_cold", "mxfp4"): ("grouped_mxfp4", "grouped_mxfp4_cold"),
}
_MODE_SUFFIX = {"dense": "", "pair": "_sparse", "k1": "_sparsek1", "k1w": "_sparsek1w"}


class PrismKernelKey(msgspec.Struct, frozen=True):
    op: Op
    fmt: Fmt
    mode: Mode = "dense"
    fused: bool = False

    def export_name(self) -> str:
        _, prefix = _FAMILY[(self.op, self.fmt)]
        gateup = "_gateup" if self.fused else ""
        return f"{prefix}{_MODE_SUFFIX[self.mode]}{gateup}"


def _module(name: str):
    return importlib.import_module(f"sglang.kernels.ops.moe.prism.{name}")


def module_exports(name: str) -> frozenset[str]:
    """Export names one wrapper module builds (reads its table; compiles nothing)."""
    return frozenset(export for export, _ in _module(name)._ENTRIES)


def unsupported_reason(key: PrismKernelKey) -> Optional[str]:
    """None if `key` has an export, else why not."""
    family = _FAMILY.get((key.op, key.fmt))
    if family is None:
        return f"no {key.op} kernel for {key.fmt}"
    if key.export_name() not in module_exports(family[0]):
        fusion = " gate+up" if key.fused else ""
        return f"{key.fmt} {key.op} has no {key.mode}{fusion} entry ({key.export_name()})"
    return None


def supports(key: PrismKernelKey) -> bool:
    return unsupported_reason(key) is None


def wrapper(key: PrismKernelKey) -> Callable:
    """Python wrapper for `key`. The sparse wrappers take every score and pick the k1/k1w
    export from the spec at call time, so a sparse key resolves to its pair-named wrapper."""
    reason = unsupported_reason(key)
    if reason is not None:
        raise ValueError(reason)
    mode = "dense" if key.mode == "dense" else "pair"
    named = PrismKernelKey(op=key.op, fmt=key.fmt, mode=mode, fused=key.fused)
    return getattr(
        _module(_FAMILY[(key.op, key.fmt)][0]), named.export_name()
    )


@functools.cache
def resolve(*, op: str, fmt: str, mode: str, fused: bool) -> Optional[Callable]:
    """Cached wrapper lookup for the per-step tier paths; None if the key is not built."""
    key = PrismKernelKey(op=op, fmt=fmt, mode=mode, fused=fused)
    return wrapper(key) if supports(key) else None

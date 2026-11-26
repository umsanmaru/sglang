"""Kernel keys for the Prism GEMV and grouped GEMM entry points.

A key names one entry point up to its weight placement: (op, store format, sparse score,
gate+up fusion). Hot and warm placements are the same kernel and differ only in the export
name (`_pinned` = warm). Support is read from the wrappers' export tables, so a key is
supported exactly when its exports are built.
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

    def placements(self) -> tuple[bool, ...]:
        """on_device values with an export: cold slabs are host-only, the rest have both."""
        return (False,) if self.op == "grouped_cold" else (True, False)

    def export_name(self, *, on_device: bool) -> str:
        _, prefix = _FAMILY[(self.op, self.fmt)]
        pinned = "" if on_device or self.op == "grouped_cold" else "_pinned"
        gateup = "_gateup" if self.fused else ""
        return f"{prefix}{pinned}{_MODE_SUFFIX[self.mode]}{gateup}"


def _module(name: str):
    return importlib.import_module(f"sglang.kernels.ops.moe.prism.{name}")


def module_exports(name: str) -> frozenset[str]:
    """Export names one wrapper module builds (reads its table; compiles nothing)."""
    return frozenset(export for export, _ in _module(name)._ENTRIES)


def unsupported_reason(key: PrismKernelKey) -> Optional[str]:
    """None if every placement of `key` has an export, else why not."""
    family = _FAMILY.get((key.op, key.fmt))
    if family is None:
        return f"no {key.op} kernel for {key.fmt}"
    exports = module_exports(family[0])
    missing = [
        key.export_name(on_device=p)
        for p in key.placements()
        if key.export_name(on_device=p) not in exports
    ]
    if missing:
        fusion = " gate+up" if key.fused else ""
        return (
            f"{key.fmt} {key.op} has no {key.mode}{fusion} entry ({', '.join(missing)})"
        )
    return None


def supports(key: PrismKernelKey) -> bool:
    return unsupported_reason(key) is None


def wrapper(key: PrismKernelKey, *, on_device: bool) -> Callable:
    """Python wrapper for `key`. The sparse wrappers take every score and pick the k1/k1w
    export from the spec at call time, so a sparse key resolves to its pair-named wrapper."""
    reason = unsupported_reason(key)
    if reason is not None:
        raise ValueError(reason)
    mode = "dense" if key.mode == "dense" else "pair"
    named = PrismKernelKey(op=key.op, fmt=key.fmt, mode=mode, fused=key.fused)
    return getattr(
        _module(_FAMILY[(key.op, key.fmt)][0]), named.export_name(on_device=on_device)
    )


@functools.cache
def resolve(
    *, op: str, fmt: str, mode: str, fused: bool, on_device: bool
) -> Optional[Callable]:
    """Cached wrapper lookup for the per-step tier paths; None if the key is not built."""
    key = PrismKernelKey(op=op, fmt=fmt, mode=mode, fused=fused)
    return wrapper(key, on_device=on_device) if supports(key) else None

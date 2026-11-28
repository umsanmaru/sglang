"""Launch tuning for the Prism kernels: performance-only knobs, per GPU model.

These values change speed, never results. Knobs that set the accumulation order (the NY
rule, kNCol) or the tile contract with grouping.py (TILE_M) stay in code. The defaults below
are measured values; configs/<device>.json overrides them for one GPU model, where <device>
is torch.cuda.get_device_name() lowercased with non-alphanumerics replaced by "_".
"""

from __future__ import annotations

import functools
import pathlib
import re

import msgspec
import torch

_CONFIG_DIR = pathlib.Path(__file__).parent / "configs"


class LaunchTuning(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    # Block cap for PCIe-bound launches (pinned warm, cold slabs), leaving SMs for hot kernels
    # on another stream. Measured 2026-08-27 on H100 (35B h50/w50 gate+up, M=2048): 128 blocks
    # still saturate PCIe (51 GB/s) and fully hide hot, 13.1 -> 10.5 ms. 0 = no cap.
    warm_max_blocks: int = 128
    # Send PCIe grouped launches to the W-resident kernel, which reads each expert's W once.
    wres_pcie: bool = True


def config_name(device_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", device_name.lower()).strip("_")


@functools.cache
def for_device(device_name: str) -> LaunchTuning:
    path = _CONFIG_DIR / f"{config_name(device_name)}.json"
    if not path.is_file():
        return LaunchTuning()
    return msgspec.json.decode(path.read_bytes(), type=LaunchTuning)


@functools.cache
def _for_index(device_index: int) -> LaunchTuning:
    return for_device(torch.cuda.get_device_name(device_index))


def current() -> LaunchTuning:
    """Tuning for the current CUDA device; cached, so launch paths may call it per launch."""
    return _for_index(torch.cuda.current_device())

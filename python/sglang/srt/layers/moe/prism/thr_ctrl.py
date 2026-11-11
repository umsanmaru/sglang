"""Online threshold controller: holds each (layer, proj)'s realized decode sparsity at its plan budget.

The calibrated threshold curves are fixed, but long contexts shrink activations, so a fixed
threshold cuts more than the budget p. Each layer gets one alpha per proj, shared by the GPU warm
masks and the kt cold masks. Both read live threshold tables (calib * alpha) through pointers that
are fixed at load, so rewriting the tables in place changes the next step's masks without touching
any kernel argument, including under CUDA-graph replay.

Per masked decode step the mask builders add {kept, total} mask units to counters (device for the
GPU tiers, pinned host for kt), and `step` launches one single-block kernel after the layer's rejoin.
Every `window` steps that kernel computes the realized full-K sparsity
d = (total - kept) / (window * top_k * K units), with hot rows counted as kept, sets
alpha <- clamp(alpha * (1 - mu * (d - p))), and rewrites both live tables from the calibrated copy.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import threading
import time
from typing import Mapping, Optional

import torch

from sglang.srt.layers.moe.prism.plan import PAIR_GROUP, Plan, Proj, Tier

logger = logging.getLogger(__name__)

# Row order of every [3, ...] controller buffer; kt's SparsityGeometry::keep_ctr uses the same order.
PROJ_ORDER = (Proj.GATE, Proj.UP, Proj.DOWN)
# Ring length of the per-window (alpha, d) log of each layer. Arbitrary; 1024 windows of 4096 tokens.
LOG_WINDOWS = 1024
# Seconds between report dumps while serving; arbitrary.
DUMP_INTERVAL_S = 30.0


class LayerThresholdState:
    """One layer's controller buffers. Every leading axis of size 3 follows PROJ_ORDER."""

    def __init__(self, calib_thr: Mapping[Proj, torch.Tensor], device: torch.device):
        calib = torch.stack([calib_thr[p].to(torch.float32) for p in PROJ_ORDER]).contiguous()
        self.calib_dev = calib.to(device)  # [3, E, ng], never written
        self.live_dev = self.calib_dev.clone()  # read by the GPU sparse tiers
        self.live_host = calib.pin_memory()  # read by kt; written by the step kernel through UVA
        self.gpu_ctr = torch.zeros(3, 2, dtype=torch.int64, device=device)
        self.cpu_ctr = torch.zeros(3, 2, dtype=torch.int64).pin_memory()
        self.snap = torch.zeros(3, 2, dtype=torch.int64, device=device)
        self.tokens = torch.zeros(1, dtype=torch.int64, device=device)
        self.alpha = torch.ones(3, dtype=torch.float32, device=device)
        # Explicit dtype: the server sets torch's default dtype to the model's (bf16) during load.
        self.log = torch.full((LOG_WINDOWS, 3, 2), float("nan"), dtype=torch.float32).pin_memory()
        # Finished windows, published by the step kernel after their log entries.
        self.windows_host = torch.zeros(1, dtype=torch.int64).pin_memory()

    def host_tables(self) -> dict[Proj, torch.Tensor]:
        """Live [E, ng] tables for kt, in place of the calibrated curves."""
        return {p: self.live_host[i] for i, p in enumerate(PROJ_ORDER)}

    def device_table(self, proj: Proj) -> torch.Tensor:
        return self.live_dev[PROJ_ORDER.index(proj)]

    def gpu_counter(self, proj: Proj) -> torch.Tensor:
        return self.gpu_ctr[PROJ_ORDER.index(proj)]


class ThresholdController:
    """Owns every layer's LayerThresholdState and launches the per-step update."""

    def __init__(self, plan: Plan, *, log_path: Optional[str] = None):
        sparsity = plan.sparsity
        if sparsity is None or sparsity.controller is None:
            raise ValueError("ThresholdController needs plan.sparsity.controller")
        self._plan = plan
        self._ctrl = sparsity.controller
        rows_per_unit = PAIR_GROUP if sparsity.score == "k2wl2" else 1
        dims = plan.dims
        self._units_per_token = tuple(
            dims.top_k * dims.k_of(p) / rows_per_unit for p in PROJ_ORDER)
        self._layers: dict[int, LayerThresholdState] = {}
        if log_path:
            # The scheduler process may be killed without running atexit, so also dump periodically.
            atexit.register(self._dump, log_path)
            threading.Thread(target=self._dump_forever, args=(log_path,), daemon=True,
                             name="prism-thr-ctrl-dump").start()

    def register_layer(self, layer_idx: int, calib_thr: Mapping[Proj, torch.Tensor],
                       device: torch.device) -> LayerThresholdState:
        if layer_idx in self._layers:
            raise RuntimeError(f"thr_ctrl: layer {layer_idx} already registered")
        self._warn_unreachable(layer_idx)
        state = LayerThresholdState(calib_thr, device)
        self._layers[layer_idx] = state
        return state

    def layer(self, layer_idx: int) -> Optional[LayerThresholdState]:
        return self._layers.get(layer_idx)

    def step(self, layer_idx: int) -> None:
        """One masked decode step of this layer; call it on the stream after the layer's rejoin."""
        from sglang.jit_kernel.prism_thr_ctrl import thr_ctrl_step

        s = self._layers[layer_idx]
        thr_ctrl_step(
            live_dev=s.live_dev, calib_dev=s.calib_dev, live_host=s.live_host,
            gpu_ctr=s.gpu_ctr, cpu_ctr=s.cpu_ctr, snap=s.snap, tokens=s.tokens,
            alpha=s.alpha, log=s.log, windows_host=s.windows_host,
            window=self._ctrl.window, mu=self._ctrl.mu,
            alpha_min=self._ctrl.alpha_min, alpha_max=self._ctrl.alpha_max,
            targets=self._targets(layer_idx), units_per_token=self._units_per_token,
        )

    def report(self) -> dict:
        """Per layer: the (alpha, sparsity) logged for each finished window, oldest first.

        Reads pinned host memory only, so it is safe from any thread while the server runs; a
        window appears once the step kernel that finished it has completed.
        """
        out = {}
        for idx, s in sorted(dict(self._layers).items()):
            done = int(s.windows_host[0])
            n = min(done, LOG_WINDOWS)
            windows = [s.log[(done - n + w) % LOG_WINDOWS].tolist() for w in range(n)]
            out[idx] = {
                "windows_done": done,
                "target": dict(zip((p.value for p in PROJ_ORDER), self._targets(idx))),
                "windows": [{p.value: {"alpha": w[i][0], "sparsity": w[i][1]}
                             for i, p in enumerate(PROJ_ORDER)} for w in windows],
            }
        return out

    def _targets(self, layer_idx: int) -> tuple[float, ...]:
        # Budgets are per layer and proj; kt and the GPU tiers both read expert 0 (cold_backend).
        ep = self._plan.expert(layer_idx, 0)
        return tuple(float(ep.proj(p).sparsity_p) for p in PROJ_ORDER)

    def _warn_unreachable(self, layer_idx: int) -> None:
        # Hot rows are never masked, so full-K sparsity cannot exceed the masked share of K.
        ep = self._plan.expert(layer_idx, 0)
        for p, target in zip(PROJ_ORDER, self._targets(layer_idx)):
            masked = 1.0 - ep.proj(p).rows(Tier.HOT) / self._plan.dims.k_of(p)
            if target > masked:
                logger.warning(
                    "[prism] thr_ctrl layer %d %s: budget p=%.3f exceeds the masked share %.3f "
                    "of K; alpha will saturate at alpha_max", layer_idx, p.value, target, masked)

    def _dump(self, path: str) -> None:
        try:
            tmp = f"{path}.tmp"
            with open(tmp, "w") as f:
                json.dump({"window": self._ctrl.window, "mu": self._ctrl.mu,
                           "alpha_range": [self._ctrl.alpha_min, self._ctrl.alpha_max],
                           "layers": self.report()}, f, indent=1)
            os.replace(tmp, path)
        except Exception as err:  # never let reporting take down serving or the exit path
            logger.warning("[prism] thr_ctrl report failed: %s", err)

    def _dump_forever(self, path: str) -> None:
        while True:
            time.sleep(DUMP_INTERVAL_S)
            self._dump(path)

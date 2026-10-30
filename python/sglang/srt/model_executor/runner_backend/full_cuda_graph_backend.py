# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""FullCudaGraphBackend — captures the entire model forward as one
torch.cuda.CUDAGraph per shape.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, contextmanager
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    set_graph_pool_id,
)
from sglang.srt.model_executor.runner_backend.base_cuda_graph_backend import (
    BaseCudaGraphBackend,
)
from sglang.srt.model_executor.runner_utils.pool import (
    get_or_create_global_graph_memory_pool,
)
from sglang.srt.utils import get_bool_env_var

# ---- async graph launch experiment (SGLANG_PRISM_ASYNC_GRAPH_LAUNCH=1) ----
import ctypes as _ct
import os as _os
import threading as _threading

_ASYNC_LAUNCH = _os.environ.get("SGLANG_PRISM_ASYNC_GRAPH_LAUNCH") == "1"
# Forward-thread mode: the scheduler runs run_batch+sample on a dedicated thread;
# replay() then launches synchronously through ctypes so the (blocking) launch
# releases the GIL and the scheduler thread keeps running Python meanwhile.
_FORWARD_THREAD = _os.environ.get("SGLANG_PRISM_FORWARD_THREAD") == "1"
_LAUNCH_STATS = [] if _os.environ.get("SGLANG_PRISM_FT_STATS") == "1" else None
_pending_launch: Optional[_threading.Thread] = None
_pending_err: list = []
_cudart = None


def _get_cudart():
    global _cudart
    if _cudart is None:
        # torch's own runtime instance (already loaded in-process).
        import glob as _glob
        cands = _glob.glob(_os.path.join(_os.path.dirname(torch.__file__), "..", "nvidia", "cu13", "lib", "libcudart.so.13*"))
        cands += ["libcudart.so.13", "libcudart.so"]
        for c in cands:
            try:
                _cudart = _ct.CDLL(c)
                break
            except OSError:
                continue
        assert _cudart is not None, "libcudart not found for async graph launch"
        _cudart.cudaGraphLaunch.argtypes = [_ct.c_void_p, _ct.c_void_p]
        _cudart.cudaGraphLaunch.restype = _ct.c_int
        _cudart.cudaSetDevice.argtypes = [_ct.c_int]
    return _cudart


def _launch_worker(exec_handle: int, stream_handle: int, device: int):
    rt = _get_cudart()
    rt.cudaSetDevice(device)
    rc = rt.cudaGraphLaunch(_ct.c_void_p(exec_handle), _ct.c_void_p(stream_handle))
    if rc != 0:
        _pending_err.append(rc)


def _start_async_launch(exec_handle: int, stream_handle: int):
    global _pending_launch
    _get_cudart()
    t = _threading.Thread(
        target=_launch_worker,
        args=(exec_handle, stream_handle, torch.cuda.current_device()),
        name="prism-graph-launch",
        daemon=True,
    )
    t.start()
    _pending_launch = t


def wait_pending_launch():
    """Block until the helper's cudaGraphLaunch has returned (all graph work submitted)."""
    global _pending_launch
    t = _pending_launch
    if t is not None:
        t.join()
        _pending_launch = None
    if _pending_err:
        rc = _pending_err.pop()
        raise RuntimeError(f"async cudaGraphLaunch failed rc={rc}")
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.runner.base_cuda_graph_runner import (
        BaseCudaGraphRunner,
    )
    from sglang.srt.model_executor.runner.shape_key import ShapeKey


class FullCudaGraphBackend(BaseCudaGraphBackend):
    """One torch.cuda.CUDAGraph per shape; attention metadata is
    captured inside the graph. Memory-saver-aware.
    """

    def __init__(
        self,
        cuda_graph_runner: BaseCudaGraphRunner,
        *,
        enable_memory_saver: bool = False,
    ) -> None:
        self._graphs: Dict[Any, torch.cuda.CUDAGraph] = {}
        self._outputs: Dict[Any, Any] = {}
        self._pool = None
        self._cuda_graph_runner = cuda_graph_runner
        self._device_module = cuda_graph_runner.device_module
        self._tp_group = cuda_graph_runner.model_runner.tp_group
        self._capture_stream: Optional[torch.cuda.Stream] = None
        self._memory_saver_adapter: Optional[Any] = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
            and get_bool_env_var("SGLANG_MEMORY_SAVER_CUDA_GRAPH")
        )

    @contextmanager
    def capture_session(self, stream: torch.cuda.Stream):
        if self._pool is None:
            self._pool = get_or_create_global_graph_memory_pool(self._device_module)
        set_graph_pool_id(self._pool)
        self._capture_stream = stream
        try:
            yield
        finally:
            self._capture_stream = None

    def capture_one(
        self,
        shape_key: ShapeKey,
        forward_fn: Callable[[], Any],
        capture_inputs: Optional[Any] = None,
        post_warmup_hook: Optional[Callable[[], None]] = None,
    ) -> None:
        # When per-bs capture traces are enabled (--enable-profile-cuda-graph +
        # SGLANG_GRAPH_BATCH_CAPTURE), the runner created a scheduled
        # torch profiler (wait=2, active=1) and exposed it as _profiler. We step()
        # past the two warmup runs so only the capture run is recorded, and each
        # batch size produces its own trace via the profiler's on_trace_ready.
        # With --enable-profile-cuda-graph alone the runner leaves _profiler None
        # (its unscheduled profiler records the whole capture in one pass), so no
        # stepping happens here.
        runner = self._cuda_graph_runner
        profiler = (
            getattr(runner, "_profiler", None)
            if getattr(runner, "enable_profile_cuda_graph", False)
            else None
        )

        # Two warmups so kernels are loaded and one-time setup is paid before capture.
        # post_warmup_hook lets the attention backend reset state that warmup mutated.
        for _ in range(2):
            self._device_module.synchronize()
            self._tp_group.barrier()
            forward_fn()
            if profiler is not None:
                profiler.step()
            if post_warmup_hook is not None:
                post_warmup_hook()

        graph = torch.cuda.CUDAGraph()

        graph_ctx: Callable[..., AbstractContextManager]
        if (
            self._memory_saver_adapter is not None
            and self._memory_saver_adapter.enabled
        ):
            graph_ctx = partial(
                self._memory_saver_adapter.cuda_graph,
                tag=GPU_MEMORY_TYPE_CUDA_GRAPH,
            )
        else:
            graph_ctx = self._device_module.graph

        with graph_ctx(cuda_graph=graph, pool=self._pool, stream=self._capture_stream):
            out = forward_fn()

        if profiler is not None:
            profiler.step()

        self._graphs[shape_key] = graph
        self._outputs[shape_key] = out

    def can_run(self, forward_batch: ForwardBatch, shape_key: ShapeKey) -> bool:
        return shape_key in self._graphs

    @contextmanager
    def replay_session(self):
        yield

    def replay(
        self,
        shape_key: ShapeKey,
        static_forward_batch: ForwardBatch,
        **kwargs,
    ) -> Any:
        if _FORWARD_THREAD:
            rt = _get_cudart()
            exec_handle = self._graphs[shape_key].raw_cuda_graph_exec()
            stream = torch.cuda.current_stream()
            if _LAUNCH_STATS is not None:
                ev0 = torch.cuda.Event(enable_timing=True); ev0.record(stream)
            import time as _time
            t0 = _time.perf_counter()
            rc = rt.cudaGraphLaunch(_ct.c_void_p(exec_handle), _ct.c_void_p(stream.cuda_stream))
            dl = (_time.perf_counter() - t0) * 1e3
            if rc != 0:
                raise RuntimeError(f"cudaGraphLaunch (ctypes, forward thread) failed rc={rc}")
            if _LAUNCH_STATS is not None:
                ev1 = torch.cuda.Event(enable_timing=True); ev1.record(stream)
                _LAUNCH_STATS.append((dl, ev0, ev1))
                if len(_LAUNCH_STATS) >= 100:
                    done = [(d, a, b) for d, a, b in _LAUNCH_STATS if b.query()]
                    if len(done) >= 50:
                        import statistics as _st
                        spans = [a.elapsed_time(b) for _, a, b in done]
                        launches = [d for d, _, _ in done]
                        print(f"[prism-ft] launch {_st.mean(launches):.2f} ms (p10 {_st.quantiles(launches,n=10)[0]:.2f} p90 {_st.quantiles(launches,n=10)[-1]:.2f}) | graph gpu span {_st.mean(spans):.2f} ms | launch returns at {100*_st.mean(l/s for l,(s) in zip(launches,spans)):.0f}% | n={len(done)}", flush=True)
                        _LAUNCH_STATS.clear()
            return self._outputs[shape_key]
        if _ASYNC_LAUNCH:
            # Experiment (2026-09-17): a graph with host nodes makes cudaGraphLaunch
            # block the calling thread for ~90% of the graph. Launch from a helper
            # thread via the raw runtime API (ctypes releases the GIL) so the
            # scheduler's Python bookkeeping overlaps the graph. The scheduler must
            # call wait_pending_launch() before enqueuing anything else on this
            # stream (sampling, WAR barrier) -- see scheduler.event_loop_overlap.
            wait_pending_launch()
            exec_handle = self._graphs[shape_key].raw_cuda_graph_exec()
            stream_handle = torch.cuda.current_stream().cuda_stream
            _start_async_launch(exec_handle, stream_handle)
            return self._outputs[shape_key]
        self._graphs[shape_key].replay()
        return self._outputs[shape_key]

    def cleanup(self) -> None:
        self._graphs.clear()
        self._outputs.clear()
        self._pool = None

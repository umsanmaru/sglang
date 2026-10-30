"""Versioned, content-addressed loading caches. Cache misses remain authoritative.

Plan/index caching is small and enabled by default. Weight caching is opt-in:
its keys hash actual tensor bytes, so replacing a checkpoint at the same path
cannot serve stale weights. Entries are written atomically and bounded on disk.
"""
import fcntl
import hashlib
import logging
import os
from pathlib import Path
import pickle
import tempfile
from dataclasses import replace

import torch

logger = logging.getLogger(__name__)
REVISION = 'v1'


def _root():
    if os.environ.get('SGLANG_PRISM_LOAD_CACHE', '1') == '0':
        return None
    return Path(os.environ.get('SGLANG_PRISM_LOAD_CACHE_DIR',
                str(Path(os.environ.get('XDG_CACHE_HOME', str(Path.home() / '.cache'))) / 'prism' / 'loading')))


def _atomic(path, writer):
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.tmp-', delete=False) as f:
            name = f.name
            writer(f)
        os.replace(name, path)
    finally:
        if name is not None:
            Path(name).unlink(missing_ok=True)


class _PlanReader(pickle.Unpickler):
    def find_class(self, module, name):
        from sglang.srt.layers.moe.prism import plan
        allowed = ('Plan', 'ModelDims', 'KernelSpec', 'BandSpec', 'NumaShard',
                   'ExpertPlan', 'ExpertProjPlan', 'SparsitySpec', 'CalibRef', 'Tier', 'Proj')
        if name in allowed and module == getattr(plan, name).__module__:
            return getattr(plan, name)
        raise pickle.UnpicklingError(f'unsupported plan cache class {module}.{name}')


def _intern_bands(plan):
    # Block-2 plans repeat a small vocabulary of bands millions of times.
    # Sharing these objects also makes pickle memo references compact.
    bands, projections, experts, shared_experts = {}, {}, {}, {}
    def projection(pp):
        identity = id(pp)
        if identity not in projections:
            shared = []
            for b in pp.bands:
                key = b.start, b.end, b.tier
                if key not in bands:
                    bands[key] = b
                shared.append(bands[key])
            projections[identity] = replace(pp, bands=tuple(shared))
        return projections[identity]
    for key, ep in plan.experts.items():
        if id(ep) not in shared_experts:
            shared_experts[id(ep)] = replace(ep, gate=projection(ep.gate), up=projection(ep.up), down=projection(ep.down))
        experts[key] = shared_experts[id(ep)]
    return replace(plan, experts=experts)


def load_plan_cached(source):
    from sglang.srt.layers.moe.prism.plan import parse_plan, Plan
    root = _root()
    if root is None:
        return parse_plan(source)
    digest = hashlib.sha256()
    with open(source, 'rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''):
            digest.update(chunk)
    key = digest.hexdigest()
    path = root / REVISION / (key + '.plan')
    try:
        with path.open('rb') as f:
            saved_key, plan = _PlanReader(f).load()
        if saved_key != key or not isinstance(plan, Plan):
            raise ValueError('plan cache key/type mismatch')
        logger.info('[prism] plan cache hit: %s', path)
    except (OSError, EOFError, ValueError, TypeError, AttributeError, pickle.UnpicklingError):
        plan = _intern_bands(parse_plan(source))
        try:
            _atomic(path, lambda f: pickle.dump((key, plan), f, protocol=5))
        except OSError as exc:
            logger.warning('[prism] plan cache write skipped: %s', exc)
    # The caller still validates the plan and reloads/verifies calibration.
    object.__setattr__(plan, '_loading_cache_key', key)
    return plan


def load_layer_index(plan, layer):
    from sglang.srt.layers.moe.prism.index import LayerIndex, TierIndex, from_bands, validate_layer
    from sglang.srt.layers.moe.prism.plan import Proj, Tier
    root, key = _root(), getattr(plan, '_loading_cache_key', None)
    path = root / REVISION / f'{key}.{layer}.index' if root is not None and key else None
    if path is not None:
        try:
            raw = torch.load(path, map_location='cpu', weights_only=True)
            if raw['key'] != key or raw['layer'] != layer:
                raise ValueError('index cache key mismatch')
            index = LayerIndex({(Proj(p), Tier(t)): None if v is None else TierIndex(*v)
                                for p, t, v in raw['tiers']})
            validate_layer(index, plan.dims, layer)
            return index
        except (OSError, EOFError, RuntimeError, ValueError, KeyError, TypeError, pickle.UnpicklingError):
            pass
    index = from_bands(plan, layer)
    validate_layer(index, plan.dims, layer)
    if path is not None:
        raw = {'key': key, 'layer': layer,
               'tiers': [(p.value, t.value, None if v is None else (v.row_off, v.idx, v.contiguous))
                         for (p, t), v in index.tiers.items()]}
        try:
            _atomic(path, lambda f: torch.save(raw, f))
        except OSError as exc:
            logger.warning('[prism] index cache write skipped: %s', exc)
    return index


def _hash_tensor(digest, tensor):
    digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
    # No whole-model contiguous copy. Even noncontiguous projection views are
    # hashed one expert at a time; the underlying bytes are the source of truth.
    chunks = tensor if tensor.ndim > 1 else (tensor,)
    for chunk in chunks:
        digest.update(memoryview(chunk.contiguous().view(torch.uint8).numpy()))


def cached_columns(src, ti, real_rows, tile, options, compute):
    configured = os.environ.get('SGLANG_PRISM_WEIGHT_CACHE_DIR')
    if not configured:
        return compute()
    root = Path(configured) / REVISION
    limit = int(float(os.environ.get('SGLANG_PRISM_WEIGHT_CACHE_MAX_GB', '4')) * (1 << 30))
    if limit <= 0:
        return compute()
    digest = hashlib.sha256()
    digest.update(repr((REVISION, tuple(int(r) for r in real_rows), tile, options)).encode())
    for t in (src, ti.row_off, ti.idx):
        _hash_tensor(digest, t)
    key = digest.hexdigest()
    path = root / (key + '.pt')
    try:
        entry = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        if entry['key'] != key:
            raise ValueError('weight cache key mismatch')
        path.touch()
        logger.info('[prism] prepared weight cache hit: %s', path)
        return tuple(entry['tensors'])
    except (OSError, EOFError, RuntimeError, ValueError, KeyError, TypeError, pickle.UnpicklingError):
        pass
    result = compute()
    estimated = sum(t.numel() * t.element_size() for t in result) + 8192
    if estimated > limit:
        return result
    try:
        root.mkdir(parents=True, exist_ok=True)
        with (root / '.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            files = sorted(root.glob('*.pt'), key=lambda p: p.stat().st_mtime_ns)
            total = sum(p.stat().st_size for p in files)
            for old in files:
                if total + estimated <= limit:
                    break
                total -= old.stat().st_size
                old.unlink(missing_ok=True)
            _atomic(path, lambda f: torch.save({'key': key, 'tensors': result}, f))
    except OSError as exc:
        logger.warning('[prism] prepared weight cache write skipped: %s', exc)
    return result

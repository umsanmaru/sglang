"""SASS fingerprints of the Prism JIT kernels (refactor gate, K0).

A structure-only refactor (moving host code, renaming entry points) must leave every
device function's machine code unchanged. This tool builds the Prism tvm-ffi modules
into a chosen cache directory, hashes the SASS of each device function, and diffs two
such fingerprints.

    python test/prism/kernel_sass.py build <cache_dir>
    python test/prism/kernel_sass.py dump  <cache_dir> <out.json>
    python test/prism/kernel_sass.py diff  <a.json> <b.json>

Function names are demangled and stripped of anonymous-namespace and `sglang::`
qualifiers, so a function keeps its key when its file or namespace changes.

Both JIT caches are read: the vendored loader's `TVM_FFI_CACHE_DIR/<module>_<hash>/` and
upstream sglang.kernels.jit's `SGLANG_JIT_CACHE_DIR/<target>/<module>/build-*/deps-*/`.
Set both variables to the same fresh directory before `build`.
"""

from __future__ import annotations

import glob
import hashlib
import importlib
import json
import os
import re
import shutil
import subprocess
import sys

from kernel_golden import WRAPPER_MODULES

_CUDA_HOME = os.environ.get("CUDA_HOME", "/usr/local/cuda")
_SO_PREFIX = "sgl_kernel_jit_prism_"
_ADDR = re.compile(r"/\*[0-9a-f]{4,}\*/")
_ENCODING = re.compile(r"/\* 0x[0-9a-f]{16} \*/")
_QUALIFIERS = re.compile(r"\(anonymous namespace\)::|\bsglang::")


def build(cache_dir: str) -> None:
    """Compile every Prism module into `cache_dir` (both JIT cache variables must point there)."""
    for var in ("TVM_FFI_CACHE_DIR", "SGLANG_JIT_CACHE_DIR"):
        if os.path.abspath(os.environ.get(var, "")) != os.path.abspath(cache_dir):
            sys.exit(f"set {var}={cache_dir} before importing sglang")
    for modname in WRAPPER_MODULES:
        m = importlib.import_module(modname)
        getters = [
            v
            for k, v in vars(m).items()
            if k.startswith("_jit_") and k.endswith("_module")
        ]
        if len(getters) != 1:
            sys.exit(
                f"{modname}: expected one _jit_*_module getter, found {len(getters)}"
            )
        getters[0]()
        print(f"built {modname}")


def _demangle(names):
    out = subprocess.run(
        ["c++filt"], input="\n".join(names), capture_output=True, text=True, check=True
    ).stdout.splitlines()
    return [_QUALIFIERS.sub("", n) for n in out]


def _functions(so_path: str) -> dict:
    sass = subprocess.run(
        [os.path.join(_CUDA_HOME, "bin", "cuobjdump"), "-sass", so_path],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    blocks, cur, lines = {}, None, []
    for line in sass.splitlines():
        m = re.match(r"\s*Function : (\S+)", line)
        if m:
            if cur is not None:
                blocks[cur] = lines
            cur, lines = m.group(1), []
            continue
        if cur is None:
            continue
        text = _ENCODING.sub("", _ADDR.sub("", line)).strip()
        if text and not text.startswith(".headerflags"):
            lines.append(text)
    if cur is not None:
        blocks[cur] = lines
    mangled = sorted(blocks)
    result = {}
    for raw, name in zip(mangled, _demangle(mangled)):
        if name in result:
            raise RuntimeError(f"{so_path}: two functions normalize to {name!r}")
        h = hashlib.blake2b("\n".join(blocks[raw]).encode(), digest_size=12).hexdigest()
        result[name] = {"sass": h, "instructions": len(blocks[raw])}
    return result


def dump(cache_dir: str, out_json: str) -> None:
    fp = {}
    pattern = os.path.join(cache_dir, "**", _SO_PREFIX + "*.so")
    for so in sorted(glob.glob(pattern, recursive=True)):
        module = os.path.basename(so)[: -len(".so")]
        if module in fp:
            sys.exit(f"{cache_dir}: two builds of {module}; use a fresh cache dir")
        fp[module] = _functions(so)
    with open(out_json, "w") as f:
        json.dump(fp, f, indent=1, sort_keys=True)
    n = sum(len(v) for v in fp.values())
    print(f"{len(fp)} modules, {n} device functions -> {out_json}")


def diff(a_json: str, b_json: str) -> int:
    with open(a_json) as f:
        a = {
            (m, fn): v["sass"]
            for m, fns in json.load(f).items()
            for fn, v in fns.items()
        }
    with open(b_json) as f:
        b = {
            (m, fn): v["sass"]
            for m, fns in json.load(f).items()
            for fn, v in fns.items()
        }
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    changed = sorted(k for k in set(a) & set(b) if a[k] != b[k])
    for tag, keys in (("removed", only_a), ("added", only_b), ("changed", changed)):
        for m, fn in keys:
            print(f"{tag:8s} {m}: {fn}")
    print(
        f"same={len(set(a) & set(b)) - len(changed)} changed={len(changed)} "
        f"removed={len(only_a)} added={len(only_b)}"
    )
    return 1 if (only_a or only_b or changed) else 0


def main(argv) -> int:
    if len(argv) >= 2 and argv[0] == "build":
        build(argv[1])
        return 0
    if len(argv) >= 3 and argv[0] == "dump":
        if shutil.which("c++filt") is None:
            sys.exit("c++filt not found")
        dump(argv[1], argv[2])
        return 0
    if len(argv) >= 3 and argv[0] == "diff":
        return diff(argv[1], argv[2])
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

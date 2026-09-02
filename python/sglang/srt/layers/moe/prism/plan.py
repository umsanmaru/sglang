"""Prism Plan: 스키마, 파싱, 정적 검증.

계약 전문은 CONTRACTS.md ① 참조. 이 모듈은 Plan의 스키마·파서·검증기만
소유한다 — Plan 생성기는 이 코드베이스 밖이다.

이 모듈은 의도적으로 runtime을 모른다: stdlib과 `layers/prism/geometry`(그 자체가
순수 stdlib인 기하 정의) 외에 아무것도 import하지 않는다. Plan abstraction이
runtime을 알게 되는 순간 경계가 무너진다.

Plan 파일(JSON) 형식::

    {
      "schema_version": 1,
      "model_id": "Qwen/Qwen3-30B-A3B",
      "dims": {
        "hidden_size": 2048, "intermediate_size": 768,
        "num_layers": 48, "num_experts": 128, "top_k": 8,
        "dtype": "bfloat16"
      },
      "kernels": {"gpu_warm": "torch_bmm", "cpu_cold": "kt_amx_bf16"},
      "sparsity": {
        "score": "k2wl2",
        "calib": {"path": "assets/qwen35/gatedyn_calib.pt", "sha256": "<64 hex>"},
        "pmax": 0.9, "grid": 0.005, "ng": 201, "renorm_it": 3
      },
      "default": {
        "gate": {"bands": [[0, 192, "warm"], [192, 2048, "cold"]],
                  "cold_shards": [[0, 0, 384], [1, 384, 768]],
                  "p": 0.5, "lambda": 4.305},
        "up":   {...},
        "down": {...}
      },
      "overrides": [
        {"layer": 3, "expert": 17, "gate": {...}, "up": {...}, "down": {...}}
      ]
    }

"sparsity"는 model-global이고 생략 가능하다 (없으면 dense = 현행 동작).
있으면 모든 proj가 예산 (p, lambda)를 가져야 한다 — threshold **값**은
Plan에 없다. calib 자산의 곡선에서 step마다 조회된다:

    s   = clip(p - lambda*(g_e - g_mean), 0, pmax)   # renorm_it회 재정규화 후
    thr = table[layer, expert, round(s / grid)]

"default"는 모든 (layer, expert)에 적용되고 "overrides"가 개별 항목을
대체한다. 파싱 결과(메모리 표현)는 항상 (layer, expert) 완전 명시형이다 —
default/overrides는 파일 형식의 sugar일 뿐 추상화에는 존재하지 않는다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, Union

# 축 무관한 기하는 공유 코어가 소유한다 (2026-08-31 승격). 여기서 re-export하는
# 것은 기존 import 경로(`moe.prism.plan.Tier` 등)를 살리기 위해서다 — 이 모듈의
# 나머지(Proj, ModelDims, ExpertPlan, 파서)는 expert 축에 묶여 있어 안 올라갔다.
from sglang.srt.layers.prism.geometry import (  # noqa: F401
    COL_GROUP,
    PAIR_GROUP,
    ROW_GROUP,
    BandSpec,
    KernelSpec,
    NumaShard,
    PlanError,
    Tier,
)

SUPPORTED_SCHEMA_VERSIONS = (1, 2)
# sparsity 블록이 등장할 수 있는 최소 schema_version
SPARSITY_SCHEMA_VERSION = 2
# sparsity score 변종 — calib 자산의 어느 테이블 계열을 쓰는지.
# k2wl2 = 인접 페어의 실제 에너지(교차항 포함)이므로 wn(열 노름)과
# pair_dot(인접열 내적)을 모두 요구한다.
KNOWN_SPARSITY_SCORES = ("k2wl2",)


class Proj(str, Enum):
    GATE = "gate"
    UP = "up"
    DOWN = "down"


@dataclass(frozen=True)
class ModelDims:
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_experts: int
    top_k: int
    dtype: str

    def k_of(self, proj: Proj) -> int:
        """proj의 contraction 축 길이."""
        return self.intermediate_size if proj is Proj.DOWN else self.hidden_size

    def n_of(self, proj: Proj) -> int:
        """proj의 출력 축 길이."""
        return self.hidden_size if proj is Proj.DOWN else self.intermediate_size


@dataclass(frozen=True)
class CalibRef:
    """sparsity 테이블 자산의 참조. 내용은 이 모듈이 열지 않는다 (순수 stdlib).

    경로 해석과 로드는 로더의 몫이고, 여기는 참조와 무결성 해시만 소유한다.
    """

    path: str
    sha256: str


@dataclass(frozen=True)
class SparsitySpec:
    """model-global sparsity 설정 (계약 ①의 kernels와 같은 급).

    per-(layer, expert, proj)로 갈리는 것은 ExpertProjPlan의
    (sparsity_p, sparsity_lambda)뿐이다. threshold **값**은 Plan에 없다 —
    calib 곡선에서 step마다 조회된다. 마스크는 각 proj의 K(contraction) 축에
    걸리고 점수는 그 proj의 입력에서만 나오므로(gate/up은 layer 입력, down은
    act), 티어별로 로컬하게 적용할 수 있다 — rejoin을 기다릴 필요가 없다.
    """

    score: str
    calib: CalibRef
    pmax: float
    grid: float
    ng: int
    renorm_it: int

    def expected_calib_shapes(self, dims: ModelDims) -> Mapping[str, tuple[int, ...]]:
        """calib 자산이 가져야 하는 논리 테이블 -> shape.

        논리명과 자산 키의 매핑은 로더가 소유한다 (이 모듈은 자산 포맷의
        어휘를 모른다). validate_static에 calib_probe가 주어지면 이 shape와
        대조된다 — 다른 모델/설정의 calib을 적용하는 것은 dims 불일치와 같은
        급의 silent failure이므로 startup 즉사다.
        """
        L, E = dims.num_layers, dims.num_experts
        shapes: dict[str, tuple[int, ...]] = {}
        for proj in Proj:
            K = dims.k_of(proj)
            shapes[f"thr_{proj.value}"] = (L, E, self.ng)
            shapes[f"wn_{proj.value}"] = (L, E, K)
            shapes[f"pair_dot_{proj.value}"] = (L, E, K // PAIR_GROUP)
        return shapes


@dataclass(frozen=True)
class ExpertProjPlan:
    """한 (expert, proj)의 분할 기하. bands는 start 오름차순."""

    bands: tuple[BandSpec, ...]
    cold_shards: tuple[NumaShard, ...]
    # sparsity 예산 (threshold가 아니다 — 곡선 조회의 입력). Plan.sparsity가
    # 있으면 반드시 둘 다 존재, 없으면 반드시 둘 다 None (validate가 강제).
    sparsity_p: Optional[float] = None
    sparsity_lambda: Optional[float] = None

    def rows(self, tier: Tier) -> int:
        return sum(b.end - b.start for b in self.bands if b.tier is tier)

    def has_tier(self, tier: Tier) -> bool:
        return any(b.tier is tier for b in self.bands)


@dataclass(frozen=True)
class ExpertPlan:
    """한 (layer, expert)의 기하. 커널 선택은 여기 없다 — model-global."""

    gate: ExpertProjPlan
    up: ExpertProjPlan
    down: ExpertProjPlan

    def proj(self, p: Proj) -> ExpertProjPlan:
        return {Proj.GATE: self.gate, Proj.UP: self.up, Proj.DOWN: self.down}[p]


@dataclass(frozen=True)
class Plan:
    schema_version: int
    model_id: str
    dims: ModelDims
    kernels: KernelSpec
    # (layer, expert) → ExpertPlan. 항상 완전 명시형 (validate가 강제).
    experts: Mapping[tuple[int, int], ExpertPlan]
    # None이면 dense. schema_version 1 plan은 항상 None이다.
    sparsity: Optional[SparsitySpec] = None

    def expert(self, layer: int, expert: int) -> ExpertPlan:
        return self.experts[(layer, expert)]


# ---------------------------------------------------------------------------
# 파싱
# ---------------------------------------------------------------------------


def _parse_proj(obj: dict, where: str) -> ExpertProjPlan:
    try:
        bands = tuple(
            BandSpec(int(s), int(e), Tier(t)) for s, e, t in obj["bands"]
        )
        shards = tuple(
            NumaShard(int(n), int(a), int(b))
            for n, a, b in obj.get("cold_shards", [])
        )
        # p/lambda는 쌍으로만 유효 — 한쪽만 있으면 KeyError로 즉사한다.
        p = lam = None
        if "p" in obj or "lambda" in obj:
            p, lam = float(obj["p"]), float(obj["lambda"])
    except (KeyError, TypeError, ValueError) as err:
        raise PlanError(f"{where}: malformed proj entry: {err}") from err
    return ExpertProjPlan(
        bands=bands, cold_shards=shards, sparsity_p=p, sparsity_lambda=lam
    )


def _parse_sparsity(obj: Optional[dict]) -> Optional[SparsitySpec]:
    """model-global sparsity 블록. 없으면 None (dense)."""
    if obj is None:
        return None
    try:
        calib = obj["calib"]
        return SparsitySpec(
            score=str(obj["score"]),
            calib=CalibRef(path=str(calib["path"]), sha256=str(calib["sha256"])),
            pmax=float(obj["pmax"]),
            grid=float(obj["grid"]),
            ng=int(obj["ng"]),
            renorm_it=int(obj["renorm_it"]),
        )
    except (KeyError, TypeError, ValueError) as err:
        raise PlanError(f"malformed sparsity block: {err}") from err


def _parse_expert(obj: dict, where: str) -> ExpertPlan:
    for key in ("gate", "up", "down"):
        if key not in obj:
            raise PlanError(f"{where}: missing proj '{key}'")
    return ExpertPlan(
        gate=_parse_proj(obj["gate"], f"{where}.gate"),
        up=_parse_proj(obj["up"], f"{where}.up"),
        down=_parse_proj(obj["down"], f"{where}.down"),
    )


def parse_plan(source: Union[str, Path, dict]) -> Plan:
    """JSON 파일 경로 또는 이미 로드된 dict에서 Plan을 만든다.

    구조 오류만 여기서 잡는다. 의미 오류(커버리지, 정렬, ...)는
    validate_static의 몫 — 파서는 스키마를 해석할 뿐 판단하지 않는다.
    """
    if isinstance(source, (str, Path)):
        try:
            raw = json.loads(Path(source).read_text())
        except (OSError, json.JSONDecodeError) as err:
            raise PlanError(f"cannot read plan file {source}: {err}") from err
    else:
        raw = source

    try:
        version = int(raw["schema_version"])
        model_id = str(raw["model_id"])
        d = raw["dims"]
        dims = ModelDims(
            hidden_size=int(d["hidden_size"]),
            intermediate_size=int(d["intermediate_size"]),
            num_layers=int(d["num_layers"]),
            num_experts=int(d["num_experts"]),
            top_k=int(d["top_k"]),
            dtype=str(d["dtype"]),
        )
        k = raw["kernels"]
        kernels = KernelSpec(gpu_warm=str(k["gpu_warm"]), cpu_cold=str(k["cpu_cold"]))
    except (KeyError, TypeError, ValueError) as err:
        raise PlanError(f"malformed plan header: {err}") from err

    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise PlanError(
            f"unsupported schema_version {version} "
            f"(supported: {SUPPORTED_SCHEMA_VERSIONS})"
        )

    if "sparsity" in raw and version < SPARSITY_SCHEMA_VERSION:
        raise PlanError(
            f"sparsity block requires schema_version >= "
            f"{SPARSITY_SCHEMA_VERSION}, got {version}"
        )
    sparsity = _parse_sparsity(raw.get("sparsity"))

    experts: dict[tuple[int, int], ExpertPlan] = {}
    default = raw.get("default")
    if default is not None:
        shared = _parse_expert(default, "default")
        for layer in range(dims.num_layers):
            for expert in range(dims.num_experts):
                experts[(layer, expert)] = shared

    for i, ov in enumerate(raw.get("overrides", [])):
        where = f"overrides[{i}]"
        try:
            key = (int(ov["layer"]), int(ov["expert"]))
        except (KeyError, TypeError, ValueError) as err:
            raise PlanError(f"{where}: malformed layer/expert: {err}") from err
        experts[key] = _parse_expert(ov, where)

    return Plan(
        schema_version=version,
        model_id=model_id,
        dims=dims,
        kernels=kernels,
        experts=experts,
        sparsity=sparsity,
    )


# ---------------------------------------------------------------------------
# 정적 검증 — Plan 자체만 보고 판단 가능한 것 전부.
# 실행 환경(메모리 예산, 실제 NUMA topology, capture-bs)이 필요한 검증은
# validate_runtime_feasibility(별도 모듈)의 몫이다.
# ---------------------------------------------------------------------------


def _validate_bands(
    proj_plan: ExpertProjPlan, K: int, where: str
) -> None:
    bands = proj_plan.bands
    if not bands:
        raise PlanError(f"{where}: no bands")
    cursor = 0
    for b in bands:
        if b.start != cursor:
            kind = "overlap" if b.start < cursor else "gap"
            raise PlanError(
                f"{where}: band {kind} at row {min(b.start, cursor)} "
                f"(expected start {cursor}, got {b.start})"
            )
        if b.end <= b.start:
            raise PlanError(f"{where}: empty/negative band [{b.start}, {b.end})")
        if b.start % ROW_GROUP or b.end % ROW_GROUP:
            raise PlanError(
                f"{where}: band [{b.start}, {b.end}) not aligned to "
                f"ROW_GROUP={ROW_GROUP}"
            )
        cursor = b.end
    if cursor != K:
        raise PlanError(f"{where}: bands cover [0, {cursor}) but K={K}")


def _validate_shards(
    proj_plan: ExpertProjPlan, N: int, where: str
) -> None:
    shards = proj_plan.cold_shards
    if not proj_plan.has_tier(Tier.COLD):
        if shards:
            raise PlanError(f"{where}: cold_shards present but no COLD band")
        return
    if not shards:
        raise PlanError(f"{where}: COLD band exists but cold_shards is empty")
    cursor = 0
    for s in shards:
        if s.node < 0:
            raise PlanError(f"{where}: negative numa node {s.node}")
        if s.n_start != cursor:
            kind = "overlap" if s.n_start < cursor else "gap"
            raise PlanError(
                f"{where}: shard {kind} at col {min(s.n_start, cursor)} "
                f"(expected start {cursor}, got {s.n_start})"
            )
        if s.n_end <= s.n_start:
            raise PlanError(f"{where}: empty/negative shard [{s.n_start}, {s.n_end})")
        if s.n_start % COL_GROUP or s.n_end % COL_GROUP:
            raise PlanError(
                f"{where}: shard [{s.n_start}, {s.n_end}) not aligned to "
                f"COL_GROUP={COL_GROUP}"
            )
        cursor = s.n_end
    if cursor != N:
        raise PlanError(f"{where}: shards cover [0, {cursor}) but N={N}")


def _validate_sparsity_spec(
    spec: SparsitySpec,
    dims: ModelDims,
    calib_probe: Optional[Callable[[CalibRef], Mapping[str, Sequence[int]]]],
) -> None:
    if spec.score not in KNOWN_SPARSITY_SCORES:
        raise PlanError(
            f"unknown sparsity.score '{spec.score}' "
            f"(known: {sorted(KNOWN_SPARSITY_SCORES)})"
        )
    if not 0.0 < spec.pmax <= 1.0:
        raise PlanError(f"sparsity.pmax must be in (0, 1], got {spec.pmax}")
    if spec.grid <= 0.0:
        raise PlanError(f"sparsity.grid must be positive, got {spec.grid}")
    if spec.ng < 2:
        raise PlanError(f"sparsity.ng must be >= 2, got {spec.ng}")
    # 격자가 pmax를 못 덮으면 idx가 상단에서 clamp되어 threshold가 조용히
    # 포화한다 — 의도보다 덜/더 자르는 무증상 오차가 되므로 즉사.
    span = (spec.ng - 1) * spec.grid
    if span + 1e-9 < spec.pmax:
        raise PlanError(
            f"sparsity grid spans [0, {span}] but pmax={spec.pmax} — "
            f"(ng-1)*grid must reach pmax"
        )
    if spec.renorm_it < 0:
        raise PlanError(f"sparsity.renorm_it must be >= 0, got {spec.renorm_it}")
    if not spec.calib.path:
        raise PlanError("sparsity.calib.path must be non-empty")
    digest = spec.calib.sha256.lower()
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise PlanError(
            f"sparsity.calib.sha256 must be 64 hex chars, got {spec.calib.sha256!r}"
        )
    for proj in Proj:
        if dims.k_of(proj) % PAIR_GROUP:
            raise PlanError(
                f"K of {proj.value} ({dims.k_of(proj)}) not divisible by "
                f"PAIR_GROUP={PAIR_GROUP}"
            )

    if calib_probe is None:
        return
    expected = spec.expected_calib_shapes(dims)
    actual = calib_probe(spec.calib)
    missing = sorted(set(expected) - set(actual))
    if missing:
        raise PlanError(f"calib asset missing tables: {missing}")
    for name in sorted(expected):
        got = tuple(int(v) for v in actual[name])
        if got != expected[name]:
            raise PlanError(
                f"calib table '{name}' shape {got} != expected {expected[name]} — "
                f"calib이 다른 모델/설정에 대해 생성된 것일 가능성"
            )


def _validate_proj_sparsity(
    proj_plan: ExpertProjPlan, spec: Optional[SparsitySpec], where: str
) -> None:
    """예산 존재 여부는 all-or-nothing (cold_shards와 같은 대칭 규칙)."""
    p, lam = proj_plan.sparsity_p, proj_plan.sparsity_lambda
    if spec is None:
        if p is not None or lam is not None:
            raise PlanError(
                f"{where}: sparsity budget present but no model-global "
                f"sparsity block"
            )
        return
    if p is None or lam is None:
        raise PlanError(f"{where}: sparsity block exists but proj has no (p, lambda)")
    if not 0.0 <= p <= spec.pmax:
        raise PlanError(f"{where}: p={p} not in [0, pmax={spec.pmax}]")
    if lam < 0.0:
        raise PlanError(f"{where}: lambda={lam} must be >= 0")


def validate_static(
    plan: Plan,
    known_gpu_kernels: Optional[Sequence[str]] = None,
    known_cpu_kernels: Optional[Sequence[str]] = None,
    calib_probe: Optional[Callable[[CalibRef], Mapping[str, Sequence[int]]]] = None,
) -> None:
    """Plan 자체의 정합성 검증. 위반은 전부 PlanError (startup hard error).

    커널 registry가 주어지면 이름 존재도 확인한다. calib_probe가 주어지면
    sparsity 자산의 테이블 shape까지 확인한다 (probe는 CalibRef를 받아
    "논리 테이블명 -> shape"를 돌려주는 콜러블 — 자산을 여는 것은 호출자다). dims와 실제 모델 config의
    대조는 호출자(로더)가 이 함수 호출 직전에 수행한다 — 이 모듈은 모델
    config의 존재를 모른다.
    """
    dims = plan.dims
    for name, val in (
        ("hidden_size", dims.hidden_size),
        ("intermediate_size", dims.intermediate_size),
        ("num_layers", dims.num_layers),
        ("num_experts", dims.num_experts),
        ("top_k", dims.top_k),
    ):
        if val <= 0:
            raise PlanError(f"dims.{name} must be positive, got {val}")
    for proj in Proj:
        if dims.k_of(proj) % ROW_GROUP:
            raise PlanError(
                f"K of {proj.value} ({dims.k_of(proj)}) not divisible by "
                f"ROW_GROUP={ROW_GROUP}"
            )
        if dims.n_of(proj) % COL_GROUP:
            raise PlanError(
                f"N of {proj.value} ({dims.n_of(proj)}) not divisible by "
                f"COL_GROUP={COL_GROUP}"
            )

    if not plan.kernels.gpu_warm or not plan.kernels.cpu_cold:
        raise PlanError("kernels.gpu_warm / kernels.cpu_cold must be non-empty")
    if known_gpu_kernels is not None and plan.kernels.gpu_warm not in known_gpu_kernels:
        raise PlanError(
            f"unknown gpu_warm kernel '{plan.kernels.gpu_warm}' "
            f"(known: {sorted(known_gpu_kernels)})"
        )
    if known_cpu_kernels is not None and plan.kernels.cpu_cold not in known_cpu_kernels:
        raise PlanError(
            f"unknown cpu_cold kernel '{plan.kernels.cpu_cold}' "
            f"(known: {sorted(known_cpu_kernels)})"
        )

    if plan.sparsity is not None:
        _validate_sparsity_spec(plan.sparsity, dims, calib_probe)

    expected_keys = {
        (layer, expert)
        for layer in range(dims.num_layers)
        for expert in range(dims.num_experts)
    }
    actual_keys = set(plan.experts.keys())
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)[:5]
        extra = sorted(actual_keys - expected_keys)[:5]
        raise PlanError(
            f"experts must cover every (layer, expert) exactly: "
            f"missing {len(expected_keys - actual_keys)} (e.g. {missing}), "
            f"extra {len(actual_keys - expected_keys)} (e.g. {extra})"
        )

    # 동일 ExpertPlan 객체(default 공유)는 한 번만 검증한다.
    seen: set[int] = set()
    for (layer, expert), ep in plan.experts.items():
        if id(ep) in seen:
            continue
        seen.add(id(ep))
        where_prefix = f"experts[({layer}, {expert})]"
        for proj in Proj:
            pp = ep.proj(proj)
            where = f"{where_prefix}.{proj.value}"
            _validate_bands(pp, dims.k_of(proj), where)
            _validate_shards(pp, dims.n_of(proj), where)
            _validate_proj_sparsity(pp, plan.sparsity, where)

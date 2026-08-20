"""Prism Plan: 스키마, 파싱, 정적 검증.

계약 전문은 CONTRACTS.md ① 참조. 이 모듈은 Plan의 스키마·파서·검증기만
소유한다 — Plan 생성기는 이 코드베이스 밖이다.

이 모듈은 의도적으로 sglang의 다른 어떤 모듈에도 의존하지 않는다
(순수 stdlib). Plan abstraction이 runtime을 알게 되는 순간 경계가 무너진다.

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
      "default": {
        "gate": {"bands": [[0, 192, "warm"], [192, 2048, "cold"]],
                  "cold_shards": [[0, 0, 384], [1, 384, 768]]},
        "up":   {...},
        "down": {...}
      },
      "overrides": [
        {"layer": 3, "expert": 17, "gate": {...}, "up": {...}, "down": {...}}
      ]
    }

"default"는 모든 (layer, expert)에 적용되고 "overrides"가 개별 항목을
대체한다. 파싱 결과(메모리 표현)는 항상 (layer, expert) 완전 명시형이다 —
default/overrides는 파일 형식의 sugar일 뿐 추상화에는 존재하지 않는다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

# K-축 밴드 경계 정렬 단위 (계약 ①)
ROW_GROUP = 64
# N-축 shard 경계 정렬 단위 (AMX pack N-타일; 값은 pack 확인 후 조정 가능)
COL_GROUP = 32

SUPPORTED_SCHEMA_VERSIONS = (1,)


class PlanError(ValueError):
    """Plan 파싱/검증 실패. 전부 startup hard error다."""


class Tier(str, Enum):
    HOT = "hot"    # VRAM 상주, GPU 계산
    WARM = "warm"  # pinned host 상주, step마다 선택 밴드만 GPU 전송, GPU 계산
    COLD = "cold"  # pageable host(NUMA-local) 상주, CPU 계산


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
class BandSpec:
    """K-축 반개구간 [start, end)와 그 티어. 경계는 ROW_GROUP 배수."""

    start: int
    end: int
    tier: Tier


@dataclass(frozen=True)
class NumaShard:
    """cold 출력의 N-축 반개구간 [n_start, n_end)를 담당하는 NUMA node."""

    node: int
    n_start: int
    n_end: int


@dataclass(frozen=True)
class ExpertProjPlan:
    """한 (expert, proj)의 분할 기하. bands는 start 오름차순."""

    bands: tuple[BandSpec, ...]
    cold_shards: tuple[NumaShard, ...]

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
class KernelSpec:
    """model-global 커널 선택. startup에 구현체로 resolve된 뒤 문자열은 소멸.

    cold의 저장 형식(pack)은 cpu_cold 키가 함의한다 (별도 codec 없음).
    """

    gpu_warm: str
    cpu_cold: str


@dataclass(frozen=True)
class Plan:
    schema_version: int
    model_id: str
    dims: ModelDims
    kernels: KernelSpec
    # (layer, expert) → ExpertPlan. 항상 완전 명시형 (validate가 강제).
    experts: Mapping[tuple[int, int], ExpertPlan]

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
    except (KeyError, TypeError, ValueError) as err:
        raise PlanError(f"{where}: malformed proj entry: {err}") from err
    return ExpertProjPlan(bands=bands, cold_shards=shards)


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


def validate_static(
    plan: Plan,
    known_gpu_kernels: Optional[Sequence[str]] = None,
    known_cpu_kernels: Optional[Sequence[str]] = None,
) -> None:
    """Plan 자체의 정합성 검증. 위반은 전부 PlanError (startup hard error).

    커널 registry가 주어지면 이름 존재도 확인한다. dims와 실제 모델 config의
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

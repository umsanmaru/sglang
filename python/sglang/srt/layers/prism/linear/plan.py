"""Prism dense Plan: 스키마, 파싱, 정적 검증.

`moe/prism/plan.py`의 dense 대응물. 기하(`Tier`/`BandSpec`/`NumaShard`/정렬 상수)는
`layers/prism/geometry.py`가 소유하고, 여기는 **좌표계와 파서**만 갖는다.

MoE plan과 갈리는 지점 셋:

  * **좌표가 `(layer, proj)`이고 proj가 열린 이름이다.** MoE는 `gate|up|down`
    enum이면 충분했다 — 세 개뿐이고 K가 `hidden`/`intermediate`에서 파생된다.
    dense는 모델마다 projection 집합이 다르고 K도 파생 공식이 없다. 그래서 이름이
    문자열이고 **k/n을 plan이 직접 적는다**. 그 값은 로드 타임에 실제 layer의 치수와
    대조된다 (`check_dims`) — 다른 모델의 plan을 먹이는 것은 조용히 틀린 슬라이스가
    되므로 startup 즉사가 맞다.

  * **커널(=스토어 형식)이 proj마다 다를 수 있다.** top-level `kernels`가 기본이고
    `projs[name].kernels`가 그 proj만 덮는다. dense는 한 모델 안에서 형식이 갈린다:
    DSV4의 `wo_a`는 `SGLANG_OPT_FP8_WO_A_GEMM`이 꺼져 있으면 `quant_config=None` +
    `params_dtype=bfloat16`으로 만들어져, 같은 모델의 `wq_b`/`wo_b`가 fp8인데 혼자
    bf16이다 (`models/deepseek_v4.py:641`).

  * **한 linear가 N축으로 쪼개질 수 있다 (`parts`).** `mlp.gate_up_proj`는
    `MergedColumnParallelLinear`라 weight가 `[2I, K]` 하나지만, **sparsity가 gate와
    up을 따로 캘리브한다** — 자산의 `wn_g ≠ wn_u`(상관 0.71), `tg2l ≠ tu2l`이고 MoE
    `_GateUpSparse`도 `gate_spec`/`up_spec`을 각각 받는다. 마스크가 K축인데 두 절반이
    다른 마스크를 요구하므로 한 번의 GEMV로 N=2I를 훑을 수 없다. 그래서 로드 시
    N축으로 쪼개 **절반마다 자기 밴딩·예산·스토어**를 준다. 쪼갠 뒤 구조가 MoE의
    gate/up과 같아지므로 기존 융합 커널(`gemv_gateup`)을 그대로 쓴다.

    `parts`는 **리스트**다 — 순서와 크기가 둘 다 구조로 드러나야 한다. `qkv_proj`는
    `[12288, 1024, 1024]`로 불균등하고(q에 attention 게이트가 실려 2배), dict 순서에
    정확성을 걸면 q 자리에 k가 들어가도 절단은 성공한다. 어느 calib 테이블을 쓰는지는
    각 조각의 `calib` 키가 말한다.

proj 이름은 sglang의 런타임 prefix에서 `model[.language_model].layers.<N>.`을 뗀
나머지다 (`split_prefix`). 그 규약을 여기 두는 이유는, "무엇이 proj인가"를 정하는
것이 plan의 어휘이기 때문이다 — 훅(registry predicate)은 이 함수를 부를 뿐 이름
규칙을 자기가 알지 않는다.

Plan 파일(JSON) 형식::

    {
      "schema_version": 1,
      "model_id": "Qwen/Qwen3.8-27B",
      "dims": {"num_layers": 64, "dtype": "bfloat16"},
      "kernels": {"gpu_warm": "gemv_worklist", "cpu_cold": "kt_tile_k2_bf16"},
      "sparsity": {
        "score": "k2wl2",
        "calib": {"path": "assets/qwen38_27b.pt", "sha256": "<64 hex>"},
        "pmax": 0.9, "grid": 0.005, "ng": 201, "renorm_it": 3
      },
      "projs": {
        "mlp.gate_up_proj": {
          "k": 5120, "n": 34816,
          "parts": [
            {"name": "gate", "n": 17408, "calib": "g", "p": 0.5, "lambda": 0.0,
             "bands": [[0, 512, "hot"], [512, 5120, "cold"]],
             "cold_shards": [[0, 0, 8704], [1, 8704, 17408]]},
            {"name": "up", "n": 17408, "calib": "u", "p": 0.5, "lambda": 0.0,
             "bands": [[0, 256, "hot"], [256, 5120, "cold"]],
             "cold_shards": [[0, 0, 8704], [1, 8704, 17408]]}
          ]
        },
        "self_attn.output_gate_proj": {          # calib이 안 덮는다 → 명시적 제외
          "k": 6656, "n": 4096, "sparse": false,
          "bands": [[0, 6656, "cold"]], "cold_shards": [[0, 0, 4096]]
        },
        "mlp.down_proj": {
          "k": 17408, "n": 5120,
          "bands": [[0, 17408, "cold"]],
          "cold_shards": [[0, 0, 2560], [1, 2560, 5120]],
          "p": 0.5, "lambda": 0.0
        }
      },
      "overrides": [
        {"layer": 3, "mlp.down_proj": {"bands": [[0, 17408, "hot"]]}}
      ]
    }

`projs`가 모델 기하(k/n)와 기본 밴딩을 함께 준다. `overrides`는 특정 layer의
밴딩만 갈아끼운다 — k/n/kernels/조각 구조는 모델 기하라 layer마다 같으므로
override 대상이 아니다. MoE와 마찬가지로 default/overrides는 **파일 형식의
sugar**이고, 파싱 결과는 항상 `(layer, proj)` 완전 명시형이다.

`projs`에 없는 projection은 Prism이 건드리지 않는다 — 그 layer는 stock 경로로
돈다. "전부 오프로드"를 뜻하는 축약형은 없다: 무엇을 오프로드하는지가 곧 실험
변수이므로 명시가 기본이다.

**sparsity (dense에서 달라지는 것).** `lambda`는 **효과가 없다.** kt
`slot_sparsity`(`moe_base.hpp:625`)에 k=1을 넣으면 `w·inv = 1`, `gbar = 1`이 되어
`s = clip(p − lam·0) = clip(p)`이고 재정규화도 멱등이다. 자산도 그것을 알고
`lam0 = 0.0`으로 생성됐다. 즉 **예산은 정적이고 마스크만 활성화 의존**이다 —
`imp_j`가 x의 함수이므로 토큰마다 다른 행이 죽는다. 필드를 남겨 둔 것은 자산·
kt 인터페이스와 어휘를 맞추기 위해서다.

sparse 티어는 **WARM + COLD**다 (HOT은 마스킹하지 않는다 — MoE `tiers.SPARSE_TIERS`와
같은 선택). 대가도 같다: 세 티어 마스크의 합집합이 full-K 마스크가 아니므로 같은
행을 warm↔hot으로 옮기면 출력이 달라진다 — **sparse plan에서는 계약 ⑤의 배치
불변성이 성립하지 않는다.**
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple, Union

from sglang.srt.layers.prism.geometry import (
    COL_GROUP,
    ROW_GROUP,
    BandSpec,
    KernelSpec,
    NumaShard,
    PlanError,
    Tier,
)
from sglang.srt.layers.prism.kernels import resolve_cpu_kernel, resolve_gpu_kernel

SUPPORTED_SCHEMA_VERSIONS = (1,)

# sparsity score 변종 — calib 자산의 어느 테이블 계열을 쓰는지 (MoE와 동일).
KNOWN_SPARSITY_SCORES = ("k2wl2",)

# sglang 런타임 prefix → (layer_idx, proj_name).
# 예: "model.layers.7.self_attn.wq_b" → (7, "self_attn.wq_b")
#
# `.language_model`은 **선택적**이다. 멀티모달 래퍼(Qwen3.8-27B 같은
# `*ForConditionalGeneration`)는 텍스트 스택을 `model.language_model.layers.N.`으로
# 짓는다 — 모델 파일 자신도 같은 형태의 정규식을 쓴다(`models/qwen3_5.py`의
# `_QWEN3_5_LORA_PATTERN`). 이 갈래를 안 받으면 plan은 정상 로드되는데 predicate가
# 아무것도 매치하지 않고 **에러도 안 난다** — "켰는데 안 켜진" 무증상 상태다.
_PREFIX_RE = re.compile(r"^model(?:\.language_model)?\.layers\.(\d+)\.(.+)$")


def split_prefix(prefix: str) -> Optional[Tuple[int, str]]:
    """LinearBase의 prefix를 plan 좌표로. 디코더 레이어 밖이면 None.

    None이 나오는 것은 정상이다 — lm_head, embedding 이웃, vision tower가 전부
    같은 훅을 지난다 (`layers/linear_method_registry.py` 참조).
    """
    m = _PREFIX_RE.match(prefix)
    if m is None:
        return None
    return int(m.group(1)), m.group(2)


@dataclass(frozen=True)
class ModelDims:
    """dense plan의 model-global 치수. proj별 k/n은 여기 없다 (ProjPlan이 갖는다)."""

    num_layers: int
    dtype: str


@dataclass(frozen=True)
class CalibRef:
    """sparsity 테이블 자산의 참조. 내용은 이 모듈이 열지 않는다 (순수 stdlib)."""

    path: str
    sha256: str


@dataclass(frozen=True)
class SparsitySpec:
    """model-global sparsity 설정.

    MoE `plan.SparsitySpec`과 필드가 같지만 타입은 따로다: 자산 shape 기대치가 축에
    묶여 있어(MoE는 `[L, E, K]`, dense는 `[L, 1, K]` + 이름 축) 한 타입이 둘을 다
    표현하면 검증이 헐거워진다. 스칼라들은 자산이 정하는 값이므로 드리프트는 로드 시
    shape/조회 오류로 즉시 드러난다.
    """

    score: str
    calib: CalibRef
    pmax: float
    grid: float
    ng: int
    renorm_it: int


@dataclass(frozen=True)
class ProjPart:
    """한 projection의 N축 조각 하나. 분할이 없으면 조각이 하나(`name=None`)다.

    K 밴딩·cold shard·sparsity 예산이 **조각마다** 따로다 — 그것이 분할의 이유다.
    """

    name: Optional[str]      # None | "gate"/"up"/"q"/"k"/"v" — 조각 이름
    n_start: int             # 이 조각이 차지하는 weight 행 [n_start, n_end)
    n_end: int
    bands: Tuple[BandSpec, ...]
    cold_shards: Tuple[NumaShard, ...]
    # 이 조각이 쓰는 calib 테이블 키 (`"g"`/`"u"`/`"d"`/`"q"`/`"k"`/`"v"`/`"o"`).
    # **plan이 말한다** — 어댑터가 모델의 이름 규약을 몰라도 되게. 모델마다
    # projection 이름이 다르므로(`wq_b` vs `q_proj` vs `in_proj_qkv`) 매핑을
    # 코드에 박으면 모델이 늘 때마다 어댑터를 고쳐야 한다.
    calib: Optional[str] = None
    # 이 조각을 마스킹하는가. plan.sparsity가 있어도 **명시적으로** 끌 수 있다 —
    # calib이 안 덮는 projection(Muse-Glimmer의 `output_gate_proj`)이 실재하기
    # 때문이다. 기본이 True인 이유는 빠뜨림과 의도적 제외를 구분하기 위해서다:
    # 그냥 안 적으면 "calib/p/lambda 누락"으로 죽고, 빼려면 `false`를 적어야 한다.
    sparse: bool = True
    # 예산 (threshold가 아니다 — 곡선 조회의 입력).
    sparsity_p: Optional[float] = None
    sparsity_lambda: Optional[float] = None

    @property
    def n(self) -> int:
        return self.n_end - self.n_start

    def rows(self, tier: Tier) -> int:
        return sum(b.end - b.start for b in self.bands if b.tier is tier)

    def has_tier(self, tier: Tier) -> bool:
        return any(b.tier is tier for b in self.bands)


@dataclass(frozen=True)
class ProjPlan:
    """한 (layer, linear)의 계획. `parts`는 weight 행 순서대로다."""

    name: str
    k: int  # contraction 축 = weight의 input_size
    n: int  # 출력 축 전체 = weight의 output_size (merged면 합)
    # 이 proj의 커널 쌍 = 스토어 형식 (계약 ①). top-level 기본값 또는 proj별 덮어쓰기.
    kernels: KernelSpec
    parts: Tuple[ProjPart, ...]

    @property
    def split(self) -> bool:
        return len(self.parts) > 1

    @property
    def sole(self) -> "ProjPart":
        """분할이 없는 linear의 유일한 조각.

        분할된 것에 부르면 즉사한다 — "조각이 하나겠지"라는 가정이 gate만 보고 up을
        빠뜨리는 형태로 조용히 틀리는 것을 막는다.
        """
        if len(self.parts) != 1:
            raise ValueError(
                f"{self.name}: split into {len(self.parts)} parts "
                f"({[p.name for p in self.parts]}) — pick one with .part(name)"
            )
        return self.parts[0]

    def part(self, name: Optional[str]) -> ProjPart:
        for p in self.parts:
            if p.name == name:
                return p
        raise KeyError(f"{self.name}: no part {name!r}")

    def has_tier(self, tier: Tier) -> bool:
        """조각 중 하나라도 이 티어를 쓰는가."""
        return any(p.has_tier(tier) for p in self.parts)


@dataclass(frozen=True)
class LinearPlan:
    schema_version: int
    model_id: str
    dims: ModelDims
    # 기본 커널 쌍. proj가 자기 것을 선언하면 그것이 이긴다 (`ProjPlan.kernels`).
    kernels: KernelSpec
    # (layer, proj_name) → ProjPlan. 항상 완전 명시형 (parse가 전개한다).
    projs: Mapping[Tuple[int, str], ProjPlan]
    # None이면 dense (마스킹 없음).
    sparsity: Optional[SparsitySpec] = None

    def proj(self, layer: int, name: str) -> ProjPlan:
        return self.projs[(layer, name)]

    def get(self, layer: int, name: str) -> Optional[ProjPlan]:
        """plan에 없으면 None — 그 linear는 stock 경로로 둔다."""
        return self.projs.get((layer, name))

    def names(self) -> frozenset[str]:
        """이 plan이 다루는 proj 이름 집합 (훅 predicate의 빠른 1차 거름)."""
        return frozenset(name for _, name in self.projs)

    def coordinates(self) -> frozenset[Tuple[int, str]]:
        """plan이 기대하는 (layer, proj) 전부.

        로딩이 끝난 뒤 실제로 걸린 것과 대조해 **하나라도 안 걸렸으면 즉사**시키는 데
        쓴다 — 오타 하나나 layer마다 다른 projection 집합(Qwen3.8은 full_attention이
        16개 layer에만 있다)이 조용히 "오프로드 안 함"이 되는 것을 막는 유일한 게이트다.
        """
        return frozenset(self.projs)


# ---------------------------------------------------------------------------
# 파싱
# ---------------------------------------------------------------------------


def _parse_bands(obj: dict, where: str) -> Tuple[BandSpec, ...]:
    try:
        return tuple(BandSpec(int(s), int(e), Tier(t)) for s, e, t in obj["bands"])
    except (KeyError, TypeError, ValueError) as err:
        raise PlanError(f"{where}: malformed bands: {err}") from err


def _parse_shards(obj: dict, where: str) -> Tuple[NumaShard, ...]:
    try:
        return tuple(
            NumaShard(int(node), int(a), int(b))
            for node, a, b in obj.get("cold_shards", [])
        )
    except (TypeError, ValueError) as err:
        raise PlanError(f"{where}: malformed cold_shards: {err}") from err


def _parse_budget(obj: dict, where: str):
    """(p, lambda) — 쌍으로만 유효하다. 한쪽만 있으면 즉사."""
    if "p" not in obj and "lambda" not in obj:
        return None, None
    try:
        return float(obj["p"]), float(obj["lambda"])
    except (KeyError, TypeError, ValueError) as err:
        raise PlanError(f"{where}: p and lambda must both be present: {err}") from err


def _parse_sparse_fields(obj: dict, where: str):
    """(calib, sparse, p, lambda)."""
    p, lam = _parse_budget(obj, where)
    calib = obj.get("calib")
    if calib is not None and not isinstance(calib, str):
        raise PlanError(f"{where}: 'calib' must be a string table key, got {calib!r}")
    sparse = obj.get("sparse", True)
    if not isinstance(sparse, bool):
        raise PlanError(f"{where}: 'sparse' must be a boolean, got {sparse!r}")
    return calib, sparse, p, lam


def _parse_parts(obj: dict, n_total: int, where: str) -> Tuple[ProjPart, ...]:
    """`parts`가 있으면 N축 분할, 없으면 조각 하나.

    **리스트**인 이유는 순서와 크기가 둘 다 구조로 드러나야 하기 때문이다.
    dict + 균등분할로는 `qkv_proj`의 `[12288, 1024, 1024]`(q에 게이트가 실려 2배)를
    표현할 수 없고, JSON dict 순서에 정확성을 걸면 조용히 뒤바뀐다 — 순서가 틀리면
    gate 자리에 up이, q 자리에 k가 들어가고 절단은 성공한다.
    """
    raw = obj.get("parts")
    if raw is None:
        calib, sparse, p, lam = _parse_sparse_fields(obj, where)
        return (
            ProjPart(
                name=None, n_start=0, n_end=n_total,
                bands=_parse_bands(obj, where), cold_shards=_parse_shards(obj, where),
                calib=calib, sparse=sparse, sparsity_p=p, sparsity_lambda=lam,
            ),
        )
    if not isinstance(raw, list) or len(raw) < 2:
        raise PlanError(f"{where}: 'parts' must be a list of 2 or more objects")
    parts, cursor, seen = [], 0, set()
    for i, sub in enumerate(raw):
        sub_where = f"{where}.parts[{i}]"
        try:
            nm, n = str(sub["name"]), int(sub["n"])
        except (KeyError, TypeError, ValueError) as err:
            raise PlanError(f"{sub_where}: each part needs 'name' and 'n': {err}") from err
        if nm in seen:
            raise PlanError(f"{sub_where}: duplicate part name {nm!r}")
        seen.add(nm)
        if n <= 0:
            raise PlanError(f"{sub_where}: non-positive n={n}")
        calib, sparse, p, lam = _parse_sparse_fields(sub, sub_where)
        parts.append(ProjPart(
            name=nm, n_start=cursor, n_end=cursor + n,
            bands=_parse_bands(sub, sub_where), cold_shards=_parse_shards(sub, sub_where),
            calib=calib, sparse=sparse, sparsity_p=p, sparsity_lambda=lam,
        ))
        cursor += n
    if cursor != n_total:
        raise PlanError(
            f"{where}: parts sum to {cursor} but n={n_total} — 분할 경계가 어긋나면 "
            f"조각이 남의 행을 가져간다"
        )
    return tuple(parts)


def _parse_sparsity(obj: Optional[dict]) -> Optional[SparsitySpec]:
    """model-global sparsity 블록. 없으면 None (마스킹 없음)."""
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


def parse_plan(source: Union[str, Path, dict]) -> LinearPlan:
    """JSON 경로 또는 이미 읽은 dict → LinearPlan (완전 명시형).

    파싱은 구조만 본다. 기하 불변식(커버리지·정렬·shard)은 `validate_static`이
    별도로 본다 — MoE plan과 같은 분업이다.
    """
    if isinstance(source, (str, Path)):
        with open(source) as f:
            raw = json.load(f)
    else:
        raw = source

    try:
        version = int(raw["schema_version"])
        model_id = str(raw["model_id"])
        d = raw["dims"]
        dims = ModelDims(num_layers=int(d["num_layers"]), dtype=str(d["dtype"]))
        kern = raw["kernels"]
        kernels = KernelSpec(
            gpu_warm=str(kern["gpu_warm"]), cpu_cold=str(kern["cpu_cold"])
        )
        proj_defaults = raw["projs"]
    except (KeyError, TypeError, ValueError) as err:
        raise PlanError(f"malformed plan header: {err}") from err

    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise PlanError(
            f"unsupported schema_version {version} "
            f"(supported: {SUPPORTED_SCHEMA_VERSIONS})"
        )
    if not isinstance(proj_defaults, dict) or not proj_defaults:
        raise PlanError("'projs' must be a non-empty object")

    sparsity = _parse_sparsity(raw.get("sparsity"))

    # 이름별 (k, n, kernels) + 기본 조각들
    geom: dict[str, tuple[int, int, KernelSpec]] = {}
    base: dict[str, Tuple[ProjPart, ...]] = {}
    for name, obj in proj_defaults.items():
        where = f"projs.{name}"
        if not name:
            raise PlanError("empty proj name")
        try:
            k_, n_ = int(obj["k"]), int(obj["n"])
        except (KeyError, TypeError, ValueError) as err:
            raise PlanError(f"{where}: missing/malformed k or n: {err}") from err
        ov = obj.get("kernels")
        if ov is None:
            kspec = kernels
        else:
            try:
                kspec = KernelSpec(
                    gpu_warm=str(ov.get("gpu_warm", kernels.gpu_warm)),
                    cpu_cold=str(ov.get("cpu_cold", kernels.cpu_cold)),
                )
            except (TypeError, ValueError) as err:
                raise PlanError(f"{where}: malformed kernels override: {err}") from err
        geom[name] = (k_, n_, kspec)
        base[name] = _parse_parts(obj, n_, where)

    # 전 layer로 전개
    projs: dict[Tuple[int, str], ProjPlan] = {}
    for layer in range(dims.num_layers):
        for name, (k, n, kspec) in geom.items():
            projs[(layer, name)] = ProjPlan(
                name=name, k=k, n=n, kernels=kspec, parts=base[name]
            )

    # overrides가 개별 (layer, proj)를 대체 (k/n/kernels/분할 구조는 상속)
    for entry in raw.get("overrides", []):
        try:
            layer = int(entry["layer"])
        except (KeyError, TypeError, ValueError) as err:
            raise PlanError(f"malformed override entry: {err}") from err
        if not 0 <= layer < dims.num_layers:
            raise PlanError(
                f"override layer {layer} out of range [0, {dims.num_layers})"
            )
        for name, obj in entry.items():
            if name == "layer":
                continue
            if name not in geom:
                raise PlanError(
                    f"override layer {layer}: proj '{name}' not declared in 'projs' "
                    f"(known: {sorted(geom)})"
                )
            where = f"overrides[layer={layer}].{name}"
            k, n, kspec = geom[name]
            parts = _parse_parts(obj, n, where)
            if len(parts) != len(base[name]):
                raise PlanError(
                    f"{where}: override changes the split ({len(base[name])} parts in "
                    f"'projs', {len(parts)} here) — 분할 구조는 모델 기하라 layer마다 같다"
                )
            projs[(layer, name)] = ProjPlan(
                name=name, k=k, n=n, kernels=kspec, parts=parts
            )

    return LinearPlan(
        schema_version=version,
        model_id=model_id,
        dims=dims,
        kernels=kernels,
        projs=projs,
        sparsity=sparsity,
    )


# ---------------------------------------------------------------------------
# 정적 검증
# ---------------------------------------------------------------------------


def _validate_bands(part: ProjPart, k: int, where: str) -> None:
    """[0, k) 완전 커버 + 무중첩 + 페어 정렬. MoE `_validate_bands`와 같은 규칙."""
    if not part.bands:
        raise PlanError(f"{where}: no bands")
    cursor = 0
    for b in part.bands:
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
    if cursor != k:
        raise PlanError(f"{where}: bands cover [0, {cursor}) but k={k}")


def _validate_shards(part: ProjPart, where: str) -> None:
    """cold N-shard는 COLD 밴드가 있을 때만, 있으면 이 조각의 [0, n) 완전 커버.

    좌표가 **조각 로컬**이라는 데 주의 — gate가 [0, I), up도 [0, I)다. 전체 weight의
    행 번호가 아니다. kt 인스턴스가 조각마다 따로이므로 그쪽 어휘와 맞다.
    """
    if not part.has_tier(Tier.COLD):
        if part.cold_shards:
            raise PlanError(f"{where}: cold_shards present but no COLD band")
        return
    if not part.cold_shards:
        raise PlanError(f"{where}: COLD band exists but cold_shards is empty")
    cursor = 0
    for s in part.cold_shards:
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
    if cursor != part.n:
        raise PlanError(f"{where}: shards cover [0, {cursor}) but n={part.n}")


def _validate_sparsity(part: ProjPart, enabled: bool, where: str) -> None:
    """sparsity 필드의 정합. 빠뜨림과 의도적 제외를 **구분**하는 것이 목적이다.

    plan.sparsity가 없으면: 조각에 calib/p/lambda가 있으면 안 된다 (값이 조용히 버려진다).
    plan.sparsity가 있으면 둘 중 하나여야 한다 —
      · `"sparse": false`  → calib/p/lambda 전부 없음 (의도적 제외)
      · 그 외             → calib/p/lambda 전부 있음 (마스킹)
    빠뜨리면 죽는다. 안 죽으면 마스킹이 조용히 사라지고 성능만 달라진다.
    """
    given = [n for n, v in (("calib", part.calib), ("p", part.sparsity_p),
                            ("lambda", part.sparsity_lambda)) if v is not None]
    if not enabled:
        if given or not part.sparse:
            raise PlanError(
                f"{where}: plan has no sparsity block but this part declares "
                f"{given or ['sparse: false']} — 값이 조용히 버려진다"
            )
        return
    if not part.sparse:
        if given:
            raise PlanError(
                f"{where}: 'sparse': false but also declares {given} — "
                f"제외하려면 calib/p/lambda를 전부 빼야 한다"
            )
        return
    missing = [n for n in ("calib", "p", "lambda") if n not in given]
    if missing:
        raise PlanError(
            f"{where}: sparsity is on but {missing} missing — 마스킹에서 빼려면 "
            f'"sparse": false 를 명시해야 한다 (빠뜨림과 구분하기 위해)'
        )
    if not 0.0 <= part.sparsity_p <= 1.0:
        raise PlanError(f"{where}: p={part.sparsity_p} out of [0, 1]")


def validate_static(plan: LinearPlan) -> None:
    """모델 없이 확인 가능한 전부. startup에서 1회, 실패는 즉사.

    확인하지 **않는** 것: proj 이름이 실제 모델에 존재하는지, k/n이 실제 layer와
    맞는지. 전자는 로딩이 끝난 뒤 `coordinates()` 대조로 잡고(오타 게이트), 후자는
    `check_dims`가 layer마다 로드 타임에 본다.
    """
    resolve_gpu_kernel(plan.kernels.gpu_warm)
    resolve_cpu_kernel(plan.kernels.cpu_cold)

    if plan.dims.num_layers <= 0:
        raise PlanError(f"num_layers must be positive, got {plan.dims.num_layers}")

    spec = plan.sparsity
    if spec is not None and spec.score not in KNOWN_SPARSITY_SCORES:
        raise PlanError(
            f"unknown sparsity score '{spec.score}' "
            f"(known: {sorted(KNOWN_SPARSITY_SCORES)})"
        )

    for (layer, name), pp in plan.projs.items():
        where = f"layer {layer} proj '{name}'"
        if pp.k <= 0 or pp.n <= 0:
            raise PlanError(f"{where}: non-positive k={pp.k} or n={pp.n}")
        resolve_gpu_kernel(pp.kernels.gpu_warm)
        resolve_cpu_kernel(pp.kernels.cpu_cold)
        if sum(p.n for p in pp.parts) != pp.n:
            raise PlanError(
                f"{where}: parts cover {sum(p.n for p in pp.parts)} rows but n={pp.n}"
            )
        for part in pp.parts:
            sub = where if part.name is None else f"{where} [{part.name}]"
            _validate_bands(part, pp.k, sub)
            _validate_shards(part, sub)
            _validate_sparsity(part, spec is not None, sub)


def check_dims(pp: ProjPlan, k: int, n: int, where: str) -> None:
    """plan이 적은 기하와 실제 layer의 치수를 대조한다 (로드 타임, layer마다).

    다른 모델의 plan을 먹이면 여기서 죽는다. 안 죽으면 K-슬라이스가 엉뚱한 행을
    집어 **결과가 조용히 틀린다** — 성능만 달라지는 오류와 달리 어떤 게이트도
    잡지 못하는 종류다.
    """
    if (pp.k, pp.n) != (k, n):
        raise PlanError(
            f"{where}: plan says (k={pp.k}, n={pp.n}) but layer is (k={k}, n={n}) — "
            f"plan이 다른 모델/설정에 적용되고 있다"
        )


def check_partition(pp: ProjPlan, output_partition_sizes: Sequence[int], where: str) -> None:
    """plan의 조각 경계가 실제 layer의 분할 경계와 **어긋나지 않는가**.

    요구는 "일치"가 아니라 **포함**이다: plan의 조각 경계가 layer 분할 경계의 부분집합
    이어야 한다. 두 경우를 다 받아야 하기 때문이다 —

      · `mlp.gate_up_proj` layer `[I, I]`, plan `[gate I, up I]` → 경계가 같다.
        sparsity가 gate/up을 따로 캘리브하므로 쪼개야 한다.
      · `linear_attn.in_proj_qkvz` layer `[2048, 2048, 6144, 6144]`, plan `[통짜 16384]`
        → calib이 이 projection을 안 덮어 마스킹을 안 하니 쪼갤 이유가 없다. 통짜가
        오히려 낫다 (스토어 하나, GEMV 하나).

    막아야 하는 것은 **경계가 layer 분할을 가로지르는 경우**다. plan이 `[8192, 8192]`
    라고 하면 두 번째 조각이 v의 뒷부분과 z의 앞부분을 섞는데, 절단은 성공하고 이름만
    거짓이 된다 — 그 이름으로 calib 테이블을 고르므로 조용히 다른 채널을 마스킹한다.
    `[gate, up]`이 뒤집힌 경우도 같은 검사에 걸린다 (경계는 맞지만 크기가 다르면).
    """
    sizes = tuple(int(s) for s in output_partition_sizes)
    total = sum(sizes)
    got = tuple(p.n for p in pp.parts)
    if sum(got) != total:
        raise PlanError(
            f"{where}: plan parts sum to {sum(got)} but the layer's "
            f"output_partition_sizes sum to {total} {sizes}"
        )
    # 경계 집합 (양 끝 제외)
    def bounds(xs):
        out, cur = set(), 0
        for x in xs[:-1]:
            cur += x
            out.add(cur)
        return out

    plan_b, layer_b = bounds(got), bounds(sizes)
    stray = sorted(plan_b - layer_b)
    if stray:
        raise PlanError(
            f"{where}: plan splits n into {got} but the layer's "
            f"output_partition_sizes is {sizes} — 조각 경계 {stray} 가 layer 분할을 "
            f"가로지른다. 조각이 남의 행을 섞으면 그 이름으로 고른 calib 테이블이 "
            f"엉뚱한 채널을 마스킹한다"
        )

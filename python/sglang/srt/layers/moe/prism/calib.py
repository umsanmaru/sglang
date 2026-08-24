"""Prism sparsity calib 자산 어댑터 — 논리 테이블 ↔ 자산 키의 유일한 번역 지점.

Plan(plan.py)은 "논리 테이블명 → shape"만 알고 자산 포맷의 어휘를 모른다
(순수 stdlib 유지). 이 모듈이 accuracy-eval의 `gatedyn_calib.pt`를 열어
논리명으로 노출하고, K축 밴드 슬라이싱까지 맡는다. 자산 생성기는 이
코드베이스 밖이다 (accuracy-eval `calib/*/gatedyn_calib.pt`).

k2wl2 점수 (계약 ①):

    imp_j  = sqrt(a[2j]·x0² + a[2j+1]·x1² + 2·c[j]·x0·x1),   a = wn²
    thr    = thr_table[expert, round(s / grid)]
    s      = clip(p − lam·(g_e − ḡ), 0, pmax)      # renorm_it회 재정규화
    keep_j = imp_j >= thr                          # 페어의 두 채널이 함께 살거나 죽는다

`wn`/`pair_dot`은 K(contraction) 축이라 weight와 **같은 밴드 절단**을 받는다 —
gate/up은 hidden 축, down은 intermediate 축. `thr` 곡선은 밴드와 무관한
per-(layer, expert) 값이라 절단하지 않는다.

threshold는 full-K 분포로 캘리브된 것을 그대로 쓴다 (계약 ① — 2026-08-24
결정). 마스크는 full-K 기준과 동일하지만 **밴드별 nnz 비율은 균일하지 않다** —
cold 밴드의 nnz는 실측 대상이다.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch

from sglang.srt.layers.moe.prism.plan import (
    PAIR_GROUP,
    CalibRef,
    ModelDims,
    PlanError,
    Proj,
    SparsitySpec,
)

# 논리명 → 자산 키. k2wl2 계열만 (계약 ①: score는 k2wl2로 고정).
# `*2l` = pairimp(교차항 포함) 곡선, `wn_*` = 열 노름, `c*` = 인접열 내적.
_ASSET_KEYS: dict[str, str] = {
    "thr_gate": "tg2l",
    "thr_up": "tu2l",
    "thr_down": "td2l",
    "wn_gate": "wn_g",
    "wn_up": "wn_u",
    "wn_down": "wn_d",
    "pair_dot_gate": "cg",
    "pair_dot_up": "cu",
    "pair_dot_down": "cd",
}

_SUPPORTED_SCORES = ("k2wl2",)


@dataclass(frozen=True)
class CalibBand:
    """한 (proj, 밴드)의 점수 재료 — K축이 밴드로 잘린 것.

    wn: fp32 [E, k_rows] / pair_dot: fp32 [E, k_rows // PAIR_GROUP].
    wn을 제곱해 두지 않는 이유: 자산의 어휘를 그대로 유지해 슬라이스의 이름이
    거짓말하지 않게 한다. a = wn²의 사전계산은 커널을 쓰는 쪽의 결정이다.
    """

    wn: torch.Tensor
    pair_dot: torch.Tensor

    @property
    def k_rows(self) -> int:
        return self.wn.shape[1]

    @property
    def wn_sq(self) -> torch.Tensor:
        """a = wn² — 점수 식이 실제로 쓰는 형태.

        GPU(sparsity.py)와 CPU(kt) 두 소비자가 같은 정의를 쓰도록 여기 둔다.
        한쪽이 wn을, 다른 쪽이 wn²을 받으면 마스크가 조용히 갈린다.
        """
        return (self.wn * self.wn).contiguous()


class CalibTables:
    """calib 자산의 소유자. per-layer 슬라이스를 뜨는 것 외의 책임은 없다."""

    def __init__(self, tables: Mapping[str, torch.Tensor]):
        missing = sorted(set(_ASSET_KEYS) - set(tables))
        if missing:
            raise PlanError(f"calib tables missing: {missing}")
        self._t = {name: tables[name] for name in _ASSET_KEYS}

    # ── 로딩 ─────────────────────────────────────────────────────────────
    @classmethod
    def load(
        cls, spec: SparsitySpec, *, verify_digest: bool = True
    ) -> "CalibTables":
        """Plan의 SparsitySpec이 가리키는 자산을 연다.

        sha256 대조가 "경로 + 해시" 계약의 무결성 절반이다 — 자산을 재생성해
        내용이 바뀌었는데 Plan이 그대로면 조용히 다른 threshold를 쓰게 된다.
        """
        if spec.score not in _SUPPORTED_SCORES:
            raise NotImplementedError(
                f"calib adapter supports {_SUPPORTED_SCORES}, got '{spec.score}'"
            )
        path = Path(spec.calib.path)
        try:
            raw = path.read_bytes()
        except OSError as err:
            raise PlanError(f"cannot read calib asset {path}: {err}") from err
        if verify_digest:
            digest = hashlib.sha256(raw).hexdigest()
            if digest != spec.calib.sha256.lower():
                raise PlanError(
                    f"calib asset digest mismatch for {path}: "
                    f"plan says {spec.calib.sha256}, file is {digest} — "
                    f"자산이 재생성되었거나 Plan이 다른 자산을 가리킨다"
                )
        try:
            blob = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as err:  # torch가 던지는 예외 종류가 버전마다 다르다
            raise PlanError(f"cannot load calib asset {path}: {err}") from err
        tables = {}
        for name, key in _ASSET_KEYS.items():
            if key not in blob:
                raise PlanError(f"calib asset {path} has no '{key}' (for {name})")
            t = blob[key]
            if not isinstance(t, torch.Tensor) or t.dim() != 3:
                raise PlanError(
                    f"calib asset {path}: '{key}' must be a 3-D tensor, "
                    f"got {type(t).__name__}"
                )
            tables[name] = t.detach().to(torch.float32).contiguous()
        return cls(tables)

    # ── validate_static의 calib_probe ────────────────────────────────────
    def shapes(self) -> dict[str, tuple[int, ...]]:
        """논리명 → shape. plan.SparsitySpec.expected_calib_shapes와 대조된다."""
        return {name: tuple(t.shape) for name, t in self._t.items()}

    def probe(self):
        """validate_static(calib_probe=...)에 넣을 콜러블.

        이미 열린 자산의 shape를 돌려주므로 CalibRef 인자는 쓰지 않는다 —
        자산을 두 번 읽지 않기 위한 형태다.
        """
        return lambda _ref: self.shapes()

    # ── 조회 ─────────────────────────────────────────────────────────────
    def thr(self, layer_idx: int, proj: Proj) -> torch.Tensor:
        """[E, ng] fp32 threshold 곡선. 밴드와 무관하므로 절단하지 않는다."""
        return self._t[f"thr_{proj.value}"][layer_idx]

    def slice_band(
        self, layer_idx: int, proj: Proj, start: int, end: int, where: str
    ) -> CalibBand:
        """K축 밴드 [start, end)의 점수 재료를 뜬다.

        페어 경계 정렬은 계약 ①의 핵심이다 — 밴드가 페어를 쪼개면 두 티어가
        같은 페어의 반쪽씩 갖게 되어 어느 쪽도 imp_j를 계산할 수 없다.
        validate_static이 밴드 정렬을 이미 보증하므로 여기 도달은 결함이다.
        """
        if start % PAIR_GROUP or end % PAIR_GROUP:
            raise AssertionError(
                f"{where}: band [{start}, {end}) splits a masking pair "
                f"(PAIR_GROUP={PAIR_GROUP})"
            )
        wn = self._t[f"wn_{proj.value}"][layer_idx][:, start:end]
        pair_dot = self._t[f"pair_dot_{proj.value}"][layer_idx][
            :, start // PAIR_GROUP : end // PAIR_GROUP
        ]
        return CalibBand(wn=wn.contiguous(), pair_dot=pair_dot.contiguous())

    def check_dims(self, dims: ModelDims, spec: SparsitySpec) -> None:
        """expected_calib_shapes와의 대조를 이 객체만으로 수행 (편의)."""
        expected = spec.expected_calib_shapes(dims)
        actual = self.shapes()
        for name in sorted(expected):
            if actual[name] != tuple(expected[name]):
                raise PlanError(
                    f"calib table '{name}' shape {actual[name]} != "
                    f"expected {tuple(expected[name])}"
                )

"""dense sparsity calib 자산 어댑터 — 논리 테이블 ↔ 자산 키의 유일한 번역 지점.

`moe/prism/calib.py`의 dense 대응물. Plan은 "어느 테이블 키를 쓰는가"(`ProjPart.calib`)
만 말하고 자산 포맷의 어휘를 모른다 (순수 stdlib 유지). 이 모듈이 자산을 열어 그 키로
노출하고 K축 gather까지 맡는다. 자산 생성기는 이 코드베이스 밖이다.

k2wl2 점수 (MoE와 동일)::

    imp_j  = sqrt(a[2j]·x0² + a[2j+1]·x1² + 2·c[j]·x0·x1),   a = wn²
    thr    = thr_table[round(s / grid)]
    s      = clip(p − lam·(g_e − ḡ), 0, pmax)
    keep_j = imp_j >= thr                      # 페어의 두 채널이 함께 살거나 죽는다

**dense에서 달라지는 것 셋:**

  * **expert 축이 없다.** 자산 shape이 `[L, 1, K]`라 `[layer, 0]`으로 읽는다. E=1
    퇴화형이므로 MoE 어댑터가 그대로 읽기도 하지만(실측 확인), 여기서는 그 1을
    벗겨 dense의 사실만 노출한다 — kt/커널이 요구하는 `[1, ng]` 모양으로의 복원은
    소비자의 몫이다 (`LinearColdShard.real_rows`를 스칼라로 둔 것과 같은 선택).

  * **`s = clip(p)` 로 퇴화한다.** 라우터가 없어 `g_e − ḡ = 0`이고 `lam`이 죽는다
    (`moe_base.hpp:625`의 `slot_sparsity`에 k=1). 자산도 `lam0 = 0.0`으로 만들어졌다.
    **예산은 정적이고 마스크만 활성화 의존**이다.

  * **테이블 이름 축이 열려 있다.** MoE는 `g`/`u`/`d` 셋이면 됐지만 dense는
    `q`/`k`/`v`/`o`가 더 있고 모델마다 늘 수 있다. 그래서 키를 plan이 준다.

**전부 0인 테이블은 즉사시킨다.** 자산이 그 (층, projection)을 캘리브하지 않았으면
`wn`과 `thr`이 0으로 채워져 있는데, 그러면 `imp = 0 >= thr = 0`이라 **전부 살아남는다**
— 출력은 정확하고 점수 계산 비용만 치른 채 sparsity가 0이 된다. 정확도 테스트도,
계약 테스트도, 서버도 전부 통과하고 **벤치 결론만 틀린다**. 실제로 Qwen3.8-27B 자산의
`linear_attn` 층 48개가 정확히 그 상태다 (`wn_o`/`co`/`to2l` 전부 0). 층 단위로 봐야
한다 — `wn_o` 전체는 0이 아니고 그 48개 층만 0이다.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

import torch

from sglang.srt.layers.prism.geometry import PAIR_GROUP, PlanError
from sglang.srt.layers.prism.linear.plan import (
    SCORES_WITH_PAIR_DOT,
    SCORES_WITH_WN,
    LinearPlan,
    SparsitySpec,
)

# score → threshold 테이블 이름의 접미사.
#   k2wl2 : tg2l/tu2l/td2l/tq2l/…   (인접 페어의 실제 에너지, 교차항 포함) + `wn_*`/`c*`
#   k1    : tg/tu/td/tq/…           (per-k |x| 분위수). 가중치 통계를 쓰지 않는다.
#   k1w1  : tg/tu/td/tq/…           (per-k **|x|·wn** 분위수) — **곡선은 k1과 같은 표**이고
#           점수가 다르다. 2026-09-07 사용자 확인: dense 자산(`assets/{qwen38_27b,
#           muse_glimer,mistral_med_128B}.pt`)의 무접미사 곡선이 이 점수의 분위수다.
#           그래서 `wn_*`를 읽고(제곱해 a로), `c*`는 읽지 않는다.
# 어느 통계가 필요한지는 `SparsitySpec.needs_wn / needs_pair_dot`가 말한다.
_THR_SUFFIX = {"k2wl2": "2l", "k1": "", "k1w1": ""}


@dataclass(frozen=True)
class LinearCalibShard:
    """한 (조각, 티어)의 점수 재료 — weight 스토어와 **같은 순서**의 flat.

    `wn`: fp32 `[k_rows]` / `pair_dot`: fp32 `[k_rows // PAIR_GROUP]`.
    같은 인덱스로 모으는 것이 전부다 — 마스크 비트 ↔ packed 타일 대응이 유지되려면
    점수의 행 순서가 gather된 weight의 순서와 같아야 한다.
    """

    wn: torch.Tensor
    pair_dot: torch.Tensor

    @property
    def wn_sq(self) -> torch.Tensor:
        """a = wn² — 점수 식이 실제로 쓰는 형태.

        GPU와 CPU(kt) 두 소비자가 같은 정의를 쓰도록 정의점을 하나로 둔다. 한쪽이
        wn을, 다른 쪽이 wn²을 받으면 마스크가 조용히 갈린다.
        """
        return (self.wn * self.wn).contiguous()


class LinearCalibTables:
    """calib 자산의 소유자. per-(layer, key) 슬라이스를 뜨는 것 외의 책임은 없다."""

    def __init__(self, tables: Mapping[str, torch.Tensor], score: str, ng: int):
        if score not in _THR_SUFFIX:
            raise PlanError(
                f"unknown sparsity score '{score}' (known: {sorted(_THR_SUFFIX)})"
            )
        self._t = tables
        self._score = score
        self._suffix = _THR_SUFFIX[score]
        self._ng = int(ng)
        self._needs_wn = score in SCORES_WITH_WN
        self._needs_pair_dot = score in SCORES_WITH_PAIR_DOT
        self._weight_stats = self._needs_wn

    @property
    def score(self) -> str:
        return self._score

    @property
    def uses_weight_stats(self) -> bool:
        """가중치 통계를 읽는 score인가 (k2wl2·k1w1 참, k1 거짓)."""
        return self._weight_stats

    @property
    def needs_wn(self) -> bool:
        return self._needs_wn

    @property
    def needs_pair_dot(self) -> bool:
        """인접 열 내적까지 읽는가 — 페어 에너지 score만 (k1w1은 wn까지)."""
        return self._needs_pair_dot

    # ── 로딩 ─────────────────────────────────────────────────────────────
    @classmethod
    def load(cls, spec: SparsitySpec, *, verify_digest: bool = True
             ) -> "LinearCalibTables":
        """Plan의 SparsitySpec이 가리키는 자산을 연다.

        sha256 대조가 "경로 + 해시" 계약의 무결성 절반이다 — 자산을 재생성해 내용이
        바뀌었는데 Plan이 그대로면 조용히 다른 threshold를 쓰게 된다.
        """
        path = Path(spec.calib.path)
        if not path.is_file():
            raise PlanError(f"calib asset not found: {path}")
        if verify_digest:
            got = hashlib.sha256(path.read_bytes()).hexdigest()
            if got != spec.calib.sha256:
                raise PlanError(
                    f"calib asset digest mismatch for {path}: "
                    f"expected {spec.calib.sha256}, got {got}"
                )
        raw = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        tables = {k: v for k, v in raw.items() if torch.is_tensor(v)}
        return cls(tables, spec.score, spec.ng)

    # ── 이름 ─────────────────────────────────────────────────────────────
    def _names(self, key: str) -> tuple[str, str, str]:
        return f"wn_{key}", f"c{key}", f"t{key}{self._suffix}"

    def _table(self, name: str, where: str) -> torch.Tensor:
        t = self._t.get(name)
        if t is None:
            raise PlanError(
                f"{where}: calib asset has no table '{name}' "
                f"(available: {sorted(self._t)[:12]}…)"
            )
        if t.dim() != 3 or t.shape[1] != 1:
            raise PlanError(
                f"{where}: table '{name}' has shape {tuple(t.shape)}, expected "
                f"[L, 1, …] — dense 자산은 expert 축이 1이어야 한다"
            )
        return t

    # ── 검증 ─────────────────────────────────────────────────────────────
    def check(self, layer_idx: int, key: str, k: int, where: str) -> None:
        """이 (층, 키)가 실제로 캘리브됐는지 + 치수가 맞는지.

        **전부 0이면 즉사한다** — 모듈 docstring의 무증상 실패를 막는 유일한 게이트다.
        """
        wn_n, pd_n, thr_n = self._names(key)
        thr = self._table(thr_n, where)
        # 읽는 표만 검증한다: k1은 thr만, k1w1은 thr+wn, k2wl2는 thr+wn+pair_dot.
        # 안 읽는 표가 자산에 없거나 그 층이 0이어도 정상이다.
        checked = []
        wn = None
        if self._needs_wn:
            wn = self._table(wn_n, where)
            checked.append((wn_n, wn, k))
        if self._needs_pair_dot:
            checked.append((pd_n, self._table(pd_n, where), k // PAIR_GROUP))
        checked.append((thr_n, thr, self._ng))
        L = checked[0][1].shape[0]
        if not 0 <= layer_idx < L:
            raise PlanError(f"{where}: layer {layer_idx} out of calib range [0, {L})")
        for n, t, exp in checked:
            if t.shape[2] != exp:
                raise PlanError(
                    f"{where}: table '{n}' has K={t.shape[2]} but expected {exp} — "
                    f"자산이 다른 모델/설정의 것이다"
                )
        zero_gated = [(thr_n, thr)] + ([(wn_n, wn)] if wn is not None else [])
        for n, t in zero_gated:
            if not bool(t[layer_idx, 0].any()):
                raise PlanError(
                    f"{where}: calib table '{n}' is all zeros at layer {layer_idx} — "
                    f"이 자산은 그 층의 그 projection을 캘리브하지 않았다. plan에서 "
                    f'"sparse": false 로 빼거나 자산을 다시 만들어야 한다 '
                    f"(마스킹이 조용히 사라지고 성능만 달라진다)"
                )

    def check_plan(self, plan: LinearPlan, *, strict: bool = False) -> list:
        """plan이 마스킹하겠다고 한 모든 (layer, proj, 조각)을 대조한다 (startup 1회).

        **plan은 실행 트리가 어느 층에 어느 모듈을 실제로 만드는지 모른다.** 생성기는
        `self_attn.*`를 64층 전부에 선언하는데 Qwen3.8-27B에서 그 모듈은 full_attention
        16층에만 있고, calib이 나머지 48층을 0으로 둔 것은 **정확한 사실**이다. 그래서
        전수 대조로 죽이면 존재하지 않을 좌표가 실모델 기동을 죽인다 (2026-09-01에
        실제로 그랬다 — 42층 웨이트를 다 채운 뒤였다).

        진짜 게이트는 `weights._prepare_part`에 있다 — 그 모듈이 실재할 때만 도는
        자리이고, 거기서의 0 테이블은 무증상 실패다. 여기는 기본이 비치명이고,
        돌려주는 목록을 호출자가 로그로 내보내 "이 자산이 안 덮는 자리"를 남긴다.
        """
        bad = []
        for (layer, name), pp in plan.projs.items():
            for part in pp.parts:
                if not part.sparse or part.calib is None:
                    continue
                sub = f"layer {layer} proj '{name}'" + (
                    "" if part.name is None else f" [{part.name}]")
                try:
                    self.check(layer, part.calib, pp.k, sub)
                except PlanError as err:
                    if strict:
                        raise
                    bad.append(str(err).split(" — ")[0])
        return bad

    # ── 조회 ─────────────────────────────────────────────────────────────
    def thr(self, layer_idx: int, key: str) -> torch.Tensor:
        """`[ng]` fp32 threshold 곡선. 밴드와 무관하므로 절단하지 않는다.

        kt/커널이 요구하는 `[E, ng]` 모양으로의 복원은 소비자의 몫이다 (E=1 퇴화).
        """
        return self._table(f"t{key}{self._suffix}", "thr")[layer_idx, 0].contiguous()

    def gather(self, layer_idx: int, key: str, k_index: torch.Tensor,
               real_rows: Optional[int] = None, where: str = "") -> LinearCalibShard:
        """티어 인덱스로 점수 재료를 모은다 — weight와 **같은 순서**.

        `k_index`는 스토어가 쓴 그 인덱스여야 한다 (cold면 타일 경계까지 패딩된 것).
        `real_rows`가 주어지면 그 뒤는 패딩이라 0으로 남긴다 — weight도 0이라 수치
        기여가 없고, kt가 마스크 tail 비트를 끈다.

        score k1에서는 부를 것이 아니다 (즉사) — 마스크가 `|x_k| >= thr` 하나라
        점수 재료가 없고, 불렀다면 호출자가 score를 안 보고 있다는 뜻이다.
        """
        if not self._needs_wn:
            raise PlanError(
                f"{where}: score '{self._score}'는 가중치 통계를 쓰지 않는다 — "
                f"gather()를 부르지 말고 thr()만 쓴다"
            )
        wn_all = self._table(f"wn_{key}", where)[layer_idx, 0]   # [K]
        # pair_dot은 페어 에너지 score만 읽는다. k1w1은 wn까지이므로 자리만 0으로 둔다 —
        # 소비자(kt·GPU)가 그 포인터를 아예 안 보므로 값이 아니라 **모양**만 맞으면 된다.
        pd_all = (self._table(f"c{key}", where)[layer_idx, 0] if self._needs_pair_dot
                  else torch.zeros(wn_all.numel() // PAIR_GROUP, dtype=torch.float32))
        total = int(k_index.numel())
        if total % PAIR_GROUP:
            raise PlanError(
                f"{where}: {total} rows is not a multiple of PAIR_GROUP={PAIR_GROUP}"
            )
        wn = torch.zeros(total, dtype=torch.float32)
        pair_dot = torch.zeros(total // PAIR_GROUP, dtype=torch.float32)
        kr = total if real_rows is None else int(real_rows)
        if kr:
            rows = k_index[:kr].to(torch.int64)
            wn[:kr] = wn_all[rows]
            # 페어 id는 gather된 순서의 짝수 위치가 가리키는 원본 페어다 (페어
            # 무결성이 보장되므로 홀수 위치는 같은 페어의 반대쪽).
            if self._needs_pair_dot:
                pair_dot[: kr // PAIR_GROUP] = pd_all[rows[::PAIR_GROUP] // PAIR_GROUP]
        return LinearCalibShard(wn=wn.contiguous(), pair_dot=pair_dot.contiguous())

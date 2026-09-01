"""dense COLD 백엔드 — 슬롯 ↔ kt 어휘의 번역과 인스턴스 수명 (계약 ②).

**kt에 새 진입점을 요구하지 않는다.** 기존 `forward_gateup_partial` /
`forward_down_partial`의 두 축에 dense 의미를 준다 (계약 ②-0). 흉내가 아니라
동형이다 — `forward_down_partial(qlen=1, k=1)`의 경로에서 expert는 per-expert
버퍼의 인덱스로만 등장하고 라우팅도 gather-scatter도 합산도 없다:

    expert 축    ↔ 슬롯 신원 (layer, proj, part)   — 로드 타임 고정
    top_k 축     ↔ 한 호출에서 같이 계산하는 슬롯  — 지금은 항상 1
    gate/up 슬롯 ↔ K를 공유하고 N이 같은 **인접 두 part**
    down 슬롯    ↔ 나머지 part 하나

**인스턴스 = (진입점, K, N, 커널, shard) 형상 그룹 하나**이고 `expert_num`은 그
형상을 가진 unit의 수(전 layer 합)다. Qwen3.8-27B는 그룹 9개 × 2노드 = C++ 객체
18개다 — 슬롯당 하나로 만들면 352 × 2 = 704개가 되는데, 그건 퇴화 경로의 성질이
아니라 그렇게 만들었을 때의 성질이다 (TODO §3).

**더미 슬롯의 치수는 최소로 잡는다.** kt는 아직 세 proj를 전부 요구하는데
(`moe.hpp:513` `no weight source`), 비싼 것은 더미 weight가 아니라 그 proj의 C
버퍼 풀이다:

    gate_bc_pool = buffer_c(pool_count_, intermediate_size)   # = 4·m·n 바이트
    down_bc_pool = buffer_c(pool_count_, down_n())

그런데 `intermediate_size`(gate/up N의 총합)와 `down_n()`(down N의 노드 shard)은
둘 다 **우리가 주는 노드 테이블이 정한다**: `moe-tp.hpp`가 gateup 테이블 합을
`intermediate_size`에, down 테이블 합을 `hidden_size`에 맞추라 요구하고 노드
config의 `intermediate_size`는 곧 `node_gateup_n_rows[i]`다. 그래서 down 매핑에서는
더미 gate/up의 N을 노드당 `cold_n_align` 한 칸으로 깎아 풀을 무해하게 만든다.
벤치가 잰 "72 MB/인스턴스"는 더미가 실제 K를 물려받았기 때문이었다 (TODO §3.1c).

gateup 매핑에서는 더미가 down이고 그 N 총합이 `hidden_size`(= 실제 K)로 **고정**
이라 깎을 수 없다 — 그룹당 `4·pool_count_·K/nodes` 한 벌을 지불한다.

sparsity와 양자화 스토어(fp8/mxfp4)는 아직 배선하지 않는다 — 둘 다 즉사시킨다.
조용히 dense/bf16으로 돌면 벤치 결론만 틀린다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from sglang.srt.layers.prism.geometry import NumaShard, PlanError
from sglang.srt.layers.prism.kernels import cold_n_align, cold_pack_tile_rows
from sglang.srt.layers.prism.linear.plan import LinearPlan
from sglang.srt.layers.prism.linear.weights import LinearColdShard, PreparedLinear

# 한 호출에 싣는 unit 수. 1로 고정한다 — down 진입점은 act의 슬롯 stride가
# `n_total`이라 unit을 묶으려면 x를 그만큼 **복제**해야 하고(계약 ②-3), 그
# 복제가 prefill에서 M×K×2 B다. 묶음은 최적화이지 계약이 아니다.
TOP_K = 1

logger = logging.getLogger(__name__)

# 이 백엔드가 아는 cold 커널. 양자화 키는 배율 셋과 `cold_load_kwargs`가 필요한데
# dense formats에 아직 그 훅이 없다 (TODO). 조용히 빠지면 pack이 쓰레기를 읽는다.
_SUPPORTED_KERNELS = frozenset({"kt_amx_bf16", "kt_tile_k2_bf16"})


# ---------------------------------------------------------------------------
# 좌표
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Unit:
    """한 expert 슬롯이 담는 것. `colds`는 1개(down 진입점) 또는 2개(gateup)다.

    plan의 `cold_shards`를 같이 나르는 이유: `PreparedPart`는 스토어만 알고
    노드 분할은 plan의 것인데, 그룹 키가 그 테이블을 포함해야 한다 (계약 ②-2).
    """

    layer: int
    proj: str
    k: int                                   # contraction 축 full 길이
    n_start: int                             # out3d의 열 오프셋
    colds: Tuple[LinearColdShard, ...]
    shards: Tuple[Tuple[int, int, int], ...]  # (node, n_start, n_end)

    @property
    def entry(self) -> str:
        return "gateup" if len(self.colds) == 2 else "down"

    @property
    def n(self) -> int:
        return self.colds[0].n

    @property
    def label(self) -> str:
        return f"L{self.layer} {self.proj}@{self.n_start}"


@dataclass(frozen=True)
class GroupKey:
    """인스턴스 하나를 정하는 것 전부. 같은 키면 같은 C++ 인스턴스를 쓴다.

    `shards`가 키에 들어가는 이유: 노드 N shard 테이블은 config 스칼라라 그룹
    전체가 하나를 공유한다. plan 생성기가 shard를 `(n, nodes)`만의 함수로 내므로
    같은 N이면 같은 테이블이지만, 그것은 **생성기의 성질이지 스키마의 보장이
    아니다** — 키에 넣어 강제한다.
    """

    entry: str                                 # "gateup" | "down"
    k: int                                     # contraction 축 길이 (full)
    n: int                                     # part 하나의 출력 축
    kernel: str                                # cpu_cold 키
    shards: Tuple[Tuple[int, int, int], ...]

    @property
    def label(self) -> str:
        return f"{self.entry}(K={self.k},N={self.n},{self.kernel})"


@dataclass(frozen=True)
class ColdCall:
    """executor가 한 (layer, proj)에서 실행할 cold 호출 하나."""

    group: "ColdGroup"
    expert: int
    n_start: int      # out3d의 열 오프셋 (proj 좌표)
    n_cols: int       # 이 호출이 채우는 열 수 = N (down) 또는 2N (gateup)


# ---------------------------------------------------------------------------
# 그룹
# ---------------------------------------------------------------------------


@dataclass
class ColdGroup:
    """형상 그룹 하나 = kt 인스턴스 하나 (노드마다 sub-instance)."""

    key: GroupKey
    units: List[Unit] = field(default_factory=list)
    wrapper: object = None
    staging: object = None                       # LinearColdStaging (resources.py)
    _expert_of: Dict[int, int] = field(default_factory=dict)

    @property
    def num_experts(self) -> int:
        return len(self.units)

    @property
    def out_cols(self) -> int:
        """한 호출이 채우는 out 열 수. gateup은 gate|up이 이어진다."""
        return 2 * self.key.n if self.key.entry == "gateup" else self.key.n

    @property
    def x_width(self) -> int:
        """kt에 넘기는 입력 행 폭. 두 진입점 모두 **full-width**다 (계약 ②-8)."""
        return self.key.k

    def expert_of(self, unit: Unit) -> int:
        return self._expert_of[id(unit)]

    def freeze(self) -> None:
        """expert id 확정. 순서는 (layer, n_start) — 결정적이어야 로그와 자산이
        재기동을 넘어 대응된다."""
        self.units.sort(key=lambda u: (u.layer, u.proj, u.n_start))
        self._expert_of = {id(u): e for e, u in enumerate(self.units)}


# ---------------------------------------------------------------------------
# unit 쪼개기
# ---------------------------------------------------------------------------


def _shard_tuple(shards: Sequence[NumaShard]) -> Tuple[Tuple[int, int, int], ...]:
    return tuple(sorted((s.node, s.n_start, s.n_end) for s in shards))


def split_units(layer: int, name: str, prepared: PreparedLinear,
                shards_of_part: Sequence[Tuple[Tuple[int, int, int], ...]]) -> List[Unit]:
    """한 (layer, proj)의 cold part들을 unit으로 쪼갠다.

    **인접한** 두 part가 N과 cold shard가 같으면 gateup unit으로 묶는다. 인접을
    요구하는 이유는 출력 열이다: kt의 gateup out은 `[gate 열 | up 열]` 순서라 두
    part가 out3d에서도 이어져 있어야 H2D가 연속 복사 하나로 끝난다. 안 이어져
    있으면 묶어도 복사가 둘이라 얻는 것이 없다.

    `mlp.gate_up_proj`(gate|up)와 `self_attn.qkv_proj`의 (k|v)가 이 규칙에
    걸리고, q는 N이 달라 혼자 남는다.

    cold 행이 없는 part는 unit이 되지 않는다 — kt가 계산할 것이 없다.
    """
    parts = prepared.parts
    units: List[Unit] = []
    i = 0
    while i < len(parts):
        a = parts[i]
        if a.cold is None:
            i += 1
            continue
        b = parts[i + 1] if i + 1 < len(parts) else None
        pairable = (
            b is not None
            and b.cold is not None
            and a.n == b.n
            and a.n_end == b.n_start
            and shards_of_part[i] == shards_of_part[i + 1]
        )
        colds = (a.cold, b.cold) if pairable else (a.cold,)
        units.append(Unit(layer=layer, proj=name, k=prepared.k, n_start=a.n_start,
                          colds=colds, shards=shards_of_part[i]))
        i += 2 if pairable else 1
    return units


# ---------------------------------------------------------------------------
# 백엔드
# ---------------------------------------------------------------------------


class KtLinearColdBackend:
    """dense cold의 kt 구현. CPUInfer(스레드풀)와 그룹 인스턴스들의 소유자.

    수명이 두 단계인 것이 MoE 백엔드와 다른 점이다:

      register(...)  layer마다. unit을 모으기만 한다 — 그룹의 expert 수를 모르면
                     config를 만들 수 없고, plan만으로는 알 수 없다 (Qwen3.8은
                     `self_attn.qkv_proj`가 64층 중 16층에만 있다. 그래서
                     `method.check_coverage`가 좌표가 아니라 이름으로 센다).
      finalize()     첫 step 직전 1회. 그룹마다 config·wrapper를 만들고 pack한다.

    finalize를 미루는 대가는 그때까지 cold weight 전량이 host RAM에 떠 있다는
    것이다. 대안(로딩 중 즉시 pack)은 그룹 크기를 모르면 불가능하다.
    """

    def __init__(self, plan: LinearPlan, *, max_tokens: int, num_numa_nodes: int,
                 cpuinfer=None, cpuinfer_threads: int = 28):
        if plan.sparsity is not None:
            raise NotImplementedError(
                "dense cold: sparse plan은 아직 배선되지 않았다 (TODO §4). "
                "조용히 dense로 돌면 마스킹이 안 걸린 채 sparse 벤치 결론이 나온다"
            )
        self._plan = plan
        self._max_tokens = int(max_tokens)
        self._nodes = int(num_numa_nodes)
        self._threads = int(cpuinfer_threads)
        self._cpuinfer = cpuinfer
        self._groups: Dict[GroupKey, ColdGroup] = {}
        self._calls: Dict[Tuple[int, str], Tuple[ColdCall, ...]] = {}
        self._pending: List[Tuple[Unit, str]] = []
        self._final = False

    @property
    def cpuinfer(self):
        """지연 생성 — kt import를 finalize까지 미룬다 (kt 없는 환경에서도 plan
        모듈을 쓸 수 있어야 한다)."""
        if self._cpuinfer is None:
            from kt_kernel import kt_kernel_ext

            self._cpuinfer = kt_kernel_ext.CPUInfer(self._threads)
        return self._cpuinfer

    # ── 등록 (layer마다) ─────────────────────────────────────────────────
    def register(self, layer_idx: int, name: str, prepared: PreparedLinear) -> None:
        if self._final:
            raise RuntimeError(
                f"layer {layer_idx} '{name}': finalize() 이후에는 등록할 수 없다 — "
                f"그룹의 expert 수가 이미 config로 굳었다"
            )
        pp = self._plan.proj(layer_idx, name)
        kernel = pp.kernels.cpu_cold
        if kernel not in _SUPPORTED_KERNELS:
            raise NotImplementedError(
                f"layer {layer_idx} '{name}': dense cold는 아직 {sorted(_SUPPORTED_KERNELS)}"
                f"만 안다 (plan은 '{kernel}'). 양자화 키는 배율 셋과 formats의 "
                f"cold_load_kwargs 훅이 선행돼야 한다"
            )
        shards = tuple(_shard_tuple(p.cold_shards) for p in pp.parts)
        if len(shards) != len(prepared.parts):
            raise PlanError(
                f"layer {layer_idx} '{name}': plan parts {len(shards)} != prepared "
                f"{len(prepared.parts)}"
            )
        for unit in split_units(layer_idx, name, prepared, shards):
            self._pending.append((unit, kernel))

    def has_work(self) -> bool:
        return bool(self._pending) or bool(self._groups)

    # ── finalize (첫 step 직전 1회) ───────────────────────────────────────
    def finalize(self) -> None:
        if self._final:
            return
        self._final = True
        for unit, kernel in self._pending:
            key = GroupKey(entry=unit.entry, k=unit.k, n=unit.n, kernel=kernel,
                           shards=unit.shards)
            self._groups.setdefault(key, ColdGroup(key)).units.append(unit)
        self._pending = []
        for group in self._groups.values():
            group.freeze()
            self._check_group(group)
        self._build_calls()
        # 통계를 **로드 전에** 뽑는다 — `_load_group`이 pack 뒤에 `unit.colds`를
        # 비우므로(계약 ③) 나중에 읽으면 전부 0이 나온다.
        stats = {g.key: _row_stats(g) for g in self._groups.values()}
        for group in self._groups.values():
            self._load_group(group)
        self._log_summary(stats)

    def _log_summary(self, stats) -> None:
        """그룹 구성과 **패딩 낭비**를 로드 타임에 보이게 한다.

        유령 밴드는 게이트가 죽이지만 "타일을 조금 넘긴" 밴드는 정상이면서도
        낭비다 (33행 → k_pad 64, 절반이 0). 그건 planner가 고칠 일이라 죽이지 않고
        숫자로 보여준다 — 안 보이면 아무도 안 고친다.
        """
        for g in sorted(self._groups.values(), key=lambda g: -g.num_experts):
            real, pad = stats[g.key]
            waste = 100.0 * (pad - real) / pad if pad else 0.0
            logger.info(
                "[prism-linear] cold %s: E=%d out_cols=%d rows=%d(+%d 패딩 %.1f%%)",
                g.key.label, g.num_experts, g.out_cols, real, pad - real, waste,
            )

    def _check_group(self, group: ColdGroup) -> None:
        """계약 ②-4 중 **그룹 키가 보장하지 않는 것**만 검사한다.

        `GroupKey`가 `(entry, k, n, kernel, shards)` 전부를 담으므로 "그룹 안에서
        N/K/노드 테이블이 같은가"는 검사할 것이 아니라 **구성상 참**이다 — 다르면
        오류가 아니라 다른 그룹, 즉 다른 인스턴스가 된다. 그 설계가 계약 ②-4의
        2·3·4항을 집행하는 방식이고, 같은 것을 여기서 다시 `if`로 쓰면 절대 참이
        되지 않는 죽은 검사가 된다 (실제로 그렇게 썼다가 테스트가 잡았다).

        남는 것은 키에서 파생되지 않는 셋이다:

          · **스토어 shape** — `LinearColdShard`의 실제 치수가 plan과 맞는가.
            gateup unit의 두 번째 part는 키에 안 들어가므로 여기서만 걸린다.
          · **노드 테이블의 완전성·정렬** — plan에서 온 값이라 자유롭게 틀릴 수 있고,
            틀리면 kt가 `partial shard table …`로 죽거나 (정렬이면) 커널이 조용히
            어긋난다.
          · **더미 축의 실현 가능성** — `_config`가 깎아 넣는 최소 치수가 kt의
            `export_*_partial` 범위 검사를 통과하는가.
        """
        key = group.key
        where = f"cold group {key.label}"
        align = cold_n_align(key.kernel)
        tile = cold_pack_tile_rows(key.kernel)

        for unit in group.units:
            for c in unit.colds:
                if c.n != key.n:
                    raise PlanError(
                        f"{where}: {unit.label} store N={c.n} != {key.n} — "
                        f"weights.py가 plan과 다른 치수의 스토어를 만들었다"
                    )
                if c.k_pad % tile:
                    raise PlanError(
                        f"{where}: {unit.label} k_pad={c.k_pad} is not a multiple of "
                        f"tile {tile} for '{key.kernel}'"
                    )
                # **유령 밴드 게이트.** pack 타일 하나도 못 채우는 cold 밴드는
                # plan 생성기의 반올림이 만든 것이지 의도가 아니다 (2026-09-01 이전에
                # 실제로 2행짜리가 나왔고, 그때는 `executor.register()`의 cold 거부가
                # 잡았다 — cold를 배선하면서 그 게이트가 사라졌으므로 여기가 대신한다).
                #
                # 안 잡으면 **값은 맞고 느리기만 하다**: 그 밴드가 계산에 기여하는
                # 것은 real_rows/K인데, 대가로 x D2H · submit/sync 왕복 · [M, N] H2D ·
                # rejoin 커널을 통째로 지불한다. 2행이면 30행이 패딩이라 CPU가 하는
                # 일의 94%가 0을 곱하는 것이다. 어떤 정확도 테스트도 이걸 못 잡는다.
                if c.real_rows < tile:
                    raise PlanError(
                        f"{where}: {unit.label}의 cold 밴드가 {c.real_rows}행뿐이라 "
                        f"pack 타일({tile}행) 하나도 못 채운다 — plan 생성기의 반올림이 "
                        f"만든 유령 밴드일 가능성이 높다. 이 밴드는 결과를 바꾸지 않지만 "
                        f"submit/sync 왕복과 H2D, rejoin을 통째로 지불한다. 밴드를 "
                        f"키우거나 그 행들을 GPU 티어로 넘겨라"
                    )

        # 노드 테이블: 노드 수만큼 있고 [0, N)을 구멍 없이 덮는다 (②-4-4).
        nodes = sorted(s[0] for s in key.shards)
        if nodes != list(range(self._nodes)):
            raise PlanError(
                f"{where}: cold_shards nodes {nodes} != 0..{self._nodes - 1} "
                f"— kt가 `partial shard table size != tp_count`로 죽는다"
            )
        cur = 0
        for node, s, e in sorted(key.shards, key=lambda t: t[1]):
            if s != cur:
                raise PlanError(f"{where}: cold_shards에 구멍 [{cur}, {s})")
            # 커널이 요구하는 정렬. `plan.validate_static`이 이미 `COL_GROUP = 32`로
            # 거르므로 bf16 키에서는 여기가 안 걸린다 — 이 검사가 사는 자리는
            # tile mxfp4/fp8(256)처럼 **커널이 더 세게 조일 때**다. plan.py는 순수
            # stdlib이라 커널 키가 함의하는 정렬을 모른다.
            if (e - s) % align:
                raise PlanError(
                    f"{where}: node {node} shard rows {e - s} is not a multiple of "
                    f"{align} (커널 '{key.kernel}'의 요구)"
                )
            cur = e
        if cur != key.n:
            raise PlanError(f"{where}: cold_shards가 {cur}에서 끝난다 (N={key.n})")

        # 더미 축이 최소 치수로 들어갈 수 있는가 (`_config` 참조).
        if tile > key.n:
            raise PlanError(f"{where}: N={key.n} < 더미 슬롯 타일 {tile}")
        if key.entry == "down":
            if align * self._nodes > key.k:
                raise PlanError(
                    f"{where}: K={key.k} < 더미 gate/up 최소 폭 {align * self._nodes} "
                    f"— export_gateup의 `n_off + inter <= n_total` 검사에 걸린다"
                )
        elif key.k % (align * self._nodes):
            raise PlanError(
                f"{where}: K={key.k}를 더미 down의 노드 테이블로 나눌 수 없다 "
                f"({align}의 배수 × {self._nodes}노드가 아니다)"
            )

    def _build_calls(self) -> None:
        by_proj: Dict[Tuple[int, str], List[ColdCall]] = {}
        for group in self._groups.values():
            for unit in group.units:
                by_proj.setdefault((unit.layer, unit.proj), []).append(
                    ColdCall(group=group, expert=group.expert_of(unit),
                             n_start=unit.n_start, n_cols=group.out_cols)
                )
        self._calls = {k: tuple(sorted(v, key=lambda c: c.n_start))
                       for k, v in by_proj.items()}

    def calls(self, layer_idx: int, name: str) -> Tuple[ColdCall, ...]:
        """이 (layer, proj)의 cold 호출들. 없으면 빈 튜플."""
        return self._calls.get((layer_idx, name), ())

    def groups(self) -> Tuple[ColdGroup, ...]:
        return tuple(self._groups.values())

    # ── Plan → kt config (유일한 번역 지점) ──────────────────────────────
    def _config(self, group: ColdGroup):
        from kt_kernel import kt_kernel_ext as ext

        key = group.key
        E = group.num_experts
        align = cold_n_align(key.kernel)
        tile = cold_pack_tile_rows(key.kernel)
        shard_off = [0] * self._nodes
        shard_rows = [0] * self._nodes
        for node, s, e in key.shards:
            shard_off[node], shard_rows[node] = s, e - s

        if key.entry == "gateup":
            # 실물 = gate/up. K축은 hidden_size, 출력축은 intermediate_size(= n_total).
            hidden, inter, n_total = key.k, key.n, key.n
            gu_off, gu_rows = shard_off, shard_rows
            # 더미 down: 노드 테이블 합이 **hidden_size로 고정**이라 폭을 못 줄인다
            # (moe-tp.hpp check_table). K축만 타일 하나로 깎는다.
            step = key.k // self._nodes
            dn_off = [i * step for i in range(self._nodes)]
            dn_rows = [step] * self._nodes
        else:
            # 실물 = down. K축은 n_total, 출력축은 hidden_size.
            hidden, n_total = key.n, key.k
            # 더미 gate/up: 테이블 합 = intermediate_size이고 그 값을 우리가 정한다.
            # 노드당 align 한 칸이면 gate_bc/up_bc 풀이 무해해진다.
            inter = align * self._nodes
            gu_off = [i * align for i in range(self._nodes)]
            gu_rows = [align] * self._nodes
            dn_off, dn_rows = shard_off, shard_rows

        # 더미의 하한 셋 (2026-09-01 실측 — 각각을 실제로 눌러봤다):
        #
        #   행 0 (슬롯 소멸)  → `no weight source`. 0원소 텐서의 `data_ptr()`가 0이라
        #                       `moe.hpp:465`의 `config.gate_proj != nullptr` 분기가
        #                       빠지고 throw로 떨어진다. **더미는 존재해야 한다.**
        #   행 2 (타일 미만)  → `per-expert rows must be a multiple of K_STEP`.
        #   N 2 (정렬 미만)   → **SEGFAULT.** 이건 예외가 아니라 조용한 죽음이라
        #                       kt가 잡아주지 않는다 — 그래서 여기서 검사한다.
        #
        # 즉 더미를 이보다 줄이는 길은 kt 쪽 "선택적 proj 슬롯"뿐이고, 그 값어치는
        # 실 형상에서 0.14 GB다 (TODO §3.3).
        dummy_rows, dummy_n = tile, (dn_rows if key.entry == "gateup" else gu_rows)
        if dummy_rows <= 0 or min(dummy_n) < align or any(r % align for r in dummy_n):
            raise PlanError(
                f"{key.label}: 더미 슬롯이 하한 아래다 (rows={dummy_rows}, "
                f"node N={dummy_n}, align={align}) — kt는 이 경우 예외가 아니라 "
                f"segfault로 죽는다"
            )

        cfg = ext.moe.MOEConfig(E, TOP_K, hidden, inter, 0)
        cfg.max_len = self._max_tokens
        cfg.layer_idx = 0        # kt에서 로그 문자열 외에 쓰이지 않는다
        cfg.partial.enabled = True
        cfg.partial.n_total = n_total
        cfg.partial.node_gateup_n_offset, cfg.partial.node_gateup_n_rows = gu_off, gu_rows
        cfg.partial.node_down_n_offset, cfg.partial.node_down_n_rows = dn_off, dn_rows

        # KIndex 셋. 더미 축 길이는 두 매핑 모두 key.n이다:
        #   gateup — 더미는 down이고 `validate_kindex(down, n_total = key.n)`
        #   down   — 더미는 gate/up이고 `validate_kindex(gate, hidden_size = key.n)`
        real = ("gate", "up") if key.entry == "gateup" else ("down",)
        for slot in ("gate", "up", "down"):
            dst = getattr(cfg.partial, slot)
            if slot in real:
                _set_kindex(dst, [u.colds[real.index(slot)] for u in group.units])
            else:
                _set_dummy_kindex(dst, E, tile, key.n)
        cfg.pool = self.cpuinfer.backend_
        return cfg

    # ── 로딩 ─────────────────────────────────────────────────────────────
    def _load_group(self, group: ColdGroup) -> None:
        from kt_kernel.experts_partial import PartialMoEWrapper

        key = group.key
        cfg = self._config(group)
        tile = cold_pack_tile_rows(key.kernel)
        E = group.num_experts

        if key.entry == "gateup":
            gate = _concat_cold([u.colds[0] for u in group.units])
            up = _concat_cold([u.colds[1] for u in group.units])
            down = _dummy_weight(E, tile, cfg.hidden_size, gate.dtype)
        else:
            down = _concat_cold([u.colds[0] for u in group.units])
            gate = _dummy_weight(E, tile, cfg.intermediate_size, down.dtype)
            up = _dummy_weight(E, tile, cfg.intermediate_size, down.dtype)

        wrapper = PartialMoEWrapper(cfg, self.cpuinfer, kernel_key=key.kernel)
        wrapper.load_weights_from_tensors(gate, up, down)
        group.wrapper = wrapper
        # full 텐서 소멸 (계약 ③): pack이 끝났으므로 소유권은 C++다. 여기 참조를
        # 놓지 않으면 cold weight 한 벌이 host RAM에 그대로 남는다.
        for unit in group.units:
            object.__setattr__(unit, "colds", ())

    # ── step-time (포인터 pass-through — staging은 호출자 소유, 계약 ④) ──
    def submit(self, call: ColdCall, *, qlen_ptr: int, x_ptr: int, out_ptr: int,
               cuda_stream: Optional[int] = None) -> None:
        """enqueue-only. CPU 완료를 기다리지 않고 즉시 반환한다 (계약 ④)."""
        g = call.group
        ids_ptr = g.staging.expert_ids_ptr(call.expert)
        submit = (g.wrapper.submit_forward_gateup if g.key.entry == "gateup"
                  else g.wrapper.submit_forward_down)
        submit(qlen_ptr, TOP_K, ids_ptr, x_ptr, out_ptr, cuda_stream, 0)

    def sync(self, cuda_stream: Optional[int] = None) -> None:
        if cuda_stream is None:
            self.cpuinfer.sync()
        else:
            self.cpuinfer.sync_with_cuda_stream(cuda_stream)


# ---------------------------------------------------------------------------
# 텐서 조립
# ---------------------------------------------------------------------------


def _row_stats(group: ColdGroup) -> Tuple[int, int]:
    """(실제 행 수, 패딩 포함 행 수). `_load_group`이 참조를 놓기 전에 불러야 한다."""
    real = pad = 0
    for u in group.units:
        for c in u.colds:
            real += c.real_rows
            pad += c.k_pad
    return real, pad


def _set_kindex(dst, shards: Sequence[LinearColdShard]) -> None:
    """`LinearColdShard`들을 kt `KIndex`로. expert 블록을 이어 붙인다.

    `real_rows`는 타일 올림 **전**의 행 수다 — dense 계산에는 패딩 0 행이 무해하고
    sparse 마스크의 tail을 끄는 데만 쓰인다. 패딩이 없으면 비워 둔다 (kt가 `k(e)`와
    같다고 읽는다).
    """
    row_off = [0]
    idx: List[int] = []
    real: List[int] = []
    for s in shards:
        row_off.append(row_off[-1] + s.k_pad)
        idx.extend(int(v) for v in s.k_index.to(torch.int32).tolist())
        real.append(int(s.real_rows))
    if len(idx) != row_off[-1]:
        raise PlanError(f"k_index 길이 {len(idx)} != Σ k_pad {row_off[-1]}")
    dst.row_off = row_off
    dst.idx = idx
    if any(r != row_off[e + 1] - row_off[e] for e, r in enumerate(real)):
        dst.real_rows = real


def _set_dummy_kindex(dst, num_experts: int, tile: int, axis_len: int) -> None:
    """빈 슬롯 대체 (`moe.hpp:513` `no weight source`). 타일 하나짜리 최소 기하.

    kt가 세 proj를 전부 요구하는 동안의 임시다 — 인덱스는 축 안에 있기만 하면
    되고(0..tile-1), weight가 0이라 계산에 참여해도 무해하다. 진짜 비용은 이
    슬롯의 C 버퍼 풀이며 그것은 노드 테이블로 깎는다 (`_config` 참조).
    """
    if tile > axis_len:
        raise PlanError(f"dummy slot needs {tile} rows but the axis is {axis_len}")
    dst.row_off = [e * tile for e in range(num_experts + 1)]
    dst.idx = list(range(tile)) * num_experts


def _concat_cold(shards: Sequence[LinearColdShard]) -> torch.Tensor:
    """expert 블록 `[N, k_pad(e)]`를 이어 붙인 flat 1-D.

    wrapper의 검증은 rank가 아니라 **원소 수** `N × Σₑ k(e)` 하나다 — 인덱스형은
    expert마다 k가 다를 수 있어 `[E, N, k]` 균일 적층이 아니기 때문이다.
    """
    return torch.cat([s.w_flat.reshape(-1) for s in shards]).contiguous()


def _dummy_weight(num_experts: int, tile: int, n: int, dtype) -> torch.Tensor:
    return torch.zeros(num_experts * n * tile, dtype=dtype)

"""K-split의 기하 — 티어, 밴드, shard, 정렬 단위 (계약 ①).

`moe/prism/plan.py`에서 **expert 축과 무관한 것만** 올라왔다. 남은 판정 기준은
하나다: "expert가 없어도 뜻이 통하는가". `Tier`(거처)와 `BandSpec`(K축 반개구간)은
통하고, `Proj`(gate/up/down)와 `ModelDims`(num_experts, top_k)는 안 통한다 —
후자는 각 오프로드의 plan 모듈이 자기 어휘로 소유한다.

이 모듈은 stdlib 외에 아무것도 import하지 않는다. `plan.py`가 지키던 성질을
그대로 물려받는 것이다 — 기하가 runtime을 알게 되는 순간 경계가 무너진다.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# K-축 경계 정렬 단위 = **페어** (계약 ① 2026-08-25).
#
# 초판의 `ROW_GROUP = 64`는 폐기됐다. 그것은 요구가 아니라 "AMX K_STEP(32)의
# 배수라 안전"이라는 보수적 선택이었고, 그 K_STEP은 cold **커널의 packed 저장**
# 성질이지 plan의 성질이 아니다 — 타일 경계까지의 올림은 로더가 하고 커널 안에서
# 끝난다 (`kernels.cold_pack_tile_rows`). plan/자산이 지켜야 하는 것은 페어뿐이다.
#
# 값어치는 planner 해상도다: down은 K=512라 %32면 per-expert 크기 선택지가 16개
# 뿐인데, 가변 per-expert 예산 배분이 이 스키마의 존재 이유다.
ROW_GROUP = 2
# N-축 shard 경계 정렬 단위 (AMX pack N-타일; 값은 pack 확인 후 조정 가능)
COL_GROUP = 32
# 인접 입력채널 페어 마스킹 단위 (= ROW_GROUP). k2wl2 점수가 페어 단위이므로
# (calib pairimp: sqrt(a0*x0^2 + a1*x1^2 + 2c*x0*x1)) 마스크 길이는
# K/PAIR_GROUP이다. 밴드 경계가 페어를 쪼개면 두 티어가 같은 페어의 반쪽씩
# 갖게 되어 어느 쪽도 점수를 재구성할 수 없다 — ROW_GROUP이 PAIR_GROUP의
# 배수라는 사실이 그것을 막는다 (import 시 확인).
PAIR_GROUP = 2
assert ROW_GROUP % PAIR_GROUP == 0, "band 경계가 마스킹 페어를 쪼갤 수 있다"


class PlanError(ValueError):
    """Plan 파싱/검증 실패. 전부 startup hard error다."""


class Tier(str, Enum):
    HOT = "hot"    # VRAM 상주, GPU 계산
    WARM = "warm"  # pinned host 상주, GPU가 UVA로 제자리 읽는다, GPU 계산
    COLD = "cold"  # pageable host(NUMA-local) 상주, CPU 계산 (큰 M에선 GPU가 읽는다)


@dataclass(frozen=True)
class BandSpec:
    """K-축 반개구간 [start, end)와 그 티어. 경계는 페어(ROW_GROUP) 배수."""

    start: int
    end: int
    tier: Tier


@dataclass(frozen=True)
class NumaShard:
    """cold 출력의 N-축 반개구간 [n_start, n_end)를 담당하는 NUMA node.

    K축(BandSpec)과 **직교**하다는 것이 요점이다: 밴드가 "무엇을 CPU가 계산하나"를
    정하고, shard는 그 CPU 몫을 소켓들이 어떻게 나누나를 정한다.
    """

    node: int
    n_start: int
    n_end: int


@dataclass(frozen=True)
class KernelSpec:
    """model-global 커널 선택. startup에 구현체로 resolve된 뒤 문자열은 소멸.

    cold의 저장 형식(pack)은 cpu_cold 키가 함의한다 (별도 codec 없음).
    """

    gpu_warm: str
    cpu_cold: str

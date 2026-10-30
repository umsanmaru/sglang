"""Prism 공유 코어 — K-split 티어링의 **축 무관한** 부분.

Prism은 하나의 아이디어다: 행렬곱의 K축(합의 축)을 거처가 다른 세 조각
(HOT=VRAM / WARM=pinned host / COLD=pageable host)으로 자르고, 각자 부분합을
낸 뒤 fp32로 더한다. 어느 행을 어디에 두든 결과가 같다는 것이 이 분할의 근거이고
(계약 ⑤), 그래서 배치가 policy(Plan)가 된다.

그 아이디어에는 expert 축이 필요 없다. MoE 오프로드(`layers/moe/prism/`)는
거기에 "토큰마다 쓰는 W가 다르다"를 얹은 것이고, dense linear 오프로드
(`layers/prism/linear/`)는 안 얹은 것이다. 이 패키지는 **둘의 교집합**만 갖는다:

    numa.py      NUMA 조회·바인딩·pinned 할당 (expert 언급 0)
    kernels.py   커널 이름표와 그 이름이 함의하는 정렬·타일·스토어 태그
    geometry.py  Tier / BandSpec / NumaShard / KernelSpec / K·N 정렬 상수

의존은 아래로만 흐른다::

    layers/moe/prism/    ──┐
                           ├──→  layers/prism/   (여기)
    layers/prism/linear/ ──┘

역방향 import는 없다. 이 패키지가 `moe`를 알게 되면 dense 경로가 MoE 런너를
끌고 오게 되고(`layers/moe/__init__.py`가 `MoeRunner`를 import한다), 그 순간
분리의 의미가 사라진다.

**이행 상태 (2026-08-31).** MoE prism 5,000줄은 `layers/moe/prism/`에 그대로
있고, 여기로 올라온 세 모듈을 자기 이름으로 re-export한다 — 기존 import 경로
(`moe.prism.numa`, `moe.prism.kernels`, `moe.prism.plan.Tier`)가 전부 살아 있어
기존 42개 파일·테스트 25개가 무수정이다. 최종형은 MoE도 `layers/prism/moe/`로
내려오는 것이지만, 그 이동은 이 작업의 범위가 아니다 (TODO.md "dense 확장").
"""

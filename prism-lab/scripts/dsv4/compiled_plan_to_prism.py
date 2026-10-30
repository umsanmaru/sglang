#!/usr/bin/env python
"""(shim) 3모델 공용 컨버터 scripts/compiled_plan_to_prism.py 로 넘긴다.

예전 CLI `compiled_plan_to_prism.py <plan.json.gz> <out.json> [옵션…]` 을 그대로 받는다
(DSV4 프리셋은 graph_id로 자동 선택된다). 새 CLI는 `<src>… [-o out] [옵션…]`.
"""
import os
import runpy
import sys

NEW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "compiled_plan_to_prism.py")
argv = sys.argv[1:]
if len(argv) >= 2 and not argv[0].startswith("-") and not argv[1].startswith("-") and argv[1].endswith(".json"):
    argv = [argv[0], "-o", argv[1]] + argv[2:]   # 예전 위치 인자 (src out) → (src -o out)
sys.argv = [NEW] + argv
runpy.run_path(NEW, run_name="__main__")

#!/usr/bin/env bash
# 기동한 서버에 한 번 물어보고 결과를 찍는다. Phase 1/2의 게이트.
#
#   scripts/qwen38/smoke.sh [port] [--wait]
#   NTOK=64 PROMPT="…" scripts/qwen38/smoke.sh 30140
#
# greedy(temperature 0)로 뽑는 이유: stock과 prism의 출력을 **토큰 단위로 대조**
# 하려면 표집이 결정적이어야 한다. Phase 2의 판정이 "같은 토큰이 나오는가"다.
set -uo pipefail
PORT=${1:-30140}
PROMPT=${PROMPT:-"The capital of France is"}
NTOK=${NTOK:-32}
URL=http://127.0.0.1:$PORT

if ! curl -sS -m 15 -o /dev/null "$URL/health" 2>/dev/null; then
  echo "서버가 포트 $PORT 에서 응답하지 않는다." >&2
  echo "  · 아직 로딩 중일 수 있다 (55.6 GB — 콜드 캐시면 수 분)." >&2
  echo "  · 기동:  scripts/qwen38/run_qwen38.sh [plan.json]" >&2
  echo "  · 대기:  scripts/qwen38/smoke.sh $PORT --wait" >&2
  [[ "${2:-}" == "--wait" ]] || exit 1
  echo -n "대기 중"
  for _ in $(seq 1 600); do
    curl -sS -m 15 -o /dev/null "$URL/health" 2>/dev/null && { echo " 준비됨"; break; }
    echo -n .; sleep 2
  done
  curl -sS -m 15 -o /dev/null "$URL/health" 2>/dev/null || { echo " 시간 초과" >&2; exit 1; }
fi

REQ=$(PROMPT="$PROMPT" NTOK="$NTOK" python3 -c "
import json, os
print(json.dumps({'text': os.environ['PROMPT'],
                  'sampling_params': {'temperature': 0,
                                      'max_new_tokens': int(os.environ['NTOK'])}}))")
RESP=$(curl -sS -m 600 "$URL/generate" -H 'Content-Type: application/json' -d "$REQ") || {
  echo "요청 실패" >&2; exit 1; }
PROMPT="$PROMPT" printf '%s' "$RESP" | PROMPT="$PROMPT" python3 -c "
import json, os, sys
raw = sys.stdin.read()
try:
    d = json.loads(raw)
except json.JSONDecodeError:
    print('서버가 JSON이 아닌 것을 돌려줬다:', raw[:300], file=sys.stderr); raise SystemExit(1)
print('프롬프트:', repr(os.environ['PROMPT']))
print('출력    :', repr(d.get('text', '')))
m = d.get('meta_info', {})
print('토큰    :', m.get('completion_tokens'),
      '| finish:', (m.get('finish_reason') or {}).get('type'))
"

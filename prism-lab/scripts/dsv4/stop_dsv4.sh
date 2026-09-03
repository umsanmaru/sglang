#!/usr/bin/env bash
# 포트 30111(기본) DSV4 서버 종료. pkill -f 패턴은 자기 자신을 죽이니 awk로 pid를 뽑는다.
PORT=${1:-30111}
PIDS=$(ps -eo pid,cmd | awk -v p="port $PORT" 'index($0,p) && !/awk/ {print $1}')
[ -n "$PIDS" ] && kill $PIDS && sleep 8
PIDS=$(ps -eo pid,cmd | awk -v p="port $PORT" 'index($0,p) && !/awk/ {print $1}')
[ -n "$PIDS" ] && kill -9 $PIDS
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i 0; free -g | sed -n 2p

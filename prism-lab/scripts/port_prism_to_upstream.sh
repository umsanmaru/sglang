#!/usr/bin/env bash
# Prism을 upstream 기반 트리(PR 워크트리)로 이식한다. 포크 트리는 읽기만 한다.
#
# 근거(2026-08-31 조사): 커밋된 prism은 68파일 19,883줄 추가에 기존 파일 수정 2개뿐이고,
# upstream에 없는 것은 quant_method_registry.py / jit_kernel/ / deepseek_v4_debug_utils.py 셋이다.
# 그래서 "PR을 포크에 merge"(= upstream 7,483커밋 merge) 대신 이 방향으로 옮긴다.
set -euo pipefail
SRC=${SRC:-/home/um3maru/prism-sglang/sglang}          # 포크(prism 원본)
DST=${DST:?대상 트리 (예: /home/um3maru/prism-sglang/sglang-glm)}
S=$SRC/python/sglang; D=$DST/python/sglang

copy() {  # copy <상대경로>
  local rel=$1
  mkdir -p "$(dirname "$D/$rel")"
  rm -rf "$D/$rel"
  cp -r "$S/$rel" "$D/$rel"
  find "$D/$rel" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
  echo "  + $rel"
}

echo "== 1) prism 소스 복사 =="
copy srt/layers/moe/prism            # MoE 오프로드 본체
copy srt/layers/prism                # 공유 코어(kernels/numa/geometry/store) + dense linear
copy srt/layers/moe/quant_method_registry.py
copy srt/layers/linear_method_registry.py   # dense: LinearBase 래퍼 슬롯
copy jit_kernel                      # 자체 JIT 로더 + prism .cuh (upstream의 sglang/kernels와 이름 충돌 없음)
mkdir -p "$D/srt/debug_utils"
cp "$S/srt/debug_utils/deepseek_v4_debug_utils.py" "$D/srt/debug_utils/" 2>/dev/null \
  && echo "  + srt/debug_utils/deepseek_v4_debug_utils.py" || echo "  (deepseek_v4_debug_utils 없음 — swiglu_limit 미사용 모델이면 무관)"
rm -rf "$DST/test/prism"; cp -r "$SRC/test/prism" "$DST/test/prism"
find "$DST/test/prism" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
echo "  + test/prism"

echo "== 2) FusedMoE 훅 1곳 삽입 =="
python3 - "$DST" <<'PY'
import re, sys, pathlib
dst = pathlib.Path(sys.argv[1])
p = dst / "python/sglang/srt/layers/moe/fused_moe_triton/layer.py"
src = p.read_text()
if "SGLANG_PRISM_PLAN" in src:
    print("  (이미 패치됨)"); raise SystemExit
anchor = "        _validate_hpc_ops_quant_method(self.quant_method)"
assert anchor in src, "앵커(_validate_hpc_ops_quant_method)를 못 찾았다 — upstream 구조가 바뀌었다"
hook = '''        # --- prism: K-split hot/warm/cold MoE 오프로드. env SGLANG_PRISM_PLAN 로만 활성화된다.
        if os.environ.get("SGLANG_PRISM_PLAN"):
            from sglang.srt.layers.moe.quant_method_registry import (
                maybe_wrap_moe_quant_method,
            )

            if getattr(self, "layer_id", None) is None:
                self.layer_id = layer_id
            self.quant_method = maybe_wrap_moe_quant_method(
                self, self.quant_method, server_args
            )
        # --- prism end ---
'''
src = src.replace(anchor, hook + anchor, 1)
if not re.search(r"^import os$", src, re.M):
    src = src.replace("import logging", "import logging\nimport os", 1)
p.write_text(src)
print("  patched", p.relative_to(dst))
PY

echo "== 2b) loader.py device 왕복 opt-out 삽입 =="
# device_loading_context는 CPU offload용이고 판정 기준이 `device.type == "cpu"` 하나뿐이라,
# 파라미터를 의도적으로 host에 만드는 prism을 offload로 오해한다. 그대로 두면 층마다 full
# expert weight를 GPU로 올렸다 내린 뒤 그 사본을 버린다 (DSV4-Flash 실측 43층 186 s;
# GLM-5.3-Flash는 층당 7.25 GB × 45층으로 더 크다). 실패가 조용하다 — 느려질 뿐 에러가 없다.
# loader.py는 upstream 파일이라 복사하면 안 된다: 두 트리가 7,475커밋 떨어져 있고 이 함수는
# 최신 upstream에서 이미 stage_module_for_post_load로 리팩터링됐다(그쪽도 같은 왕복을 한다).
# 그래서 layer.py와 같은 앵커 삽입 방식을 쓴다.
python3 - "$DST" <<'PATCH_PY'
import sys, pathlib
dst = pathlib.Path(sys.argv[1])
p = dst / "python/sglang/srt/model_loader/loader.py"
src = p.read_text()
if "keeps_params_on_host" in src:
    print("  (이미 패치됨)"); raise SystemExit
anchor = ('def device_loading_context(module: torch.nn.Module, '
          'target_device: torch.device):\n    if target_device.type == "cpu":')
assert src.count(anchor) == 1, "앵커(device_loading_context)를 못 찾았다 — upstream 구조가 바뀌었다"
tail = "\n        return\n"
i = src.index(anchor)
j = src.index(tail, i) + len(tail)      # target_device=cpu 조기 반환 블록의 끝
lines = [
    "",
    "    # 파라미터의 거처가 method의 계약인 경우는 왕복을 건너뛴다. 이 컨텍스트는 CPU",
    "    # offload(= 임시로 CPU에 내려둔 파라미터)를 위한 것이고 판정 기준이 device.type",
    "    # 하나뿐이라, 의도적으로 host에 파라미터를 만드는 method는 오해를 받는다:",
    "    # Prism의 full expert weight는 host 상주가 설계이며(hot 밴드만 VRAM) 훅이 그 CPU",
    "    # 텐서를 그대로 읽으므로, 왕복은 층당 full weight를 올렸다 내린 뒤 그 GPU 사본을",
    "    # 버리는 순수 낭비다.",
    '    if getattr(getattr(module, "quant_method", None), "keeps_params_on_host", False):',
    "        yield module",
    "        return",
    "",
]
p.write_text(src[:j] + "\n".join(lines) + src[j:])
print("  patched", p.relative_to(dst))
PATCH_PY

echo "== 2c) LinearBase 훅 1곳 삽입 (dense) =="
# MoE의 FusedMoE 훅과 같은 자리·같은 이유. `LinearBase.__init__`이 quant_method를 고른
# 직후이면서 **서브클래스의 create_weights() 전**이어야 한다 — 그 사이가 래퍼가
# 파라미터의 거처를 정할 수 있는 유일한 창이다 (Prism dense는 full weight를 host에
# 만들고 K-슬라이스한 뒤 원본을 놓는다).
python3 - "$DST" <<'PY2C'
import sys, pathlib
dst = pathlib.Path(sys.argv[1])
p = dst / "python/sglang/srt/layers/linear.py"
src = p.read_text()
if "maybe_wrap_linear_quant_method" in src:
    print("  (이미 패치됨)"); raise SystemExit

hook = '''        # --- prism: dense K-split 오프로드의 래퍼 슬롯. `FusedMoE.__init__`의
        # `maybe_wrap_moe_quant_method`와 같은 자리다. quant_method가 정해진 뒤이면서
        # **서브클래스의 `create_weights()` 전**이라, 래퍼가 파라미터의 거처를 정할 수
        # 있는 유일한 창이다. `prefix`를 명시로 넘기는 이유는 LinearBase에 layer_id가
        # 없어서고, tp_rank/tp_size는 서브클래스가 이 함수 반환 후에 대입하므로 아직
        # 읽을 수 없다. import를 함수 안에 두는 것은 registry가 prism 소유 파일이기
        # 때문이다 — 그 파일이 없는 트리에서도 linear.py 자체는 import돼야 한다.
        from sglang.srt.layers.linear_method_registry import (
            maybe_wrap_linear_quant_method,
        )

        self.quant_method = maybe_wrap_linear_quant_method(
            self, self.quant_method, prefix
        )
        # --- prism end ---

'''

# 앵커 후보를 순서대로 시도한다. base마다 `LinearBase.__init__`의 끝 모양이 다르다:
# 최신 upstream은 quant_method 선택 뒤에 `wrap_method_with_debug_kernel_once` 블록이
# 붙어 있고(그 **앞**에 넣어야 debug 래퍼가 prism의 apply를 계측한다), 구버전 포크에는
# 그 블록이 없어 `forward` 정의가 곧 다음 문장이다.
CANDIDATES = [
    ("        if self.quant_method is not None:\n"
     "            wrap_method_with_debug_kernel_once(", "before"),
    ("    def forward(self, x: torch.Tensor) -> torch.Tensor:\n"
     "        raise NotImplementedError", "before"),
]
for anchor, where in CANDIDATES:
    n = src.count(anchor)
    if n == 1:
        src = src.replace(anchor, hook + anchor, 1)
        p.write_text(src)
        print("  patched", p.relative_to(dst), f"(앵커: {anchor.strip().splitlines()[0][:48]}…)")
        break
    if n > 1:
        raise SystemExit(f"앵커가 {n}곳에 있다 — 유일하지 않아 어디 붙을지가 파일 순서에 달린다: {anchor[:60]}")
else:
    raise SystemExit("LinearBase.__init__ 앵커를 하나도 못 찾았다 — upstream 구조가 바뀌었다")
PY2C

echo "== 3) 검증 =="
grep -c "SGLANG_PRISM_PLAN" "$DST/python/sglang/srt/layers/moe/fused_moe_triton/layer.py"
python3 -c "import ast,sys; ast.parse(open('$DST/python/sglang/srt/layers/moe/fused_moe_triton/layer.py').read()); print('  layer.py 문법 OK')"
ls "$D/srt/layers/moe/prism/method.py" "$D/srt/layers/prism/kernels.py" "$D/jit_kernel/prism_gemv_fp8.py" >/dev/null && echo "  파일 배치 OK"
grep -c "maybe_wrap_linear_quant_method" "$D/srt/layers/linear.py" >/dev/null \
  && ls "$D/srt/layers/linear_method_registry.py" "$D/srt/layers/prism/linear/method.py" >/dev/null \
  && echo "  dense 훅 + registry + method OK"
python3 -c "import ast; ast.parse(open('$D/srt/layers/linear.py').read()); print('  linear.py 문법 OK')"
# dense도 왕복 opt-out을 선언해야 한다 (MoE와 같은 계약) — 없으면 층마다 full weight를
# GPU로 올렸다 내린 뒤 버린다. 에러 없이 느려지기만 하므로 여기서 센다.
test "$(grep -c keeps_params_on_host "$D/srt/layers/prism/linear/method.py")" -gt 0 \
  && echo "  dense 왕복 opt-out 선언 OK"
# 왕복 opt-out은 선언(method.py)과 소비(loader.py) 양쪽이 다 있어야 동작한다 — 한쪽만
# 있으면 조용히 느려지므로 여기서 둘 다 센다.
test "$(grep -c keeps_params_on_host "$D/srt/layers/moe/prism/method.py")" -gt 0 \
  && test "$(grep -c keeps_params_on_host "$D/srt/model_loader/loader.py")" -gt 0 \
  && echo "  왕복 opt-out 선언+소비 양쪽 OK"
echo "완료. 다음: env(prism-glm)에서 kt-kernel 재빌드 + tilelang 타일 패치."

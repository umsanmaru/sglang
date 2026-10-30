"""dense prism의 sglang 접점 e2e — 훅 → 절단 → forward (스크립트, pytest 수집 안 됨).

`test_linear_executor.py`가 수치를, 여기서는 **배관**을 본다: registry 훅이
plan에 있는 linear만 감싸는가, 파라미터가 CPU에 잡히는가, 로딩 후 full 텐서가
소멸하는가, 실제 `LinearBase.forward`가 값을 내는가, gate가 앞 절반이라
SiluAndMul이 이어지는가.

pytest로 수집하지 않는 이유는 1-rank 분산 초기화가 필요해서다 (RowParallelLinear
.forward가 `get_tp_group()`을 무조건 부른다) — 같은 프로세스의 다른 테스트와
섞이면 곤란하다. 실행::

    PATH=~/miniconda3/envs/prism-e2e/bin:$PATH \
    LD_LIBRARY_PATH=~/miniconda3/envs/prism-e2e/lib:$LD_LIBRARY_PATH \
    python test/prism/e2e_linear_method.py
"""

import json, os, tempfile, torch
K, I = 512, 256
d = tempfile.mkdtemp()
plan = {"schema_version":1,"model_id":"t","dims":{"num_layers":1,"dtype":"bfloat16"},
        "kernels":{"gpu_warm":"gemv_worklist","cpu_cold":"kt_tile_k2_bf16"},
        "projs":{
          "mlp.gate_up_proj":{"k":K,"n":2*I,"halves":{
              "gate":{"bands":[[0,128,"hot"],[128,K,"warm"]]},
              "up":  {"bands":[[0,K,"hot"]]}}},
          "mlp.down_proj":{"k":I,"n":K,"bands":[[0,64,"warm"],[64,I,"hot"]]}}}
p = os.path.join(d, "plan.json"); json.dump(plan, open(p,"w"))
os.environ["SGLANG_PRISM_LINEAR_PLAN"] = p

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29591")
from sglang.srt.distributed import init_distributed_environment, initialize_model_parallel
init_distributed_environment(world_size=1, rank=0, local_rank=0,
                             distributed_init_method="env://", backend="nccl")
initialize_model_parallel(tensor_model_parallel_size=1)

torch.set_default_dtype(torch.bfloat16)
from sglang.srt.layers.linear import MergedColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.linear_method_registry import is_wrapped_linear_method

gu = MergedColumnParallelLinear(K, [I, I], bias=False, params_dtype=torch.bfloat16,
                                prefix="model.layers.0.mlp.gate_up_proj", tp_rank=0, tp_size=1)
dn = RowParallelLinear(I, K, bias=False, params_dtype=torch.bfloat16,
                       prefix="model.layers.0.mlp.down_proj", tp_rank=0, tp_size=1)
other = RowParallelLinear(I, K, bias=False, params_dtype=torch.bfloat16,
                          prefix="model.layers.0.self_attn.o_proj", tp_rank=0, tp_size=1)
print("gate_up 래핑:", is_wrapped_linear_method(gu.quant_method, "prism_linear"))
print("down    래핑:", is_wrapped_linear_method(dn.quant_method, "prism_linear"))
print("plan 밖 o_proj 래핑:", is_wrapped_linear_method(other.quant_method, "prism_linear"), "(False여야 함)")
print("weight 거처:", gu.weight.device, gu.weight.shape, gu.weight.dtype)

torch.manual_seed(0)
with torch.no_grad():
    gu.weight.normal_(); dn.weight.normal_()
W = {"gu": gu.weight.detach().clone(), "dn": dn.weight.detach().clone()}

for m in (gu, dn):
    m.quant_method.process_weights_after_loading(m)
print("소멸 후 weight numel:", gu.weight.numel(), dn.weight.numel())

dev = torch.device("cuda:0")
for name, mod, w in (("gate_up", gu, W["gu"]), ("down", dn, W["dn"])):
    for M in (1, 8):
        x = torch.randn(M, w.shape[1], dtype=torch.bfloat16, device=dev)
        out, _ = mod(x)
        ref = x.float() @ w.float().t().to(dev)
        rel = ((out.float()-ref).abs().max()/ref.abs().max()).item()
        print(f"  forward {name:8s} M={M}: {tuple(out.shape)} rel={rel:.2e}")

# SiluAndMul이 이어붙는지 (gate가 앞 절반이어야 한다)
x = torch.randn(4, K, dtype=torch.bfloat16, device=dev)
gate_up, _ = gu(x)
# SiluAndMul과 같은 연산 (그 클래스는 server_args를 요구한다 — 하네스 한계)
act = torch.nn.functional.silu(gate_up[:, :I]) * gate_up[:, I:]
ref = x.float() @ W["gu"].float().t().to(dev)
ref_act = torch.nn.functional.silu(ref[:, :I]) * ref[:, I:]
print(f"  SiluAndMul 연결: {tuple(act.shape)} rel={((act.float()-ref_act).abs().max()/ref_act.abs().max()).item():.2e}")

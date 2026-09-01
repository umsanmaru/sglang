"""Prism dense — MoE가 아닌 linear layer의 K-split 오프로드.

qkvo(`wq_a`/`wq_b`/`wo_a`/`wo_b`)와 dense MLP(`gate_up_proj`/`down_proj`)의
weight를 K축으로 hot/warm/cold 세 조각으로 자른다. 아이디어는 MoE 쪽
(`layers/moe/prism/`)과 같고, 공유 기하는 `layers/prism/`이 소유한다.

**MoE와 갈리는 것은 expert 축 하나뿐이고, 그 하나가 아래를 전부 지운다:**

| | MoE | dense |
|---|---|---|
| plan 좌표 | `(layer, expert, proj)` — proj는 gate/up/down enum | `(layer, proj)` — proj는 열린 이름(`self_attn.wq_b`) |
| K 조달 | `dims.k_of(proj)` (hidden 또는 intermediate) | proj마다 자기 `k`/`n` (파생 공식이 없다) |
| partial 버퍼 | `[M, k, N]` — pair (m, j)가 rejoin 좌표 | `[M, N]` |
| pair 처리 | worklist/grouping (토큰을 expert로 묶기) | **없음** — K-조각당 GEMM/GEMV 하나 |
| rejoin | fp32 Σ → act → bf16 / fp32 Σ → **router 가중 k축 합** | fp32 Σ (→ act) → bf16 |
| 활성화 | `SGLANG_PRISM_PLAN` | `SGLANG_PRISM_LINEAR_PLAN` |

두 plan은 **독립적으로 켜고 끈다** — MoE만, dense만, 둘 다를 env 조합으로 스윕할
수 있어야 벤치가 성립한다. 그 대가로 VRAM/PCIe/CPU 예산을 두 plan이 따로 정하므로
합이 하드웨어를 넘을 수 있다. 그것은 planner의 책임이고 런타임은 검사하지 않는다.

**v1 범위 밖 (의도적):** sparsity. MoE의 calib 자산은 `[L, E, K]` 축이라 dense의
`[L, K]`와 포맷이 다르고, 별도 자산 생성이 선행되어야 한다.
"""

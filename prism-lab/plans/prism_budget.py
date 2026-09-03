#!/usr/bin/env python3
"""Prism tier budget calculator (nutella3: RTX 5090 32 GiB, 2x257 GiB NUMA).

hot/warm VRAM & cold RAM for a given (model, hot rows, warm rows) — with the
per-format bytes-per-element that gen_uniform_plan's [budget] line gets wrong
for fp8 (it prints bf16 bytes for every non-mxfp4 store).
"""
GIB = 2 ** 30

MODELS = {
    # name: (moe_layers, E, H, I, store, non_expert_gib_resident, host_side_extra_gib)
    "GLM-5.3-Flash":       dict(L=42, E=288, H=4096, I=2048, store="fp8",   nonexp=15.5, host_extra=0.0),
    "Qwen3.8-Flash-Next":  dict(L=48, E=512, H=2560, I=640,  store="bf16",  nonexp=10.2, host_extra=95.4),  # host_extra = PLE ngram table
    "DeepSeek-V4-Flash":   dict(L=43, E=256, H=4096, I=2048, store="mxfp4", nonexp=12.0, host_extra=0.0),
}
BPE = {"bf16": 2.0, "fp8": 1.0 + 4.0 / (128 * 128), "mxfp4": 0.5 + 1.0 / 32}
KALIGN = {"bf16": 2, "fp8": 128, "mxfp4": 32}

def rows_bytes(m, proj, rows):
    """bytes for `rows` K-rows of one proj, summed over experts, one layer."""
    b = BPE[m["store"]]
    N = m["I"] if proj in ("gate", "up") else m["H"]
    return rows * m["E"] * N * b

def report(name):
    m = MODELS[name]
    ka = KALIGN[m["store"]]
    Kgu, Kdn = m["H"], m["I"]
    total = m["L"] * (2 * rows_bytes(m, "gate", Kgu) + rows_bytes(m, "down", Kdn))
    print(f"\n=== {name} ({m['store']}, {m['L']} MoE layers, E={m['E']}, H={m['H']}, I={m['I']}, K-align {ka}) ===")
    print(f"  routed expert weights total : {total/GIB:8.1f} GiB")
    print(f"  non-expert GPU resident     : {m['nonexp']:8.1f} GiB   host-side extra: {m['host_extra']:.1f} GiB")
    print(f"  one K-row (gate|up)         : {rows_bytes(m,'gate',1)/2**20:8.2f} MiB/layer   x{m['L']} = {m['L']*rows_bytes(m,'gate',1)/GIB:.2f} GiB")
    print(f"  one K-row (down)            : {rows_bytes(m,'down',1)/2**20:8.2f} MiB/layer   x{m['L']} = {m['L']*rows_bytes(m,'down',1)/GIB:.2f} GiB")
    print(f"  {'hot(gu,dn) rows':>18} | {'hot GiB':>8} | {'VRAM used':>9} | {'cold GiB':>8} | {'cold/node':>9} | fits?")
    vram_cap, overhead = 31.4, 2.6      # 5090 usable, + ctx/workspace/graph/act
    for gu, dn in [(0,0)] + [(g, d) for g in (ka, 2*ka, 3*ka, 4*ka, 6*ka, 8*ka) for d in (0, ka, 2*ka) if g <= Kgu and d <= Kdn]:
        hot = m["L"] * (2 * rows_bytes(m, "gate", gu) + rows_bytes(m, "down", dn)) / GIB
        cold = total / GIB - hot
        vram = m["nonexp"] + hot + overhead
        ok = "OK " if vram <= vram_cap and (cold / 2 + m["host_extra"] / 2) < 240 else "no "
        print(f"  {gu:6d},{dn:6d}      | {hot:8.2f} | {vram:9.1f} | {cold:8.1f} | {cold/2:9.1f} | {ok} ({100*gu/Kgu:.1f}%/{100*dn/Kdn:.1f}% hot)")

for k in MODELS: report(k)
print("\nnote: VRAM cap 31.4 GiB, overhead 2.6 GiB (cuda ctx + graph + activations + KV@16k).")
print("      cold/node assumes the 2-node uniform N-shard; node capacity 257 GiB each (240 usable).")

import torch

from sglang.srt.layers.moe.prism.weights import WarmBand


def _band(e=16, rows=5, n=7):
    return WarmBand(k_offset=0, weights=torch.arange(e * rows * n, dtype=torch.bfloat16)
                    .reshape(e, rows, n).pin_memory())


def test_gather_kernel_bitwise():
    from sglang.jit_kernel.prism_gather import gather_bands_from_pinned

    band = _band(e=32, rows=8, n=64)  # 8*64*2 = 1024B, multiple of 16
    sel = torch.tensor([5, 17, 2], dtype=torch.int32, device="cuda")
    dst = torch.zeros(3, 8, 64, dtype=torch.bfloat16, device="cuda")
    gather_bands_from_pinned(band.weights, sel, dst, torch.cuda.current_stream())
    torch.cuda.synchronize()
    assert torch.equal(dst.cpu(), band.weights[[5, 17, 2]])

#!/usr/bin/env python3
"""Time selective_scan_fn at the model's largest sequence length, and size the
reference path's intermediates. Both are quoted in README section 3.1.

The latency figure was carried from earlier notes at 29.2 ms and is actually
17.08 ms on this card, so it is measured here rather than quoted.

Usage: python scripts/measure_scan.py results/scan_timing.json
"""
import json, os, sys, torch
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

B, D, L, N = 1, 72, 2088960, 16
torch.manual_seed(0)
u = torch.randn(B, D, L, device="cuda")
delta = torch.rand(B, D, L, device="cuda")
A = -torch.rand(D, N, device="cuda") - 0.5
Bm = torch.randn(B, N, L, device="cuda")
Cm = torch.randn(B, N, L, device="cuda")
Dm = torch.randn(D, device="cuda")

for _ in range(3):
    selective_scan_fn(u, delta, A, Bm, Cm, Dm, delta_softplus=True)
torch.cuda.synchronize()

times = []
for _ in range(10):
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    selective_scan_fn(u, delta, A, Bm, Cm, Dm, delta_softplus=True)
    e.record()
    torch.cuda.synchronize()
    times.append(s.elapsed_time(e))
times.sort()
median = times[len(times) // 2]

elems = B * D * L * N
gb = elems * 4 / 1e9
print(f"selective_scan_fn at B={B} d_inner={D} L={L:,} d_state={N}")
print(f"  median over 10: {median:.2f} ms  (min {min(times):.2f}, max {max(times):.2f})")
print(f"  reference intermediate (B,d_inner,L,d_state) fp32: {gb:.2f} GB each")
print(f"  deltaA + deltaB_u together: {2 * gb:.2f} GB")
json.dump({"scan_median_ms": median, "scan_min_ms": min(times), "scan_max_ms": max(times),
           "B": B, "d_inner": D, "L": L, "d_state": N,
           "ref_intermediate_gb": gb, "ref_two_intermediates_gb": 2 * gb,
           "gpu": torch.cuda.get_device_name(0)},
          open(sys.argv[1], "w"), indent=2)
print(f"wrote {sys.argv[1]}")

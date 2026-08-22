#!/usr/bin/env python3
"""Measure what one 4K forward pass actually costs, in memory and in time.

A reproducer's first practical question is whether the model fits on their card.
This answers that, and it measures latency correctly, which the released script
does not: inference_RetinexDual.py brackets the forward with time.time() and no
torch.cuda.synchronize(), so it times the kernel launch rather than the work.

Reported latency is the median of per-iteration CUDA-event timings, taken in
groups so that group-to-group spread is visible. A wall-clock median is printed
alongside as a cross-check. If the two disagree by much, distrust both.

Usage:
  python scripts/measure_footprint.py --repo ~/RetinexDual
  python scripts/measure_footprint.py --repo ~/RetinexDual --sweep
"""
import argparse
import json
import os
import statistics as st
import sys
import time

import torch
import torch.nn.functional as F


def check_image_size(x, mult=128):
    _, _, h, w = x.size()
    return F.pad(x, (0, (mult - w % mult) % mult, 0, (mult - h % mult) % mult), "reflect")


def forward_once(model, x):
    with torch.inference_mode():
        out = model(x)
    while isinstance(out, (tuple, list)):
        out = out[0]
    return out


def sanity_check():
    """Recover a known cost before trusting the harness on an unknown one."""
    n = 8192
    a = torch.randn(n, n, device="cuda", dtype=torch.float16)
    b = torch.randn(n, n, device="cuda", dtype=torch.float16)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(10):
        a @ b
    e.record()
    torch.cuda.synchronize()
    ms = s.elapsed_time(e) / 10
    del a, b
    torch.cuda.empty_cache()
    return (2 * n ** 3) / (ms / 1000) / 1e12, ms


def measure(model, h, w, groups, iters, warmup, mode="both"):
    """Measure latency, memory, or both.

    These want opposite settings of cudnn.benchmark and cannot share a process
    cleanly, which is why --mode exists.

    Memory needs benchmark OFF. With it ON, cuDNN trials many convolution
    algorithms and their scratch buffers land in max_memory_allocated, so the
    figure reflects the algorithm search rather than the model's working set.
    Measured with benchmark ON, peak allocated came out non-monotonic in
    resolution (15.64 GiB at 1920x1152 against 15.28 GiB at 3840x2176), which
    cannot be a real working set. With it OFF the same sweep is linear at about
    1.31 GiB per Mpix.

    Latency needs benchmark ON, since that is what the reproduction runs under.
    But torch caches the chosen algorithm per shape, so a benchmark-OFF pass
    first leaves a heuristic choice cached and flipping the flag back does not
    re-trigger the search. Doing both in one process cost about 1.5% on the 4K
    timing (1368.87 ms against 1348.49 ms from a clean run). Run the two modes
    as separate invocations for numbers you intend to publish.
    """
    x = check_image_size(torch.rand(1, 3, h, w, device="cuda"))
    out = {"padded": tuple(x.shape)}

    if mode in ("memory", "both"):
        prev = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = False
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        forward_once(model, x)
        torch.cuda.synchronize()
        out["peak_alloc"] = torch.cuda.max_memory_allocated() / 2 ** 30
        out["peak_reserved"] = torch.cuda.max_memory_reserved() / 2 ** 30
        torch.backends.cudnn.benchmark = prev
        if mode == "memory":
            return out

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        forward_once(model, x)
    torch.cuda.synchronize()

    out["peak_alloc_bench"] = torch.cuda.max_memory_allocated() / 2 ** 30
    out["peak_reserved_bench"] = torch.cuda.max_memory_reserved() / 2 ** 30

    group_medians, wall_all = [], []
    for _ in range(groups):
        ev = []
        for _ in range(iters):
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            t0 = time.perf_counter()
            s.record()
            forward_once(model, x)
            e.record()
            torch.cuda.synchronize()
            wall_all.append((time.perf_counter() - t0) * 1000)
            ev.append(s.elapsed_time(e))
        group_medians.append(st.median(ev))

    out["event_median"] = st.median(group_medians)
    out["group_spread"] = max(group_medians) - min(group_medians)
    out["wall_median"] = st.median(wall_all)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--groups", type=int, default=5)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--sweep", action="store_true",
                    help="also measure smaller resolutions to show how peak memory scales")
    ap.add_argument("--out", default=None,
                    help="write a JSON artifact so the published table has a "
                         "machine-checkable source of truth")
    ap.add_argument("--mode", choices=["both", "memory", "latency"], default="both",
                    help="latency and memory want opposite cudnn.benchmark settings and "
                         "contaminate each other within one process. Use separate "
                         "invocations for numbers you intend to publish")
    args = ap.parse_args()

    repo = os.path.abspath(os.path.expanduser(args.repo))
    weights = args.weights or os.path.join(repo, "pretrained_weights", "UHD_LL.pth")
    sys.path.insert(0, repo)
    os.chdir(repo)

    import mamba_ssm.ops.selective_scan_interface as ssi
    if "site-packages" not in ssi.__file__:
        raise SystemExit(f"[env] local mamba_ssm shim detected at {ssi.__file__}")
    if not torch.cuda.is_available():
        raise SystemExit("[env] CUDA is required")

    from basicsr.models.archs.RetinexDuelSambaFusionFinalization_arch import (
        RetinexDuelSambaFusionFinalization,
    )

    torch.backends.cudnn.benchmark = True
    gpu = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    print(f"[env] {gpu}, {total:.1f} GiB, torch {torch.__version__}", flush=True)

    tflops, mm_ms = sanity_check()
    print(f"[env] harness check: 8192^3 fp16 matmul {mm_ms:.2f} ms -> {tflops:.1f} TFLOPS", flush=True)

    model = RetinexDuelSambaFusionFinalization(
        in_channels=3, out_channels=3, L_n_feat=16, R_n_feat=16
    ).cuda().eval()
    sd = torch.load(weights, map_location="cpu")
    sd = sd["params"] if "params" in sd else sd
    missing, unexpected = model.load_state_dict(sd, strict=False)
    n_params = sum(p.numel() for p in model.state_dict().values())
    n_ckpt = sum(v.numel() for v in sd.values())
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    print(f"[load] code builds {n_params:,} params, checkpoint holds {n_ckpt:,}", flush=True)

    resolutions = [(2160, 3840)]
    if args.sweep:
        resolutions = [(768, 1280), (1088, 1920), (1408, 2560), (2160, 3840)]

    rows = []
    for h, w in resolutions:
        print(f"[measure] {w}x{h}", flush=True)
        try:
            r = measure(model, h, w, args.groups, args.iters, args.warmup, args.mode)
        except torch.cuda.OutOfMemoryError:
            print("           OOM", flush=True)
            rows.append((h, w, None))
            torch.cuda.empty_cache()
            continue
        rows.append((h, w, r))
        bits = []
        if "event_median" in r:
            bits.append(f"{r['event_median']:.2f} ms")
        if "peak_alloc" in r:
            bits.append(f"peak alloc {r['peak_alloc']:.2f} GiB")
        print(f"           {', '.join(bits)}", flush=True)

    if args.out:
        payload = {"gpu": gpu, "torch": torch.__version__, "mode": args.mode,
                   "groups": args.groups, "iters": args.iters, "warmup": args.warmup,
                   "tflops_check": tflops, "rows": []}
        for h, w, r in rows:
            row = {"h": h, "w": w, "oom": r is None}
            if r:
                row.update({k: v for k, v in r.items() if k != "padded"})
                row["padded"] = list(r["padded"])
            payload["rows"].append(row)
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[out] wrote {args.out}", flush=True)

    print(f"\n{gpu}, torch {torch.__version__}, fp32, mode={args.mode}")
    print(f"{args.groups} groups x {args.iters} iterations, {args.warmup} warmup\n")

    def cell(r, key, fmt, unit=""):
        return f"{r[key]:{fmt}}{unit}" if key in r else "n/a"

    print("| Input | Padded | Mpix | CUDA-event median | Wall median | Peak allocated | Reserved (benchmark on) |")
    print("|---|---|---|---|---|---|---|")
    for h, w, r in rows:
        if r is None:
            print(f"| {w}x{h} | | {(h * w) / 1e6:.2f} | OOM | OOM | OOM | OOM |")
            continue
        ph, pw = r["padded"][2], r["padded"][3]
        print(f"| {w}x{h} | {pw}x{ph} | {(ph * pw) / 1e6:.2f} | "
              f"**{cell(r, 'event_median', '.2f', ' ms')}** | {cell(r, 'wall_median', '.2f', ' ms')} | "
              f"**{cell(r, 'peak_alloc', '.2f', ' GiB')}** | {cell(r, 'peak_reserved_bench', '.2f', ' GiB')} |")

    ok = [(h, w, r) for h, w, r in rows if r]
    if ok:
        print()
        for h, w, r in ok:
            ph, pw = r["padded"][2], r["padded"][3]
            parts = [f"{pw}x{ph}:"]
            if "peak_alloc" in r:
                parts.append(f"{r['peak_alloc'] / ((ph * pw) / 1e6):.3f} GiB per Mpix working set,")
            if "event_median" in r:
                parts.append(f"group-to-group spread {r['group_spread']:.2f} ms,")
                parts.append(f"wall vs event "
                             f"{abs(r['wall_median'] - r['event_median']) / r['event_median'] * 100:.2f}%")
            print(" ".join(parts).rstrip(","))
        big = ok[-1][2]
        print()
        if "peak_alloc" in big:
            print(f"Working set at the largest resolution: {big['peak_alloc']:.2f} GiB allocated.")
        if "peak_reserved_bench" in big:
            print(f"With cudnn.benchmark on the allocator reserves "
                  f"{big['peak_reserved_bench']:.2f} GiB, which is the figure that decides "
                  f"whether a card holds this workload.")
        if args.mode == "both":
            print()
            print("NOTE: mode=both. The benchmark-off memory pass leaves a heuristic algorithm")
            print("cached, which costs roughly 1.5% on the 4K timing. Re-run with --mode latency")
            print("for a timing number you intend to publish.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Find the largest input this card can hold, by stepping until it fails.

`measure_footprint.py --find-limit` brackets the ceiling coarsely. This walks it
in 128-row steps, which matters because the answer depends on the allocator and a
coarse probe can land above both ceilings and wrongly suggest they are the same.
That is exactly the error this script was written to correct.

Run it twice, once per allocator setting, to see the difference:

  python scripts/find_ceiling.py --repo ~/RetinexDual --out results/a.json
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
      python scripts/find_ceiling.py --repo ~/RetinexDual --out results/b.json
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F


def check_image_size(x, mult=128):
    _, _, h, w = x.size()
    return F.pad(x, (0, (mult - w % mult) % mult, 0, (mult - h % mult) % mult), "reflect")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--start-h", type=int, default=2560)
    ap.add_argument("--stop-h", type=int, default=4096)
    ap.add_argument("--step-h", type=int, default=128, help="rows per step, 16:9 kept")
    ap.add_argument("--out", default=None)
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

    # Memory only, so benchmark stays off. With it on, cuDNN's algorithm search
    # workspaces land in the peak and the ceiling moves for the wrong reason.
    torch.backends.cudnn.benchmark = False

    gpu = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "default")
    print(f"[env] {gpu}, {total:.1f} GiB, allocator: {alloc_conf}", flush=True)

    model = RetinexDuelSambaFusionFinalization(
        in_channels=3, out_channels=3, L_n_feat=16, R_n_feat=16
    ).cuda().eval()
    sd = torch.load(weights, map_location="cpu")
    model.load_state_dict(sd.get("params", sd), strict=False)

    rows, last_ok, first_fail = [], None, None
    for h in range(args.start_h, args.stop_h + 1, args.step_h):
        w = int(round(h * 16 / 9 / 128)) * 128
        mpix = h * w / 1e6
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            x = check_image_size(torch.rand(1, 3, h, w, device="cuda"))
            with torch.inference_mode():
                o = model(x)
            while isinstance(o, (tuple, list)):
                o = o[0]
            torch.cuda.synchronize()
            rec = {"w": w, "h": h, "mpix": mpix, "oom": False,
                   "peak_alloc": torch.cuda.max_memory_allocated() / 2 ** 30,
                   "peak_reserved": torch.cuda.max_memory_reserved() / 2 ** 30}
            rec["overhead"] = rec["peak_reserved"] - rec["peak_alloc"]
            print(f"  {w}x{h}  {mpix:6.2f} Mpix  alloc {rec['peak_alloc']:6.2f}  "
                  f"reserved {rec['peak_reserved']:6.2f}  overhead {rec['overhead']:5.2f}  OK",
                  flush=True)
            last_ok = rec
            del o, x
        except torch.cuda.OutOfMemoryError:
            rec = {"w": w, "h": h, "mpix": mpix, "oom": True}
            print(f"  {w}x{h}  {mpix:6.2f} Mpix  OOM", flush=True)
            first_fail = rec
            rows.append(rec)
            torch.cuda.empty_cache()
            break
        rows.append(rec)

    print()
    if last_ok and first_fail:
        print(f"Ceiling on this card, allocator {alloc_conf}: between "
              f"{last_ok['mpix']:.2f} and {first_fail['mpix']:.2f} Mpix.")
        print(f"Largest input that runs: {last_ok['w']}x{last_ok['h']}, "
              f"{last_ok['peak_alloc']:.2f} GiB allocated, "
              f"{last_ok['peak_reserved']:.2f} GiB reserved.")
    elif last_ok:
        print(f"No OOM up to {last_ok['mpix']:.2f} Mpix. Raise --stop-h to find the ceiling.")
    else:
        print("OOM at the first resolution tried. Lower --start-h.")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"gpu": gpu, "total_gib": total, "allocator": alloc_conf,
                       "torch": torch.__version__, "rows": rows,
                       "last_ok_mpix": last_ok["mpix"] if last_ok else None,
                       "first_fail_mpix": first_fail["mpix"] if first_fail else None}, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

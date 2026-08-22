#!/usr/bin/env python3
"""Evaluate the other released tasks, to test whether the SSIM gap follows.

The paper reports four tasks and RetinexDual released a checkpoint for each. If
the SSIM shortfall seen on UHD-LL appears on the others too, it is a property of
how the paper computed SSIM rather than anything about that split, that
checkpoint or that particular number. That is a sharper test than any amount of
re-reading one result, so it is worth the GPU time.

Needs the other test sets, which are distributed separately. See README section 6.

Usage:
  python scripts/cross_task.py --repo ~/RetinexDual --data-root ~/data \\
      --ckpt-root ~/ckpts/retinexdual --out results/cross_task.json
"""
import argparse
import json
import os
import subprocess
import sys

# Published figures, from the comparison tables in arXiv:2508.04797.
PAPER = {
    "UHD_LL":   {"name": "UHD-LL",   "psnr": 28.79, "ssim": 0.934},
    "UHD_Blur": {"name": "UHD-Blur", "psnr": 30.71, "ssim": 0.886},
    "UHD_Haze": {"name": "UHD-Haze", "psnr": 26.63, "ssim": 0.956},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--ckpt-root", required=True)
    ap.add_argument("--tasks", nargs="+", default=["UHD_Blur", "UHD_Haze"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    rows = []
    for task in args.tasks:
        data = os.path.join(os.path.expanduser(args.data_root), task, "testing_set")
        ckpt = os.path.join(os.path.expanduser(args.ckpt_root), f"{task}.pth")
        for path in (data, ckpt):
            if not os.path.exists(path):
                raise SystemExit(f"[{task}] missing: {path}")

        out_dir = os.path.join("/tmp", f"cross_task_{task}")
        print(f"[run] {task}", flush=True)
        # One seed is enough: SSIM's run-to-run spread is 1.6e-5.
        subprocess.run([args.python, os.path.join(here, "evaluate.py"),
                        "--repo", os.path.expanduser(args.repo), "--data", data,
                        "--weights", ckpt, "--seeds", "0", "--out", out_dir],
                       check=True)
        s = json.load(open(os.path.join(out_dir, "reproduction_summary.json")))
        p = PAPER[task]
        rows.append({"task": p["name"], "n_images": s["n_images"],
                     "paper_psnr": p["psnr"], "paper_ssim": p["ssim"],
                     "repro_psnr": s["psnr_mean"], "repro_ssim": s["ssim_mean"],
                     "psnr_delta": s["psnr_mean"] - p["psnr"],
                     "ssim_delta": s["ssim_mean"] - p["ssim"]})

    print()
    print("| task | images | PSNR paper | PSNR here | difference | SSIM paper | SSIM here | difference |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['task']} | {r['n_images']} | {r['paper_psnr']} | {r['repro_psnr']:.4f} | "
              f"{r['psnr_delta']:+.4f} | {r['paper_ssim']} | {r['repro_ssim']:.5f} | "
              f"{r['ssim_delta']:+.5f} |")

    all_low = all(r["ssim_delta"] < 0 for r in rows)
    psnr_ok = all(abs(r["psnr_delta"]) < 0.05 for r in rows)
    print()
    if all_low and psnr_ok:
        print("Every task reproduces on PSNR and comes out low on SSIM, so the gap is a property")
        print("of how SSIM was computed rather than anything specific to one split or checkpoint.")
    elif not all_low:
        print("At least one task does NOT show the SSIM shortfall. The gap is not systematic and")
        print("SSIM_GAP.md needs revisiting.")
    else:
        print("PSNR does not reproduce on at least one task. Investigate that before reading the")
        print("SSIM column at all.")

    if args.out:
        json.dump({"note": "One seed per task. SSIM run-to-run spread is 1.6e-5.",
                   "tasks": rows}, open(args.out, "w"), indent=2)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

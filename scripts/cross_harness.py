#!/usr/bin/env python3
"""Check this repository's evaluation against the authors' own inference script.

Every other check here validates a metric implementation against a reference, or
a published figure against an artifact. None of them answers the deepest
objection to a reproduction: that the evaluation harness itself might be wrong in
a way that happens to land near the published number.

The answer is to run upstream's own `inference_RetinexDual.py` on a subset, run
`scripts/evaluate.py` on the same images, and compare what each reports. If they
agree, the harness is validated against the one the authors shipped.

Two differences are deliberate and do not affect a fully paired subset:

  - upstream counts unpaired inputs in its divisor (README section 3.2), which
    cannot fire when every image has a ground truth
  - upstream seeds once at import, this repository seeds per pass, so the two
    draw different routing. The comparison therefore has to be against the
    run-to-run spread rather than expecting equality.

Usage:
  python scripts/cross_harness.py --repo ~/RetinexDual --data <testing_set> -n 20
"""
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile

PER_IMAGE_SD = 0.029     # measured, README section 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", default=None)
    ap.add_argument("-n", "--n-images", type=int, default=20)
    ap.add_argument("--out", default=None)
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    repo = os.path.abspath(os.path.expanduser(args.repo))
    data = os.path.abspath(os.path.expanduser(args.data))
    weights = args.weights or os.path.join(repo, "pretrained_weights", "UHD_LL.pth")
    here = os.path.dirname(os.path.abspath(__file__))

    with tempfile.TemporaryDirectory() as td:
        sub_in, sub_gt = os.path.join(td, "input"), os.path.join(td, "gt")
        os.makedirs(sub_in), os.makedirs(sub_gt)
        names = sorted(n for n in os.listdir(os.path.join(data, "input"))
                       if os.path.exists(os.path.join(data, "gt", n)))[:args.n_images]
        for n in names:
            shutil.copy2(os.path.join(data, "input", n), sub_in)
            shutil.copy2(os.path.join(data, "gt", n), sub_gt)
        print(f"[subset] {len(names)} paired images", flush=True)

        print("[run] upstream inference_RetinexDual.py", flush=True)
        up = subprocess.run(
            [args.python, "inference_RetinexDual.py", "-i", sub_in, "-g", sub_gt,
             "-w", weights, "-o", os.path.join(td, "out")],
            cwd=repo, capture_output=True, text=True)
        text = up.stdout + up.stderr
        def grab(key):
            m = re.findall(rf"^{key}:([\d.]+)", text, re.M)
            return float(m[-1]) if m else None
        up_psnr, up_ssim = grab("avg_psnr"), grab("avg_ssim")
        if up_psnr is None:
            print(text[-1500:])
            raise SystemExit("[run] could not parse upstream output")
        print(f"       avg_psnr {up_psnr:.6f}  avg_ssim {up_ssim:.6f}", flush=True)

        print("[run] scripts/evaluate.py on the same images", flush=True)
        mine = subprocess.run(
            [args.python, os.path.join(here, "evaluate.py"), "--repo", repo,
             "--data", td, "--weights", weights, "--seeds", "0",
             "--out", os.path.join(td, "results")],
            capture_output=True, text=True)
        summary = os.path.join(td, "results", "reproduction_summary.json")
        if not os.path.exists(summary):
            print((mine.stdout + mine.stderr)[-1500:])
            raise SystemExit("[run] evaluate.py produced no summary")
        s = json.load(open(summary))
        me_psnr, me_ssim = s["psnr_mean"], s["ssim_mean"]
        print(f"       PSNR     {me_psnr:.6f}  SSIM     {me_ssim:.6f}", flush=True)

    n = len(names)
    sd_mean = PER_IMAGE_SD / math.sqrt(n)
    gap_sd = sd_mean * math.sqrt(2)
    d_psnr, d_ssim = abs(up_psnr - me_psnr), abs(up_ssim - me_ssim)
    sigma = d_psnr / gap_sd

    print()
    print(f"| | upstream | this repository | difference |")
    print(f"|---|---|---|---|")
    print(f"| PSNR | {up_psnr:.6f} | {me_psnr:.6f} | {d_psnr:.4f} dB |")
    print(f"| SSIM | {up_ssim:.6f} | {me_ssim:.6f} | {d_ssim:.6f} |")
    print()
    print(f"Routing is stochastic and the two scripts seed differently, so exact equality is not")
    print(f"expected. Per-image run-to-run spread is {PER_IMAGE_SD} dB, giving a {n}-image mean an")
    print(f"sd of {sd_mean:.4f} dB and a two-run gap an sd of {gap_sd:.4f} dB.")
    print(f"The observed PSNR gap is {sigma:.1f} sigma.")

    ok = sigma < 3.0 and d_ssim < 1e-3
    print()
    print("The two harnesses agree." if ok else
          "The two harnesses DISAGREE by more than routing noise explains. Investigate.")

    if args.out:
        json.dump({"n_images": n, "upstream_psnr": up_psnr, "upstream_ssim": up_ssim,
                   "repo_psnr": me_psnr, "repo_ssim": me_ssim,
                   "psnr_diff": d_psnr, "ssim_diff": d_ssim,
                   "psnr_gap_sigma": sigma, "agree": bool(ok)},
                  open(args.out, "w"), indent=2)
        print(f"wrote {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Test whether the SSIM gap could come from how per-image scores are aggregated.

Every other line of attack asks what SSIM *means*. This one asks how the 150
per-image numbers were combined. The distribution is left-skewed, a handful of
hard images drag the mean well below the median, and several ordinary
aggregation choices land close to the published 0.934.

The test that settles it applies each aggregation to BOTH metrics, dropping the
same images from each. PSNR reproduced at the full 150, so any aggregation that
lifts SSIM to 0.934 must leave PSNR near 28.79 to be viable. None do.

No GPU or dataset needed. Reads the committed per-image CSV.

Usage: python scripts/test_aggregation.py
"""
import argparse
import csv
import os
import statistics as st

PAPER_PSNR = 28.79
PAPER_SSIM = 0.934


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--tolerance-psnr", type=float, default=0.10,
                    help="how far PSNR may sit from the published value before an "
                         "aggregation is considered ruled out")
    args = ap.parse_args()

    path = os.path.join(args.results, "reproduction_per_image.csv")
    rows = list(csv.DictReader(open(path)))
    pairs = sorted((float(r["ssim_mean"]), float(r["psnr_mean_db"])) for r in rows)
    n = len(pairs)

    aggregations = [("all %d images" % n, pairs)]
    for t in (5, 10, 20):
        k = int(n * t / 100)
        aggregations.append((f"excluding worst {t}% by SSIM", pairs[k:]))
    for t in (10, 20):
        k = int(n * t / 100)
        aggregations.append((f"trimmed {t}% each end", pairs[k:n - k]))

    ssims = [p[0] for p in pairs]
    psnrs = [p[1] for p in pairs]
    print(f"Per-image distribution over {n} images")
    print(f"  SSIM  min {min(ssims):.5f}  median {st.median(ssims):.5f}  "
          f"mean {st.mean(ssims):.5f}  max {max(ssims):.5f}")
    print(f"  mean minus median = {st.mean(ssims) - st.median(ssims):+.5f}, "
          f"so the distribution is left-skewed\n")

    print(f"Each aggregation applied to BOTH metrics, same images dropped from each.")
    print(f"Paper reports PSNR {PAPER_PSNR} and SSIM {PAPER_SSIM}.\n")
    print(f"  {'aggregation':<30} {'SSIM':>9} {'dSSIM':>9} {'PSNR':>9} {'dPSNR':>9}   verdict")
    print(f"  {'-' * 30} {'-' * 9} {'-' * 9} {'-' * 9} {'-' * 9}   {'-' * 7}")

    survivors = []
    for name, sel in aggregations:
        s = st.mean(x[0] for x in sel)
        p = st.mean(x[1] for x in sel)
        ds, dp = s - PAPER_SSIM, p - PAPER_PSNR
        ssim_ok = abs(ds) < 0.002
        psnr_ok = abs(dp) < args.tolerance_psnr
        if ssim_ok and psnr_ok:
            verdict = "SURVIVES"
            survivors.append(name)
        elif ssim_ok:
            verdict = "ruled out, PSNR moves"
        elif psnr_ok:
            verdict = "PSNR fine, SSIM misses"
        else:
            verdict = "ruled out"
        print(f"  {name:<30} {s:>9.5f} {ds:>+9.5f} {p:>9.4f} {dp:>+9.4f}   {verdict}")

    print()
    if survivors:
        print("An aggregation reproduces both published numbers:")
        for s in survivors:
            print(f"  {s}")
        print("That would be an explanation for the gap. Investigate before believing it.")
        return 0

    print("No aggregation reproduces both published numbers.")
    print()
    print("This is worth spelling out, because one of them is very tempting. Excluding")
    print("the worst 10% of images by SSIM gives 0.93394 against the published 0.934, a")
    print("difference of 0.00006, which looks like a solved mystery. It is not. The same")
    print("exclusion lifts PSNR to 29.56, which is 0.77 dB above the published 28.79,")
    print("and PSNR reproduces exactly at the full 150. The images dragging SSIM down are")
    print("the same ones dragging PSNR down, so no exclusion can fix one without breaking")
    print("the other.")
    return 1


if __name__ == "__main__":
    raise SystemExit(0 if main() == 1 else 1)

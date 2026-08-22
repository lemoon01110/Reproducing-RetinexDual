#!/usr/bin/env python3
"""Render the reproduction figure from the committed CSVs.

Two panels:
  left   per-image PSNR, sorted, showing where the dataset mean sits and how
         wide the per-image spread is around it
  right  the five per-seed dataset means against the published value, which is
         the visual statement that run-to-run variation is far smaller than the
         gap to the paper

Usage: python scripts/make_figure.py --results results --out assets/reproduction.png
"""
import argparse
import csv
import json
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAPER_PSNR = 28.79

INK = "#1c1c1c"
ACCENT = "#0b6e99"
PAPER_C = "#c1440e"
GRID = "#d8d8d8"


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="assets/reproduction.png")
    ap.add_argument("--out-json", default=None,
                    help="record the values plotted, so a committed figure can be "
                         "checked against the committed data without re-rendering")
    args = ap.parse_args()

    per_image = read_csv(os.path.join(args.results, "reproduction_per_image.csv"))
    per_seed = read_csv(os.path.join(args.results, "reproduction_seeds.csv"))

    psnrs = sorted(float(r["psnr_mean_db"]) for r in per_image)
    seed_ids = [int(r["seed"]) for r in per_seed]
    seed_psnr = [float(r["psnr_db"]) for r in per_seed]
    mean_psnr = st.mean(seed_psnr)
    sd_psnr = st.stdev(seed_psnr) if len(seed_psnr) > 1 else 0.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})

    # Left panel: per-image PSNR, sorted
    ax1.plot(range(1, len(psnrs) + 1), psnrs, color=ACCENT, lw=1.8)
    ax1.axhline(mean_psnr, color=INK, lw=1.2, ls="--",
                label=f"reproduced mean {mean_psnr:.4f} dB")
    ax1.axhline(PAPER_PSNR, color=PAPER_C, lw=1.2, ls=":",
                label=f"reported {PAPER_PSNR} dB")
    ax1.set_xlabel(f"test image, sorted by PSNR (n = {len(psnrs)})")
    ax1.set_ylabel("PSNR (dB)")
    ax1.set_title("Per-image PSNR across the UHD-LL test set", fontsize=11, color=INK)
    ax1.legend(frameon=False, fontsize=9, loc="upper left")
    ax1.grid(True, color=GRID, lw=0.6)
    ax1.set_axisbelow(True)

    # Right panel: per-seed dataset means vs the published value
    ax2.axhline(PAPER_PSNR, color=PAPER_C, lw=1.4, ls=":", label=f"reported {PAPER_PSNR} dB")
    ax2.axhspan(mean_psnr - sd_psnr, mean_psnr + sd_psnr, color=ACCENT, alpha=0.15,
                label=f"+/- 1 sd ({sd_psnr:.4f} dB)")
    ax2.axhline(mean_psnr, color=INK, lw=1.2, ls="--")
    ax2.plot(seed_ids, seed_psnr, "o", color=ACCENT, ms=8, label="per-seed mean")

    span = max(abs(mean_psnr - PAPER_PSNR), sd_psnr * 3) * 1.6
    ax2.set_ylim(min(PAPER_PSNR, mean_psnr) - span * 0.4, max(PAPER_PSNR, mean_psnr) + span * 0.4)
    ax2.set_xticks(seed_ids)
    ax2.set_xlabel("seed")
    ax2.set_ylabel("dataset mean PSNR (dB)")
    ax2.set_title("Run-to-run spread vs the published value", fontsize=11, color=INK)
    ax2.legend(frameon=False, fontsize=9, loc="center right")
    ax2.grid(True, color=GRID, lw=0.6)
    ax2.set_axisbelow(True)

    for ax in (ax1, ax2):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight", facecolor="white")
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({"n_images": len(psnrs), "n_seeds": len(seed_psnr),
                       "psnr_mean": mean_psnr, "psnr_sd": sd_psnr,
                       "per_image_min": min(psnrs), "per_image_max": max(psnrs),
                       "seed_psnr": seed_psnr, "paper_psnr": PAPER_PSNR}, f, indent=2)
        print(f"wrote {args.out_json}")
    print(f"wrote {args.out}")
    print(f"  per-image n={len(psnrs)} range {min(psnrs):.3f} to {max(psnrs):.3f} dB")
    print(f"  per-seed mean {mean_psnr:.4f} +/- {sd_psnr:.4f} dB, reported {PAPER_PSNR}")


if __name__ == "__main__":
    main()

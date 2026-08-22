#!/usr/bin/env python3
"""Render a qualitative strip: input, reproduction output, ground truth.

A restoration reproduction that reports only scalars is hard to sanity check. A
PSNR of 28.8 dB means little without seeing what the model actually does, and
looking at the worst case is the fastest way to catch a reproduction that is
numerically plausible but visually broken.

Images are chosen from the committed per-image CSV by rank rather than by eye,
so the selection cannot be accused of flattering the result: the best, the
median and the worst scoring image in the test set.

Usage:
  python scripts/make_qualitative.py --repo ~/RetinexDual --data <testing_set>
"""
import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

CROP = 520          # side length of the detail crop, in source pixels
LABEL_H = 34
PANEL_GAP = 6
ROW_GAP = 10


def check_image_size(x, mult=128):
    _, _, h, w = x.size()
    return F.pad(x, (0, (mult - w % mult) % mult, 0, (mult - h % mult) % mult), "reflect")


def pick_images(csv_path):
    with open(csv_path) as f:
        rows = [r for r in csv.DictReader(f) if r["psnr_mean_db"]]
    rows.sort(key=lambda r: float(r["psnr_mean_db"]))
    worst, median, best = rows[0], rows[len(rows) // 2], rows[-1]
    return [("worst", worst), ("median", median), ("best", best)]


def busiest_crop(gt, size):
    """Pick the crop with the most gradient energy, so the panel shows detail
    rather than a flat patch of wall. Deterministic given the image."""
    g = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    energy = cv2.boxFilter(np.abs(gx) + np.abs(gy), -1, (size, size), normalize=True)
    h, w = g.shape
    half = size // 2
    inner = energy[half:h - half, half:w - half]
    idx = int(np.argmax(inner))
    cy, cx = idx // inner.shape[1] + half, idx % inner.shape[1] + half
    return max(0, cy - half), max(0, cx - half)


def label(panel, text):
    out = np.full((panel.shape[0] + LABEL_H, panel.shape[1], 3), 255, np.uint8)
    out[LABEL_H:] = panel
    cv2.putText(out, text, (6, LABEL_H - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (28, 28, 28), 1,
                cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="assets/qualitative.png")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", default=None,
                    help="record which images were selected and why, so the choice "
                         "is auditable without a GPU")
    args = ap.parse_args()

    repo = os.path.abspath(os.path.expanduser(args.repo))
    data = os.path.abspath(os.path.expanduser(args.data))
    weights = args.weights or os.path.join(repo, "pretrained_weights", "UHD_LL.pth")

    picks = pick_images(os.path.join(args.results, "reproduction_per_image.csv"))
    print(f"[pick] {[(t, r['image'], r['psnr_mean_db'][:7]) for t, r in picks]}", flush=True)

    sys.path.insert(0, repo)
    os.chdir(repo)
    import mamba_ssm.ops.selective_scan_interface as ssi
    if "site-packages" not in ssi.__file__:
        raise SystemExit(f"[env] local mamba_ssm shim detected at {ssi.__file__}")

    from basicsr.models.archs.RetinexDuelSambaFusionFinalization_arch import (
        RetinexDuelSambaFusionFinalization,
    )
    from basicsr.utils import img2tensor, tensor2img

    torch.backends.cudnn.benchmark = True
    model = RetinexDuelSambaFusionFinalization(
        in_channels=3, out_channels=3, L_n_feat=16, R_n_feat=16
    ).cuda().eval()
    sd = torch.load(weights, map_location="cpu")
    sd = sd["params"] if "params" in sd else sd
    model.load_state_dict(sd, strict=False)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    rows = []
    for tag, rec in picks:
        name, psnr = rec["image"], float(rec["psnr_mean_db"])
        img = cv2.imread(os.path.join(data, "input", name), cv2.IMREAD_UNCHANGED)
        gt = cv2.imread(os.path.join(data, "gt", name), cv2.IMREAD_UNCHANGED)
        if img is None or gt is None:
            raise SystemExit(f"could not read {name}")

        t = (img2tensor(img).cuda() / 255.0).unsqueeze(0)
        _, _, h, w = t.size()
        with torch.inference_mode():
            o = model(check_image_size(t))
        while isinstance(o, (tuple, list)):
            o = o[0]
        out = tensor2img(o[:, :, :h, :w])

        y, x = busiest_crop(gt, CROP)
        panels = [
            label(img[y:y + CROP, x:x + CROP], "input (low light)"),
            label(out[y:y + CROP, x:x + CROP], f"reproduced   {psnr:.2f} dB"),
            label(gt[y:y + CROP, x:x + CROP], "ground truth"),
        ]
        gap = np.full((panels[0].shape[0], PANEL_GAP, 3), 255, np.uint8)
        strip = np.hstack([panels[0], gap, panels[1], gap, panels[2]])

        head = np.full((26, strip.shape[1], 3), 255, np.uint8)
        cv2.putText(head, f"{tag}  ({name})", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (90, 90, 90), 1, cv2.LINE_AA)
        rows.append(np.vstack([head, strip]))
        print(f"[render] {tag} {name} {psnr:.2f} dB", flush=True)

    gap = np.full((ROW_GAP, rows[0].shape[1], 3), 255, np.uint8)
    canvas = rows[0]
    for r in rows[1:]:
        canvas = np.vstack([canvas, gap, r])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cv2.imwrite(args.out, canvas, [cv2.IMWRITE_PNG_COMPRESSION, 6])

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({
                "seed": args.seed, "crop": CROP,
                "selection": "by rank from reproduction_per_image.csv, not by eye",
                "crop_rule": "highest gradient energy on the ground truth",
                "picks": [{"rank": tag, "image": rec["image"],
                           "psnr_mean_db": float(rec["psnr_mean_db"])}
                          for tag, rec in picks],
            }, f, indent=2)
        print(f"wrote {args.out_json}")
    print(f"wrote {args.out}  {canvas.shape[1]}x{canvas.shape[0]}")
    print(f"Crops are {CROP}x{CROP} at native resolution, selected by gradient energy "
          f"on the ground truth, not by eye.")


if __name__ == "__main__":
    main()

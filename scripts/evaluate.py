#!/usr/bin/env python3
"""Reproduce RetinexDual's UHD-LL headline number.

Runs the released inference path, unmodified, over the full UHD-LL testing set
once per seed, and writes per-image and per-seed results.

Default inference is stochastic (see DETERMINISM.md), so a single run is not a
reproducible quantity. This is why we sweep seeds and report a spread.

Metric protocol matches the repository's own utils.calculate_psnr/ssim: RGB over
all three channels, range [0, 255], no border crop, computed per image and then
averaged. Two deliberate differences from inference_RetinexDual.py:

  1. Only images with a matching ground truth file are counted. The released
     script increments num_img in both branches but accumulates psnr_all only in
     the ground-truth branch, so an unpaired input deflates the reported average.
  2. Timing is not reported here. The released script's avg_inference_time has no
     torch.cuda.synchronize() around the forward pass.
"""
import argparse
import csv
import json
import os
import statistics
import sys
from collections import defaultdict

import cv2
import torch
import torch.nn.functional as F


def check_image_size(x, mult=128):
    """Reflect-pad to a multiple of `mult`, matching inference_RetinexDual.py."""
    _, _, h, w = x.size()
    return F.pad(x, (0, (mult - w % mult) % mult, 0, (mult - h % mult) % mult), "reflect")


def build_model(repo, weights, device):
    from basicsr.models.archs.RetinexDuelSambaFusionFinalization_arch import (
        RetinexDuelSambaFusionFinalization,
    )

    model = RetinexDuelSambaFusionFinalization(
        in_channels=3, out_channels=3, L_n_feat=16, R_n_feat=16
    ).to(device).eval()

    state = torch.load(weights, map_location="cpu")
    state = state["params"] if "params" in state else state

    # strict=False is what the released inference script uses. It hides 60 missing
    # keys, all SpectralGuidanceModule tensors, which are inert at inference.
    # See README section 4.1. We surface the count instead of swallowing it.
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if unexpected:
        raise SystemExit(f"[load] unexpected keys present, refusing to continue: {unexpected[:5]}")
    return model


def evaluate_one_pass(model, pairs, device, calculate_psnr, calculate_ssim,
                      img2tensor, tensor2img, skip_ssim=False):
    from math import isfinite

    out_rows = []
    for k, (name, in_path, gt_path) in enumerate(pairs):
        img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
        gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)

        t = (img2tensor(img).to(device) / 255.0).unsqueeze(0)
        _, _, h, w = t.size()
        t = check_image_size(t)

        with torch.inference_mode():
            out = model(t)
        while isinstance(out, (tuple, list)):
            out = out[0]

        out = out[:, :, :h, :w]
        out_img = tensor2img(out)

        psnr = calculate_psnr(out_img, gt)
        # SSIM at 3840x2160 runs on the CPU and dominates the wall clock, so it is
        # skippable for a fast PSNR-only check. Its run-to-run spread is 1e-5.
        ssim = float("nan") if skip_ssim else calculate_ssim(out_img, gt)
        if not isfinite(psnr) or (not skip_ssim and not isfinite(ssim)):
            raise SystemExit(f"[eval] non-finite metric on {name}")
        out_rows.append((name, psnr, ssim))

        if (k + 1) % 25 == 0:
            print(f"  {k + 1}/{len(pairs)} images", flush=True)
    return out_rows


def collect_pairs(data_root):
    in_dir = os.path.join(data_root, "input")
    gt_dir = os.path.join(data_root, "gt")
    for d in (in_dir, gt_dir):
        if not os.path.isdir(d):
            raise SystemExit(f"[data] missing directory: {d}\nRun scripts/check_data.py first.")

    pairs = []
    unpaired = []
    for name in sorted(os.listdir(in_dir)):
        gt_path = os.path.join(gt_dir, name)
        if os.path.exists(gt_path):
            pairs.append((name, os.path.join(in_dir, name), gt_path))
        else:
            unpaired.append(name)

    if unpaired:
        print(f"[data] WARNING {len(unpaired)} input images have no ground truth "
              f"and are excluded: {unpaired[:5]}", flush=True)
    if not pairs:
        raise SystemExit("[data] no paired images found")
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="path to the upstream RetinexDual checkout")
    ap.add_argument("--data", required=True, help="path to UHD-LL testing_set (contains input/ gt/)")
    ap.add_argument("--weights", default=None, help="defaults to <repo>/pretrained_weights/UHD_LL.pth")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default="results")
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the first N images, for a quick plumbing check")
    ap.add_argument("--skip-ssim", action="store_true",
                    help="PSNR only. Roughly 4x faster, since SSIM at 4K is CPU-bound")
    args = ap.parse_args()

    repo = os.path.abspath(os.path.expanduser(args.repo))
    data = os.path.abspath(os.path.expanduser(args.data))
    weights = args.weights or os.path.join(repo, "pretrained_weights", "UHD_LL.pth")

    # Guard against a local mamba_ssm directory shadowing the real CUDA kernels.
    # Python puts the working directory ahead of site-packages, so a stray package
    # silently replaces the kernels and every number afterwards is meaningless.
    sys.path.insert(0, repo)
    os.chdir(repo)
    import mamba_ssm.ops.selective_scan_interface as ssi
    if "site-packages" not in ssi.__file__:
        raise SystemExit(f"[env] mamba_ssm resolves to {ssi.__file__}, which is not an "
                         f"installed package. A local shim is shadowing the CUDA kernels.")
    print(f"[env] mamba_ssm: {ssi.__file__}", flush=True)

    from basicsr.utils import img2tensor, tensor2img
    from utils import calculate_psnr, calculate_ssim

    if not torch.cuda.is_available():
        raise SystemExit("[env] CUDA is required")
    device = "cuda"
    torch.backends.cudnn.benchmark = True
    print(f"[env] torch {torch.__version__}, {torch.cuda.get_device_name(0)}", flush=True)

    pairs = collect_pairs(data)
    if args.limit:
        pairs = pairs[:args.limit]
        print(f"[data] LIMITED to the first {len(pairs)} images. This is a plumbing check, "
              f"not a reproduction of the reported number.", flush=True)
    if args.skip_ssim:
        print("[data] SSIM disabled. The SSIM columns will be empty.", flush=True)
    print(f"[data] {len(pairs)} paired images", flush=True)

    model = build_model(repo, weights, device)

    os.makedirs(args.out, exist_ok=True)
    per_image = defaultdict(list)
    seed_rows = []

    for seed in args.seeds:
        print(f"[run] seed {seed}", flush=True)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        rows = evaluate_one_pass(model, pairs, device, calculate_psnr, calculate_ssim,
                                 img2tensor, tensor2img, skip_ssim=args.skip_ssim)
        psnr = statistics.mean(r[1] for r in rows)
        ssim = float("nan") if args.skip_ssim else statistics.mean(r[2] for r in rows)
        seed_rows.append((seed, len(rows), psnr, ssim))
        for name, p, s in rows:
            per_image[name].append((p, s))
        ssim_txt = "skipped" if args.skip_ssim else f"{ssim:.5f}"
        print(f"[run] seed {seed}: PSNR {psnr:.4f}  SSIM {ssim_txt}", flush=True)

    with open(os.path.join(args.out, "reproduction_seeds.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "n_images", "psnr_db", "ssim"])
        for seed, n, p, s in seed_rows:
            w.writerow([seed, n, f"{p:.6f}", "" if args.skip_ssim else f"{s:.6f}"])

    with open(os.path.join(args.out, "reproduction_per_image.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "n_seeds", "psnr_mean_db", "psnr_sd_db", "ssim_mean"])
        for name in sorted(per_image):
            vals = per_image[name]
            ps = [v[0] for v in vals]
            w.writerow([name, len(vals), f"{statistics.mean(ps):.6f}",
                        f"{statistics.stdev(ps):.6f}" if len(ps) > 1 else "",
                        "" if args.skip_ssim else f"{statistics.mean(v[1] for v in vals):.6f}"])

    psnrs = [r[2] for r in seed_rows]
    ssims = [r[3] for r in seed_rows]
    sd = statistics.stdev(psnrs) if len(psnrs) > 1 else 0.0

    print("\n" + "=" * 62)
    print("  RetinexDual UHD-LL reproduction")
    print("=" * 62)
    print(f"  seeds            {len(seed_rows)}   images {seed_rows[0][1]}")
    print(f"  PSNR reported    28.79")
    print(f"  PSNR reproduced  {statistics.mean(psnrs):.4f} +/- {sd:.4f}"
          f"   (range {min(psnrs):.4f} to {max(psnrs):.4f})")
    print(f"  difference       {statistics.mean(psnrs) - 28.79:+.4f} dB")
    print(f"  SSIM reported    0.934")
    if args.skip_ssim:
        print(f"  SSIM reproduced  skipped (--skip-ssim)")
    else:
        print(f"  SSIM reproduced  {statistics.mean(ssims):.5f}")
        print(f"  difference       {statistics.mean(ssims) - 0.934:+.4f}")
    if args.limit:
        print(f"  NOTE             limited to {seed_rows[0][1]} images, not a reproduction")
    print("=" * 62)

    with open(os.path.join(args.out, "reproduction_summary.json"), "w") as f:
        json.dump({
            "n_seeds": len(seed_rows), "n_images": seed_rows[0][1],
            "psnr_reported": 28.79, "psnr_mean": statistics.mean(psnrs), "psnr_sd": sd,
            "psnr_min": min(psnrs), "psnr_max": max(psnrs),
            "ssim_reported": 0.934,
            "ssim_mean": None if args.skip_ssim else statistics.mean(ssims),
            "limited": bool(args.limit), "ssim_skipped": bool(args.skip_ssim),
            "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
        }, f, indent=2)


if __name__ == "__main__":
    main()

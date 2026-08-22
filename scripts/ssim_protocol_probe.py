#!/usr/bin/env python3
"""Try to account for the SSIM gap between this reproduction and the paper.

Reproduced SSIM is 0.92217 under the repository's own helper. The paper reports
0.934. PSNR matches to 0.030 dB under the same outputs, so the model is almost
certainly fine and the difference is in how SSIM is defined.

"SSIM" names a family, not one number. This runs inference once per image and
scores the same output under several conventions that papers in this area
actually use, to see whether any of them lands on 0.934.

This is a diagnostic, not a claim. If one protocol matches, that is evidence
about the metric, not proof of what the authors ran.

Usage:
  python scripts/ssim_protocol_probe.py --repo ~/RetinexDual --data <testing_set>
"""
import argparse
import os
import statistics as st
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def check_image_size(x, mult=128):
    _, _, h, w = x.size()
    return F.pad(x, (0, (mult - w % mult) % mult, 0, (mult - h % mult) % mult), "reflect")


def bgr2y(img):
    """BT.601 luma in the 16-235 studio range, which is what 'Y channel' means
    in most restoration papers (and what BasicSR's bgr2ycbcr produces)."""
    img = img.astype(np.float64) / 255.0
    y = np.dot(img, [24.966, 128.553, 65.481]) + 16.0  # BGR order
    return y


def ssim_matlab(img1, img2):
    """The repository's own single-channel SSIM: 11x11 Gaussian, sigma 1.5."""
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    img1, img2 = img1.astype(np.float64), img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    m = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
        ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return m.mean()


def crop(img, b):
    return img if b == 0 else img[b:-b, b:-b, ...]


def protocols(out, gt, fast=False):
    """out, gt are uint8 BGR at full resolution."""
    from skimage.metrics import structural_similarity as sk_ssim
    r = {}

    # What this repository reports: per-channel MATLAB SSIM over BGR, averaged.
    r["repo_rgb_mean"] = float(np.mean([ssim_matlab(out[:, :, i], gt[:, :, i]) for i in range(3)]))

    # Same kernel, luma only. The most common alternative convention.
    r["matlab_y"] = float(ssim_matlab(bgr2y(out), bgr2y(gt)))

    # Same as above but with the 4-pixel border crop several benchmarks apply.
    r["matlab_y_border4"] = float(ssim_matlab(bgr2y(crop(out, 4)), bgr2y(crop(gt, 4))))

    if fast:
        # The scikit-image variants below agreed with the two above to 5 decimals
        # on a 40-image pilot, so they are redundant on a long run. Kept for the
        # default path because agreement across two implementations is the check
        # that the MATLAB-style kernel here is implemented correctly.
        return r

    # scikit-image defaults: 7x7 uniform window, sample covariance.
    r["skimage_rgb_default"] = float(sk_ssim(out, gt, channel_axis=2, data_range=255))

    # scikit-image configured to match the MATLAB kernel.
    r["skimage_rgb_gaussian"] = float(sk_ssim(out, gt, channel_axis=2, data_range=255,
                                              gaussian_weights=True, sigma=1.5,
                                              use_sample_covariance=False))
    r["skimage_y_gaussian"] = float(sk_ssim(bgr2y(out), bgr2y(gt), data_range=255,
                                            gaussian_weights=True, sigma=1.5,
                                            use_sample_covariance=False))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="score only the first N images. Comparison to the published "
                         "full-set number is then invalid, and the script says so")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target", type=float, default=0.934)
    ap.add_argument("--full-n", type=int, default=150,
                    help="size of the full test set, used to detect a partial run")
    ap.add_argument("--fast", action="store_true",
                    help="only the three protocols that differ meaningfully, roughly 3x faster")
    args = ap.parse_args()

    repo = os.path.abspath(os.path.expanduser(args.repo))
    data = os.path.abspath(os.path.expanduser(args.data))
    weights = args.weights or os.path.join(repo, "pretrained_weights", "UHD_LL.pth")
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

    in_dir, gt_dir = os.path.join(data, "input"), os.path.join(data, "gt")
    names = [n for n in sorted(os.listdir(in_dir)) if os.path.exists(os.path.join(gt_dir, n))]
    if args.limit:
        names = names[:args.limit]
    print(f"[probe] {len(names)} images, seed {args.seed}", flush=True)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    acc = {}
    for k, name in enumerate(names):
        img = cv2.imread(os.path.join(in_dir, name), cv2.IMREAD_UNCHANGED)
        gt = cv2.imread(os.path.join(gt_dir, name), cv2.IMREAD_UNCHANGED)
        t = (img2tensor(img).cuda() / 255.0).unsqueeze(0)
        _, _, h, w = t.size()
        t = check_image_size(t)
        with torch.inference_mode():
            o = model(t)
        while isinstance(o, (tuple, list)):
            o = o[0]
        out = tensor2img(o[:, :, :h, :w])

        for key, val in protocols(out, gt, fast=args.fast).items():
            acc.setdefault(key, []).append(val)
        if (k + 1) % 10 == 0:
            print(f"  {k + 1}/{len(names)}", flush=True)

    base = st.mean(acc["repo_rgb_mean"])
    partial = len(names) < args.full_n

    print(f"\nSSIM under different conventions, same model outputs, n = {len(names)}")
    print(f"Paper reports {args.target}\n")

    if partial:
        print(f"WARNING: this is {len(names)} of {args.full_n} images. The published {args.target}")
        print("is a full-test-set mean, so the 'vs paper' column below is not a valid comparison:")
        print("a subset has its own baseline offset, which can be as large as the gap being")
        print("investigated. The 'vs repo_rgb_mean' column IS valid, because it is measured on")
        print("these same images. Use it, and re-run on the full set before concluding.\n")

    print("| protocol | SSIM | vs repo_rgb_mean | vs paper |")
    print("|---|---|---|---|")
    for key, vals in sorted(acc.items(), key=lambda kv: -st.mean(kv[1])):
        m = st.mean(vals)
        vs_paper = "n/a (subset)" if partial else f"{m - args.target:+.5f}"
        print(f"| `{key}` | {m:.5f} | {m - base:+.5f} | {vs_paper} |")

    print()
    if partial:
        print("Offsets relative to the repository's own helper are the transferable quantity.")
        print("Adding them to the full-set repo_rgb_mean estimates what each convention would")
        print("give over all 150 images. Verify by re-running without --limit.")
        return

    ranked = sorted(acc.items(), key=lambda kv: abs(st.mean(kv[1]) - args.target))
    best_key, best_vals = ranked[0]
    best = st.mean(best_vals)
    if abs(best - args.target) < 0.002:
        print(f"Closest: `{best_key}` at {best:.5f}, within {abs(best - args.target):.5f} of "
              f"{args.target}.")
        print("Close enough to explain the gap as a metric convention rather than a difference")
        print("in the model outputs. Still a hypothesis, not a confirmation of what the authors ran.")
    else:
        lo = min(st.mean(v) for v in acc.values())
        hi = max(st.mean(v) for v in acc.values())
        print(f"Closest: `{best_key}` at {best:.5f}, still {abs(best - args.target):.5f} away "
              f"from {args.target}.")
        if lo < args.target < hi:
            print(f"Note that {args.target} falls BETWEEN the conventions tested "
                  f"({lo:.5f} to {hi:.5f}), so it is not simply that one of them was used.")
        print("No standard convention tested accounts for the gap. It stays unexplained, and")
        print("the report should continue to say so.")


if __name__ == "__main__":
    main()

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
import json
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


def bgr2y_float(img):
    """Same luma transform as bgr2y but for an already-float [0, 255] array."""
    return np.dot(img / 255.0, [24.966, 128.553, 65.481]) + 16.0


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


def ssim_err_cly(img1, img2):
    """SSIM exactly as ERR computes it, which is the benchmark RetinexDual's
    upstream README credits for "the UHD restoration benchmarks and references".

    Two differences from the RetinexDual repository's own helper, both material:

      1. cv2.filter2D runs with borderType=BORDER_REPLICATE and the result is
         NOT cropped to [5:-5, 5:-5]. The repo's version crops, discarding the
         border. Averaging the full map instead includes edge regions where the
         replicated border makes both images agree closely, which raises SSIM.
      2. It is applied to the Y channel after a 1-pixel border crop.

    Source: NJU-PCALab/ERR, comput_psnr_ssim.py, _ssim_cly and calculate_ssim,
    whose defaults are crop_border=1 and test_y_channel=True.
    """
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    img1, img2 = img1.astype(np.float64), img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    bt = cv2.BORDER_REPLICATE

    mu1 = cv2.filter2D(img1, -1, window, borderType=bt)
    mu2 = cv2.filter2D(img2, -1, window, borderType=bt)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window, borderType=bt) - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window, borderType=bt) - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window, borderType=bt) - mu1_mu2
    m = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
        ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return m.mean()


def bgr2ycbcr_full(img):
    """Full YCbCr, BT.601 studio range, from BGR uint8. Matches BasicSR."""
    img = img.astype(np.float64) / 255.0
    b, g, r = img[..., 0], img[..., 1], img[..., 2]
    y = 65.481 * r + 128.553 * g + 24.966 * b + 16.0
    cb = -37.797 * r - 74.203 * g + 112.0 * b + 128.0
    cr = 112.0 * r - 93.786 * g - 18.214 * b + 128.0
    return np.stack([y, cb, cr], axis=-1)


def ssim_torch01(out, gt):
    """SSIM as ERR's basicsr/models/cal_ssim.py computes it.

    That file is a separate implementation from the one in comput_psnr_ssim.py,
    and differs in two ways that both matter: it scores tensors in [0, 1] with
    C1 = 0.01^2 and C2 = 0.03^2 rather than the [0, 255] constants, and it
    convolves with padding = window // 2 rather than cropping to valid.
    Zero padding at the border drags both means toward zero, which behaves
    differently from replicate padding or from cropping.
    """
    import torch
    import torch.nn.functional as Fn

    k = cv2.getGaussianKernel(11, 1.5)
    w = torch.from_numpy(np.outer(k, k.T)).float()
    w = w.expand(3, 1, 11, 11).contiguous()

    a = torch.from_numpy(out.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    b = torch.from_numpy(gt.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)

    mu1 = Fn.conv2d(a, w, padding=5, groups=3)
    mu2 = Fn.conv2d(b, w, padding=5, groups=3)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    s1 = Fn.conv2d(a * a, w, padding=5, groups=3) - mu1_sq
    s2 = Fn.conv2d(b * b, w, padding=5, groups=3) - mu2_sq
    s12 = Fn.conv2d(a * b, w, padding=5, groups=3) - mu1_mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu1_mu2 + C1) * (2 * s12 + C2)) / ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))
    return float(m.mean())


def crop(img, b):
    return img if b == 0 else img[b:-b, b:-b, ...]


def protocols(out, gt, fast=False, out_float=None):
    """out, gt are uint8 BGR at full resolution.

    out_float, when given, is the model output before tensor2img quantised it to
    uint8, on the same [0, 255] scale. Some codebases score in float space, which
    removes the quantisation noise and raises SSIM slightly.
    """
    from skimage.metrics import structural_similarity as sk_ssim
    r = {}

    # What this repository reports: per-channel MATLAB SSIM over BGR, averaged.
    r["repo_rgb_mean"] = float(np.mean([ssim_matlab(out[:, :, i], gt[:, :, i]) for i in range(3)]))

    # Same kernel, luma only. The most common alternative convention.
    r["matlab_y"] = float(ssim_matlab(bgr2y(out), bgr2y(gt)))

    # Same as above but with the 4-pixel border crop several benchmarks apply.
    r["matlab_y_border4"] = float(ssim_matlab(bgr2y(crop(out, 4)), bgr2y(crop(gt, 4))))

    # ERR's protocol, at its own defaults. This is the strongest candidate for
    # what the published table used, since RetinexDual credits ERR for the
    # benchmark and its numbers sit in a table alongside ERR's.
    r["err_y_cly"] = float(ssim_err_cly(bgr2y(crop(out, 1)), bgr2y(crop(gt, 1))))

    # Per-channel over full YCbCr rather than luma only. Chroma channels score
    # lower than luma, so averaging all three lands between the RGB and Y
    # figures, which is where the published value sits.
    o_ycc, g_ycc = bgr2ycbcr_full(out), bgr2ycbcr_full(gt)
    r["ycbcr_mean"] = float(np.mean(
        [ssim_matlab(o_ycc[:, :, i], g_ycc[:, :, i]) for i in range(3)]))

    # ERR's other SSIM implementation: [0,1] constants, zero padding, per channel.
    r["torch_rgb_01"] = ssim_torch01(out, gt)

    # Scored before uint8 quantisation. Tests whether the published figure could
    # come from evaluating in float space rather than on saved images.
    if out_float is not None:
        gt_f = gt.astype(np.float64)
        r["matlab_rgb_float"] = float(np.mean(
            [ssim_matlab(out_float[:, :, i], gt_f[:, :, i]) for i in range(3)]))
        r["matlab_y_float"] = float(ssim_matlab(bgr2y_float(out_float), bgr2y_float(gt_f)))

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
    ap.add_argument("--out", default=None,
                    help="write a JSON artifact so the published table has a "
                         "machine-checkable source of truth")
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
        cropped = o[:, :, :h, :w]
        out = tensor2img(cropped)
        # tensor2img clamps to [0,1] then scales to [0,255] and rounds. Keep the
        # pre-rounding values on the same scale for the float protocols.
        out_float = (cropped.squeeze(0).permute(1, 2, 0).clamp(0, 1)
                     .float().cpu().numpy()[:, :, ::-1] * 255.0).astype(np.float64)

        # Validate the float path before trusting anything computed from it.
        # tensor2img rounds and reorders channels, so if out_float were built
        # wrongly (wrong channel order being the likely error, since img2tensor
        # converts BGR to RGB on the way in) the float protocols would silently
        # score a mismatched pair. Rounding out_float must recover out.
        if k == 0:
            delta = np.abs(np.rint(out_float) - out.astype(np.float64)).max()
            print(f"[check] float path vs tensor2img: max |delta| = {delta:.1f} "
                  f"(must be <= 1, larger means the channel order is wrong)", flush=True)
            if delta > 1:
                raise SystemExit("[check] float reconstruction does not match tensor2img, "
                                 "refusing to report float protocols")

        for key, val in protocols(out, gt, fast=args.fast, out_float=out_float).items():
            acc.setdefault(key, []).append(val)
        if (k + 1) % 10 == 0:
            print(f"  {k + 1}/{len(names)}", flush=True)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"n_images": len(names), "seed": args.seed, "target": args.target,
                       "fast": bool(args.fast),
                       "protocols": {k: st.mean(v) for k, v in acc.items()}},
                      f, indent=2)
        print(f"[out] wrote {args.out}", flush=True)

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

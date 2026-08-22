#!/usr/bin/env python3
"""Check the metric implementations this report's conclusions depend on.

SSIM_GAP.md argues that nine conventions cluster into two groups and that the
published 0.934 sits between them. That argument is only as good as the
implementations behind it. If `ssim_matlab` were subtly wrong, the whole
clustering result would be an artifact of my code rather than a fact about SSIM.

These tests pin each implementation against an independent reference or against
a case whose answer is known analytically. No GPU, no dataset, so CI runs them.

Usage: python scripts/test_metrics.py
"""
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from ssim_protocol_probe import (  # noqa: E402
    bgr2y, bgr2y_float, bgr2ycbcr_full, crop, ssim_err_cly, ssim_matlab, ssim_torch01,
)

FAILED = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{'   ' + detail if detail else ''}")
    if not ok:
        FAILED.append(name)


def rand_pair(h=96, w=128, seed=0, noise=12.0):
    """A reference image and a noisy version, uint8 BGR."""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 256, (h, w, 3)).astype(np.uint8)
    # Smooth it, so the pair has real local structure rather than pure noise.
    import cv2
    a = cv2.GaussianBlur(a, (0, 0), 3)
    b = np.clip(a.astype(np.float64) + rng.normal(0, noise, a.shape), 0, 255).astype(np.uint8)
    return a, b


def main():
    import cv2
    from skimage.metrics import structural_similarity as sk_ssim

    print("Identity cases, where the answer is known analytically")
    a, _ = rand_pair(seed=1)
    ay = bgr2y(a)
    check("ssim_matlab(x, x) == 1", abs(ssim_matlab(ay, ay) - 1.0) < 1e-9,
          f"got {ssim_matlab(ay, ay):.12f}")
    check("ssim_err_cly(x, x) == 1", abs(ssim_err_cly(ay, ay) - 1.0) < 1e-9,
          f"got {ssim_err_cly(ay, ay):.12f}")
    check("ssim_torch01(x, x) == 1", abs(ssim_torch01(a, a) - 1.0) < 1e-5,
          f"got {ssim_torch01(a, a):.9f}")

    print("\nssim_matlab against scikit-image with a matched kernel")
    # skimage with gaussian_weights, sigma 1.5 and population covariance is the
    # same estimator as the MATLAB-style implementation, so they must agree.
    for seed in (0, 2, 7):
        a, b = rand_pair(seed=seed)
        ay, by = bgr2y(a), bgr2y(b)
        mine = ssim_matlab(ay, by)
        theirs = sk_ssim(ay, by, data_range=255.0, gaussian_weights=True, sigma=1.5,
                         use_sample_covariance=False)
        check(f"seed {seed}", abs(mine - theirs) < 2e-3,
              f"mine {mine:.6f} skimage {theirs:.6f} delta {abs(mine - theirs):.2e}")

    print("\nbgr2y against the BT.601 definition BasicSR uses")
    # Pure blue, green, red in BGR order, so each coefficient is isolated.
    for name, px, expect in (("blue", (255, 0, 0), 16 + 24.966),
                             ("green", (0, 255, 0), 16 + 128.553),
                             ("red", (0, 0, 255), 16 + 65.481)):
        img = np.zeros((4, 4, 3), np.uint8)
        img[:, :] = px
        got = bgr2y(img)[0, 0]
        check(f"{name} -> {expect:.3f}", abs(got - expect) < 1e-6, f"got {got:.6f}")

    print("\nbgr2y_float agrees with bgr2y on integral input")
    a, _ = rand_pair(seed=3)
    d = np.abs(bgr2y(a) - bgr2y_float(a.astype(np.float64))).max()
    check("max delta below 1e-9", d < 1e-9, f"got {d:.2e}")

    print("\nbgr2ycbcr_full puts luma in channel 0 matching bgr2y")
    a, _ = rand_pair(seed=4)
    d = np.abs(bgr2ycbcr_full(a)[:, :, 0] - bgr2y(a)).max()
    check("max delta below 1e-9", d < 1e-9, f"got {d:.2e}")
    # Neutral grey must give chroma exactly 128.
    grey = np.full((4, 4, 3), 128, np.uint8)
    cb, cr = bgr2ycbcr_full(grey)[0, 0, 1], bgr2ycbcr_full(grey)[0, 0, 2]
    check("grey has Cb = Cr = 128", abs(cb - 128) < 1e-6 and abs(cr - 128) < 1e-6,
          f"got Cb {cb:.6f} Cr {cr:.6f}")

    print("\ncrop removes the right number of pixels")
    a, _ = rand_pair(h=50, w=60, seed=5)
    check("crop(x, 0) is a no-op", crop(a, 0).shape == a.shape)
    check("crop(x, 4) removes 8 rows and columns", crop(a, 4).shape == (42, 52, 3),
          f"got {crop(a, 4).shape}")

    print("\nOrdering the argument depends on: luma scores above per-channel RGB")
    # The report's central claim is that these form two separated clusters. If
    # that ordering ever inverted, the clustering discussion would be wrong.
    a, b = rand_pair(seed=6, noise=20.0)
    rgb = float(np.mean([ssim_matlab(a[:, :, i], b[:, :, i]) for i in range(3)]))
    luma = float(ssim_matlab(bgr2y(a), bgr2y(b)))
    check("luma > rgb on a noisy pair", luma > rgb, f"luma {luma:.5f} rgb {rgb:.5f}")

    print()
    if FAILED:
        print(f"{len(FAILED)} check(s) failed: {', '.join(FAILED)}")
        print("The conclusions in SSIM_GAP.md depend on these, so fix before trusting it.")
        return 1
    print("All metric implementations agree with their references.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

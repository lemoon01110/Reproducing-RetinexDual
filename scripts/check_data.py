#!/usr/bin/env python3
"""Verify the UHD-LL testing set before spending GPU time on it.

The dataset is distributed through Google Drive, which rate-limits folder
downloads of this size, so partial and silently truncated downloads are the
normal failure mode. This checks pairing, dimensions and decodability up front.
"""
import argparse
import os
import sys
from collections import Counter

import cv2

EXPECTED_PAIRS = 150
EXPECTED_SIZE = (2160, 3840)  # height, width


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to UHD-LL testing_set")
    ap.add_argument("--quick", action="store_true", help="skip decoding every image")
    args = ap.parse_args()

    root = os.path.abspath(os.path.expanduser(args.data))
    in_dir, gt_dir = os.path.join(root, "input"), os.path.join(root, "gt")

    print(f"[data] root {root}")
    missing_dirs = [d for d in (in_dir, gt_dir) if not os.path.isdir(d)]
    if missing_dirs:
        print("\nExpected layout:\n")
        print("  <testing_set>/")
        print("  |-- input/    150 low-light images, 3840x2160")
        print("  `-- gt/       150 ground truth images, same filenames\n")
        print("Source: https://drive.google.com/drive/folders/1IneTwBsSiSSVXGoXQ9_hE1cO2d4Fd4DN")
        print("Download testing_set/input and testing_set/gt.\n")
        for d in missing_dirs:
            print(f"  MISSING  {d}")
        return 1

    ins = sorted(os.listdir(in_dir))
    gts = set(os.listdir(gt_dir))
    paired = [n for n in ins if n in gts]
    unpaired = [n for n in ins if n not in gts]
    gt_only = sorted(gts - set(ins))

    print(f"[data] input {len(ins)}   gt {len(gts)}   paired {len(paired)}")
    if unpaired:
        print(f"[data] {len(unpaired)} input images with no ground truth: {unpaired[:5]}")
    if gt_only:
        print(f"[data] {len(gt_only)} ground truth images with no input: {gt_only[:5]}")

    ok = True
    if len(paired) != EXPECTED_PAIRS:
        print(f"[data] FAIL expected {EXPECTED_PAIRS} pairs, found {len(paired)}")
        ok = False

    if not args.quick:
        sizes, corrupt = Counter(), []
        for k, name in enumerate(paired):
            for d in (in_dir, gt_dir):
                img = cv2.imread(os.path.join(d, name), cv2.IMREAD_UNCHANGED)
                if img is None:
                    corrupt.append(os.path.join(d, name))
                else:
                    sizes[img.shape[:2]] += 1
            if (k + 1) % 50 == 0:
                print(f"  checked {k + 1}/{len(paired)}", flush=True)

        print(f"[data] dimensions: {dict(sizes)}")
        if corrupt:
            print(f"[data] FAIL {len(corrupt)} unreadable files: {corrupt[:5]}")
            ok = False
        if set(sizes) != {EXPECTED_SIZE}:
            print(f"[data] NOTE not all images are {EXPECTED_SIZE[1]}x{EXPECTED_SIZE[0]}. "
                  f"The reported number is for the standard split.")

    print("[data] OK" if ok else "[data] FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

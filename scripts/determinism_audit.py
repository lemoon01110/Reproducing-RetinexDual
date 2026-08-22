#!/usr/bin/env python3
"""Reproduce the determinism table in DETERMINISM.md.

Three experiments on the same input, five forward passes each, runs 2 to 5
compared against run 1:

  A  natural inference, exactly as released
  B  identical but with CPU and CUDA RNG state restored before each forward
  C  identical but with gumbel_softmax replaced by a deterministic argmax

A differing while B is bit-identical is the result that matters. It isolates the
variation to random sampling in the routing path and rules out nondeterministic
kernels, which would still have perturbed B.

Usage:
  python scripts/determinism_audit.py --repo ~/RetinexDual --image path/to/one.png
  python scripts/determinism_audit.py --repo ~/RetinexDual --synthetic
"""
import argparse
import math
import os
import sys

import torch
import torch.nn.functional as F


def check_image_size(x, mult=128):
    _, _, h, w = x.size()
    return F.pad(x, (0, (mult - w % mult) % mult, 0, (mult - h % mult) % mult), "reflect")


def deterministic_gumbel(logits, tau=1.0, hard=False, eps=1e-10, dim=-1):
    """Drop-in for F.gumbel_softmax that takes the argmax instead of sampling.

    Returns the same one-hot shape the hard=True path returns, so the downstream
    matmul into the prompt and the sort key both stay well defined.
    """
    idx = logits.argmax(dim=dim, keepdim=True)
    return torch.zeros_like(logits).scatter_(dim, idx, 1.0)


def forward_once(model, x):
    with torch.inference_mode():
        out = model(x)
    while isinstance(out, (tuple, list)):
        out = out[0]
    return out.detach().float().clone()


def compare(ref, other):
    """Deltas in [0, 1] image scale, plus PSNR treating ref as the signal."""
    d = (ref - other).abs()
    mx, mean = d.max().item(), d.mean().item()
    mse = ((ref - other) ** 2).mean().item()
    psnr = float("inf") if mse == 0 else 10 * math.log10(1.0 / mse)
    return mx, mean, psnr


def run_experiment(model, x, mode, n=5):
    outs = []
    orig = F.gumbel_softmax

    if mode == "argmax":
        F.gumbel_softmax = deterministic_gumbel
    try:
        if mode == "replay":
            cpu_state = torch.get_rng_state()
            cuda_state = torch.cuda.get_rng_state_all()
        for _ in range(n):
            if mode == "replay":
                torch.set_rng_state(cpu_state)
                torch.cuda.set_rng_state_all(cuda_state)
            outs.append(forward_once(model, x))
    finally:
        F.gumbel_softmax = orig

    rows = [compare(outs[0], o) for o in outs[1:]]
    identical = all(r[0] == 0.0 for r in rows)
    return {
        "identical": identical,
        "max_lo": min(r[0] for r in rows),
        "max_hi": max(r[0] for r in rows),
        "mean": sum(r[1] for r in rows) / len(rows),
        "psnr": min(r[2] for r in rows),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--image", default=None, help="one input image, ideally a real 4K low-light frame")
    ap.add_argument("--synthetic", action="store_true", help="use a random 3840x2160 input instead")
    ap.add_argument("--passes", type=int, default=5)
    args = ap.parse_args()

    repo = os.path.abspath(os.path.expanduser(args.repo))
    weights = args.weights or os.path.join(repo, "pretrained_weights", "UHD_LL.pth")
    sys.path.insert(0, repo)
    os.chdir(repo)

    import mamba_ssm.ops.selective_scan_interface as ssi
    if "site-packages" not in ssi.__file__:
        raise SystemExit(f"[env] local mamba_ssm shim detected at {ssi.__file__}")
    if not torch.cuda.is_available():
        raise SystemExit("[env] CUDA is required")

    from basicsr.models.archs.RetinexDuelSambaFusionFinalization_arch import (
        RetinexDuelSambaFusionFinalization,
    )
    from basicsr.utils import img2tensor

    torch.backends.cudnn.benchmark = True
    model = RetinexDuelSambaFusionFinalization(
        in_channels=3, out_channels=3, L_n_feat=16, R_n_feat=16
    ).cuda().eval()
    sd = torch.load(weights, map_location="cpu")
    sd = sd["params"] if "params" in sd else sd
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    if args.image:
        import cv2
        img = cv2.imread(os.path.expanduser(args.image), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise SystemExit(f"could not read {args.image}")
        x = (img2tensor(img).cuda() / 255.0).unsqueeze(0)
        src = os.path.basename(args.image)
    elif args.synthetic:
        torch.manual_seed(0)
        x = torch.rand(1, 3, 2160, 3840, device="cuda")
        src = "synthetic 3840x2160"
    else:
        raise SystemExit("pass --image or --synthetic")

    x = check_image_size(x)
    print(f"[input] {src}, padded to {tuple(x.shape)}", flush=True)

    print("[warmup]", flush=True)
    for _ in range(2):
        forward_once(model, x)
    torch.cuda.synchronize()

    modes = [("A", "natural", "Natural inference, as released", "no"),
             ("B", "replay", "Exact RNG replay", "yes"),
             ("C", "argmax", "Deterministic argmax routing", "n/a")]

    results = []
    for tag, mode, label, reset in modes:
        print(f"[run] {tag} {label}", flush=True)
        r = run_experiment(model, x, mode, n=args.passes)
        results.append((tag, label, reset, r))
        print(f"      identical={r['identical']} max={r['max_hi']:.4f} "
              f"mean={r['mean']:.2e} psnr={r['psnr']:.2f}", flush=True)

    print(f"\nInput {src}, padded to {tuple(x.shape)}. {args.passes} passes, "
          f"runs 2 to {args.passes} compared against run 1.\n")
    print("| | Mode | RNG reset per forward | Outputs identical | Max pixel delta | Mean delta | Pairwise PSNR |")
    print("|---|---|---|---|---|---|---|")
    for tag, label, reset, r in results:
        if r["identical"]:
            ident, delta, mean, psnr = "**yes, bit-identical**", "0.000", "0.000", "inf"
        else:
            ident = "**no**"
            delta = (f"{r['max_lo']:.3f} to {r['max_hi']:.3f}"
                     if r["max_lo"] != r["max_hi"] else f"{r['max_hi']:.3f}")
            mean, psnr = f"{r['mean']:.2e}", f"**{r['psnr']:.2f} dB**"
        print(f"| **{tag}** | {label} | {reset} | {ident} | {delta} | {mean} | {psnr} |")

    a = next(r for t, _, _, r in results if t == "A")
    b = next(r for t, _, _, r in results if t == "B")
    print()
    if not a["identical"] and b["identical"]:
        print("Verdict: A differs, B is bit-identical. Run-to-run variation is attributable")
        print("entirely to random sampling in the routing path. No kernel-level nondeterminism.")
    elif a["identical"]:
        print("Verdict: A did not differ. Either this build suppresses the sampling or the")
        print("input is degenerate. Investigate before citing anything here.")
    else:
        print("Verdict: B also differed, so something beyond routing RNG is nondeterministic.")
        print("This contradicts the recorded result and needs investigating.")


if __name__ == "__main__":
    main()

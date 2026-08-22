#!/usr/bin/env python3
"""Audit the released checkpoint against the released code, and emit an artifact.

README section 4.1 claims the checkpoint predates the code by 60 tensors, that
all 60 belong to SpectralGuidanceModule instances, and that those modules cannot
affect the output. The last part is the load-bearing one, because it is what
turns a scary-looking 60 missing keys into something harmless. It deserves to be
measured rather than asserted.

Three independent lines of evidence for inertness, any one of which would do:

  1. every SGM alpha parameter is exactly 0.0, and the module's output is
     freq + alpha * (conditioned - freq), so alpha = 0 makes it the identity
  2. forward hooks count zero SGM invocations during a full forward pass
  3. physically replacing every SGM with nn.Identity leaves the output
     bit-identical

Also records the numbers ENVIRONMENT.md quotes: tensor and parameter counts, the
kernel agreement against the reference implementation, and the harness matmul
check.

Usage: python scripts/verify_checkpoint.py --repo ~/RetinexDual --out results/checkpoint_audit.json
"""
import argparse
import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


def check_image_size(x, mult=128):
    _, _, h, w = x.size()
    return F.pad(x, (0, (mult - w % mult) % mult, 0, (mult - h % mult) % mult), "reflect")


def forward_once(model, x):
    with torch.inference_mode():
        out = model(x)
    while isinstance(out, (tuple, list)):
        out = out[0]
    return out.detach().float().clone()


def find_sgm(model):
    """SGM instances, located by class name rather than by parameter path."""
    return [(n, m) for n, m in model.named_modules()
            if "spectralguidance" in type(m).__name__.lower()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--res", type=int, nargs=2, default=[1088, 1920],
                    help="resolution for the bit-identity test. Half 4K is plenty and "
                         "leaves room on smaller cards")
    args = ap.parse_args()

    repo = os.path.abspath(os.path.expanduser(args.repo))
    weights = args.weights or os.path.join(repo, "pretrained_weights", "UHD_LL.pth")
    sys.path.insert(0, repo)
    os.chdir(repo)

    import mamba_ssm.ops.selective_scan_interface as ssi
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
    if "site-packages" not in ssi.__file__:
        raise SystemExit(f"[env] local mamba_ssm shim detected at {ssi.__file__}")
    if not torch.cuda.is_available():
        raise SystemExit("[env] CUDA is required")

    from basicsr.models.archs.RetinexDuelSambaFusionFinalization_arch import (
        RetinexDuelSambaFusionFinalization,
    )

    torch.backends.cudnn.benchmark = False
    rec = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}

    model = RetinexDuelSambaFusionFinalization(
        in_channels=3, out_channels=3, L_n_feat=16, R_n_feat=16
    ).cuda().eval()
    sd = torch.load(weights, map_location="cpu")
    sd = sd["params"] if "params" in sd else sd
    missing, unexpected = model.load_state_dict(sd, strict=False)

    code_sd = model.state_dict()
    rec["code_tensors"] = len(code_sd)
    rec["code_params"] = sum(v.numel() for v in code_sd.values())
    rec["ckpt_tensors"] = len(sd)
    rec["ckpt_params"] = sum(v.numel() for v in sd.values())
    rec["missing_keys"] = len(missing)
    rec["unexpected_keys"] = len(unexpected)
    rec["missing_all_sgm"] = all(("sgm_mag" in k or "sgm_pha" in k) for k in missing)
    print(f"[load] code {rec['code_tensors']} tensors / {rec['code_params']:,} params")
    print(f"[load] ckpt {rec['ckpt_tensors']} tensors / {rec['ckpt_params']:,} params")
    print(f"[load] missing {rec['missing_keys']}, unexpected {rec['unexpected_keys']}, "
          f"all missing are SGM: {rec['missing_all_sgm']}")

    sgm = find_sgm(model)
    rec["sgm_modules"] = len(sgm)
    alphas = [float(p.abs().max()) for n, m in sgm
              for pn, p in m.named_parameters() if pn.endswith("alpha")]
    rec["sgm_alpha_count"] = len(alphas)
    rec["sgm_alpha_max_abs"] = max(alphas) if alphas else None
    print(f"[sgm ] {len(sgm)} modules, {len(alphas)} alpha tensors, "
          f"max |alpha| = {rec['sgm_alpha_max_abs']}")

    # Evidence 2: are they invoked at all?
    calls = {"n": 0}
    handles = [m.register_forward_hook(lambda *a, **k: calls.__setitem__("n", calls["n"] + 1))
               for _, m in sgm]
    h, w = args.res
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    x = check_image_size(torch.rand(1, 3, h, w, device="cuda"))

    # Capture the full RNG state, not just a seed. Routing samples from
    # gumbel_softmax on CUDA, so seeding the CPU generator alone leaves the two
    # forwards routing differently and the comparison below measures that instead
    # of the SGM removal. This is the same trap DETERMINISM.md documents, and the
    # first version of this script fell into it: it reported a 0.0936 delta that
    # was entirely stochastic routing.
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all()
    base = forward_once(model, x)
    for hd in handles:
        hd.remove()
    rec["sgm_forward_calls"] = calls["n"]
    print(f"[sgm ] forward invocations during one pass: {calls['n']}")

    # Evidence 3: replace with identity and compare bit for bit.
    for name, _ in sgm:
        parent = model
        parts = name.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], nn.Identity())
    torch.set_rng_state(cpu_state)
    torch.cuda.set_rng_state_all(cuda_state)
    stripped = forward_once(model, x)
    delta = float((base - stripped).abs().max())
    rec["sgm_removed_max_abs_delta"] = delta
    rec["sgm_removed_bit_identical"] = delta == 0.0
    print(f"[sgm ] max |delta| with every SGM replaced by Identity: {delta}")

    # Kernel agreement, the number ENVIRONMENT.md quotes.
    torch.manual_seed(0)
    b, d, l, n = 1, 8, 64, 16
    u = torch.randn(b, d, l, device="cuda")
    dl = torch.rand(b, d, l, device="cuda")
    A = -torch.rand(d, n, device="cuda") - 0.5
    B = torch.randn(b, n, l, device="cuda")
    C = torch.randn(b, n, l, device="cuda")
    D = torch.randn(d, device="cuda")
    kd = float((selective_scan_fn(u, dl, A, B, C, D, delta_softplus=True)
                - selective_scan_ref(u, dl, A, B, C, D, delta_softplus=True)).abs().max())
    rec["kernel_vs_reference_max_abs_diff"] = kd
    print(f"[kern] selective_scan_fn vs ref: {kd:.2e}")

    # Harness matmul check.
    N = 8192
    a = torch.randn(N, N, device="cuda", dtype=torch.float16)
    bb = torch.randn(N, N, device="cuda", dtype=torch.float16)
    for _ in range(3):
        a @ bb
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(10):
        a @ bb
    e.record()
    torch.cuda.synchronize()
    ms = s.elapsed_time(e) / 10
    rec["matmul_tflops"] = (2 * N ** 3) / (ms / 1000) / 1e12
    print(f"[kern] 8192^3 fp16 matmul: {rec['matmul_tflops']:.1f} TFLOPS")

    ok = (rec["unexpected_keys"] == 0 and rec["missing_all_sgm"]
          and rec["sgm_alpha_max_abs"] == 0.0 and rec["sgm_forward_calls"] == 0
          and rec["sgm_removed_bit_identical"] and rec["kernel_vs_reference_max_abs_diff"] < 1e-3)
    rec["verdict_inert"] = bool(ok)

    print()
    if ok:
        print("All three lines of evidence agree: the missing tensors cannot reach the output.")
        print("alpha is exactly zero, the modules are never invoked, and removing them")
        print("leaves the output bit-identical.")
    else:
        print("At least one check failed. README section 4.1 claims these modules are inert,")
        print("so investigate before trusting that claim.")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(rec, f, indent=2)
        print(f"wrote {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

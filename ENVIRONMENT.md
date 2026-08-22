# Environment and Provenance

> Part of [**Reproducing RetinexDual**](README.md), an independent reproduction of
> [arXiv:2508.04797](https://arxiv.org/abs/2508.04797) on the UHD-LL low-light benchmark.
> The result and the defects are in [the main report](README.md). This file is the
> supporting detail: versions, checksums, and every deviation with its reason.
>
> Related: [`DETERMINISM.md`](DETERMINISM.md) on why inference is not reproducible run to run,
> and [`results/`](results/) for the raw data.

## Contents

- [Provenance](#provenance)
- [Hardware and OS](#hardware-and-os)
- [Software](#software)
- [Deviations from requirements.txt, and why](#deviations-from-requirementstxt-and-why)
- [Verification performed after install](#verification-performed-after-install)
- [Baseline configuration](#baseline-configuration)

Everything needed to judge whether the numbers in [`README.md`](README.md) are comparable to the
paper's, and to rebuild this environment from scratch.

## Provenance

| Item | Value |
|---|---|
| Upstream repository | https://github.com/ErrorLogic1211/RetinexDual |
| Commit | `9feec2c0814d740221db2323e5e815a4d455abb6` (master HEAD) |
| Weights source | HuggingFace `ErrorLogic/RetinexDual`, file `UHD_LL.pth` |
| `UHD_LL.pth` size | 19,272,064 bytes |
| `UHD_LL.pth` sha256 | `1977bb774cefb360bcb6edecdf4606568fa59f53aab8717fbe13bc35bacae182` |
| Dataset | UHD-LL testing set, 150 pairs, all 3840x2160 |
| Dataset size | 197,703,282 bytes, 0 corrupt, 150/150 paired |

## Hardware and OS

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4090, 24 GB (24564 MiB), compute capability 8.9 |
| Driver | 580.126.09 |
| OS | Ubuntu 20.04.6 LTS, kernel 5.15.0-67 |
| **glibc** | **2.31**, which is what constrains the choice of prebuilt CUDA wheel |
| CPU RAM | 62 GB |
| gcc | 11.5.0 |

GPU state during measurement, idle to loaded: 41 C and 555 MHz idle, then 2700 MHz SM clock,
10251 MHz memory, 69 to 73 C, 304 to 306 W against a 450 W limit under load. Persistence mode
disabled. No other process held GPU memory.

**No thermal or power throttling was observed.** SM clock stayed pinned at 2700 MHz across all
measurement groups. This matters because a throttled run would depress latency numbers in a way
that looks like a real effect.

## Software

| Package | Version | Note |
|---|---|---|
| Python | 3.11 (conda-forge) | repository is tested on 3.9, system python3.8 is too old for torch 2.6 |
| **torch** | **2.6.0+cu124** | **deviation, see below** |
| torchvision | 0.21.0 | paired with torch 2.6.0 |
| **mamba_ssm** | **2.2.4** | exactly as pinned in `requirements.txt` |
| **causal_conv1d** | **1.5.0.post8** | exactly as pinned in `requirements.txt` |
| CUDA (torch build) | 12.4 | |
| cxx11 ABI | False | torch 2.6.0 linux wheels are pre-cxx11-ABI, and wheel tags must match |

## Deviations from `requirements.txt`, and why

**`requirements.txt` cannot be satisfied as written on any system with glibc 2.31 or older.**

It pins `torch==2.7.1` alongside `mamba_ssm==2.2.4` and `causal_conv1d==1.5.0.post8`. Checked
against the GitHub release API, neither kernel package publishes a torch2.7 wheel. Both stop at
torch2.6, having been released on 2024-12-06, before PyTorch 2.7 existed. The pinned combination
therefore has no solution from prebuilt wheels.

Going the other way, keeping torch 2.7.1 and bumping the kernels to their first torch2.7 builds
(`mamba_ssm` 2.3.0 and `causal_conv1d` 1.6.0), fails on this machine:

```
ImportError: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.32' not found
    (required by .../selective_scan_cuda.cpython-311-x86_64-linux-gnu.so)
```

Maximum required glibc symbol, measured per wheel rather than assumed:

| Wheel | Requires | Loads on glibc 2.31 |
|---|---|---|
| `mamba_ssm` 2.3.0 / torch2.7 | GLIBC_2.32 | no |
| `causal_conv1d` 1.6.0 / torch2.7 | GLIBC_2.32 | no |
| **`mamba_ssm` 2.2.4 / torch2.6** | **GLIBC_2.14** | **yes** |
| **`causal_conv1d` 1.5.0.post8 / torch2.6** | **GLIBC_2.14** | **yes** |

**Resolution: pin torch to 2.6.0 and use the repository's exact pinned kernel versions.** This keeps
`mamba_ssm` and `causal_conv1d` byte-identical to the authors' intent and moves only torch.

**Move only torch. The other pins are load-bearing.** Relaxing `transformers` while fixing the torch
conflict breaks the build at the last step, because `mamba_ssm`'s top-level `__init__` imports
`MambaLMHeadModel`, which reaches `transformers.generation` for `GreedySearchDecoderOnlyOutput`, a
name removed in transformers 5.x. Unpinned, pip resolves to 5.x and `import mamba_ssm` fails.
Upstream's `transformers==4.52.4` is correct and `setup_env.sh` keeps it. Verified by rebuilding the
environment from scratch.

That substitution was checked empirically rather than argued from version numbers. After install,
`selective_scan_fn` (CUDA) agrees with `selective_scan_ref` (pure PyTorch) to a maximum absolute
difference of 3.8e-6, and the full reproduction lands within 0.030 dB of the published PSNR.
`setup_env.sh` runs the kernel comparison as an assertion, so a mismatched or miscompiled wheel
fails at setup time instead of quietly producing wrong numbers.

### Install step

`python setup.py develop --no_cuda_ext`, the command the upstream README gives, fails. `setup.py`
calls `import torch` at line 9, and pip's PEP-517 build isolation hides torch from the build
environment. `--no_cuda_ext` also cannot survive a PEP-517 build.

Because `--no_cuda_ext` means there are no extensions to build, `basicsr` is instead made importable
with a `.pth` file in site-packages. `basicsr` has no `__init__.py` and resolves as a namespace
package, so this is equivalent.

## Verification performed after install

Each of these was run before any result was recorded.

1. **Kernel correctness.** `selective_scan_fn` (CUDA) against `selective_scan_ref` (pure PyTorch
   reference): maximum absolute difference **3.8e-6**.
2. **No shadowing.** Confirmed that no local directory named `mamba_ssm` sits ahead of
   site-packages on `sys.path`. Python puts the working directory first, so a stray local package
   silently replaces the real kernels, and any timing taken afterwards is meaningless.
3. **Harness sanity.** An 8192-cubed fp16 matmul reaches **161.4 TFLOPS**, consistent with this
   card's dense fp16 specification. A harness that cannot recover a known cost is not trusted to
   measure an unknown one.
4. **Weight load.** 60 missing keys and 0 unexpected keys, all inside 12 `SpectralGuidanceModule`
   instances whose `alpha` is zero-initialised. Proven inert by substituting an identity and
   getting bit-identical output. Checkpoint parameter count 4,725,531 matches the paper's stated
   4.726M, while the released code builds 4,747,035.

## Baseline configuration

Recorded explicitly so none of it can later be presented as an optimization win.

- `torch.backends.cudnn.benchmark = True`, since input dimensions are fixed.
- `model.eval()` together with `torch.inference_mode()`, not merely `no_grad`.
- Input reflect-padded from 3840x2160 to **3840x2176** by `check_image_size`, a multiple of 128,
  matching `inference_RetinexDual.py`. Output cropped back to 2160 before metrics.
- Baseline forward pass, stochastic routing as released: **1347.73 ms** median (5 groups of 5
  iterations, CUDA-event timing, wall clock agreeing to 0.02%).
- The working set is **10.94 GiB** at 3840x2176, linear at 1.31 GiB per Mpix.
- Reserved depends on two independent settings, so state which you mean:

  | `cudnn.benchmark` | allocator | reserved at 3840x2176 |
  |---|---|---|
  | on (as the reproduction runs) | default | **19.54 GiB** |
  | off | default | 19.67 GiB |
  | off | `expandable_segments:True` | **11.64 GiB** |

  The allocator setting is what matters, not `cudnn.benchmark`, which moves reserved by about
  0.1 GiB here. With expandable segments 4K fits on a 16 GB card. Without, it needs 24 GB. See
  README section 5, and the note there on why peak allocated is the wrong figure to quote while
  `cudnn.benchmark` is on.

# Reproducing RetinexDual

An independent reproduction of **RetinexDual** (Kishawy, Hussein and Chen, ICPR 2026,
[arXiv:2508.04797](https://arxiv.org/abs/2508.04797)) on the UHD-LL low-light benchmark.

The paper's headline number reproduces. Getting there took working around three undocumented defects
in the released repository, each of which stops a clean checkout from running at all, plus two
further traps that do not stop it running and instead hand you a number that is quietly wrong. This
repo records both halves, the confirmation and the fixes, with the raw per-image data behind them.

**Result: PSNR 28.8236 dB against 28.79 reported, a difference of +0.034 dB.**

Upstream: [ErrorLogic1211/RetinexDual](https://github.com/ErrorLogic1211/RetinexDual) at commit
`9feec2c0814d740221db2323e5e815a4d455abb6`.

---

## 1. The headline reproduces

UHD-LL testing set, all 150 pairs at 3840x2160, released `UHD_LL.pth` weights, released inference
path, RTX 4090.

Default inference is stochastic (see section 4.2), so a single run is not a reproducible quantity.
Every number below is the mean over 5 independent seeds, each a full 150-image evaluation.

| Metric | Reported | Reproduced | Difference |
|---|---|---|---|
| PSNR (dB) | 28.79 | **28.8236 +/- 0.0042** | **+0.034** |
| SSIM | 0.934 | 0.92217 +/- 0.00001 | -0.012 |

![Per-image PSNR across the UHD-LL test set, and the per-seed spread against the published value](assets/reproduction.png)

Spread across seeds was 28.8180 to 28.8287. Per-image scores range from 16.73 dB to 40.33 dB, with
a mean run-to-run standard deviation of 0.029 dB per image.

The right panel is the useful one. All five seeds sit inside a 0.011 dB band, roughly a third of the
0.034 dB gap to the published value, so the reproduction is comfortably tighter than the quantity it
is being compared against. The left panel is a reminder that a dataset mean hides a lot: individual
images span more than 23 dB.

PSNR lands 0.034 dB above the published value, which is well inside what counts as a match. **The
SSIM gap of 0.012 does not reproduce and I could not explain it.** The most likely cause is a
protocol difference rather than a model difference, because the repository's own SSIM helper
averages per-channel SSIM over RGB, and several common implementations do not. I did not confirm
this, and I am not claiming the paper is wrong. It is an open question and the one thing I would
ask the authors.

Raw data: [`results/reproduction_seeds.csv`](results/reproduction_seeds.csv) (5 rows, one per seed)
and [`results/reproduction_per_image.csv`](results/reproduction_per_image.csv) (150 rows).

### Metric protocol

Stated explicitly, because PSNR is only comparable when the protocol is:

- RGB channels, not luma. Range [0, 255]. `20 * log10(255 / sqrt(mse))`.
- No border crop.
- Computed per image, then averaged over the 150 images.
- Input is reflect-padded from 3840x2160 to 3840x2176 by `check_image_size`, matching the released
  `inference_RetinexDual.py`. Output is cropped back to 2160 before scoring.

---

## 2. Three defects that stop the released code from running

All three are in the released repository as of commit `9feec2c`. They are independent, and you hit
them in this order.

### 2.1 `requirements.txt` cannot be installed as written

The file pins these three lines together:

```
torch==2.7.1
mamba_ssm==2.2.4
causal_conv1d==1.5.0.post8
```

`mamba_ssm` and `causal_conv1d` ship prebuilt CUDA wheels tagged against a specific torch version.
Checked against the GitHub release API, **neither package publishes a wheel for torch 2.7**. Both
top out at torch 2.6, having been built in December 2024, before PyTorch 2.7 existed.

So the pinned set has no solution. Any environment built by following the README either silently
lacks the Mamba kernels or fails outright.

### 2.2 The obvious fix fails on any glibc older than 2.32

The natural response is to keep torch 2.7.1 and bump the kernels to their first torch 2.7 builds,
`mamba_ssm==2.3.0` and `causal_conv1d==1.6.0`. On Ubuntu 20.04 that produces:

```
ImportError: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.32' not found
    (required by .../selective_scan_cuda.cpython-311-x86_64-linux-gnu.so)
```

Those wheels are built on newer CI runners. Maximum required glibc symbol, measured per wheel:

| Wheel | Requires | Loads on glibc 2.31 |
|---|---|---|
| `mamba_ssm` 2.3.0 / torch2.7 | GLIBC_2.32 | no |
| `causal_conv1d` 1.6.0 / torch2.7 | GLIBC_2.32 | no |
| `mamba_ssm` 2.2.4 / torch2.6 | GLIBC_2.14 | yes |
| `causal_conv1d` 1.5.0.post8 / torch2.6 | GLIBC_2.14 | yes |

Ubuntu 20.04 ships glibc 2.31. It is still a very common deployment target and it is what this
machine runs.

**Resolution used here: pin torch to 2.6.0 and keep the repository's exact pinned kernel versions.**
This moves only torch and leaves `mamba_ssm` and `causal_conv1d` at exactly the versions the authors
pinned, which is the smaller deviation of the two available.

The check that backs this is empirical rather than an argument about version numbers. After
install, `selective_scan_fn` (CUDA) agrees with `selective_scan_ref` (the pure PyTorch reference) to
a maximum absolute difference of **3.8e-6**, and the end-to-end reproduction in section 1 lands
within 0.034 dB of the published PSNR. `setup_env.sh` runs that kernel comparison as an assertion,
so a mismatched wheel fails at setup rather than silently producing wrong numbers.

### 2.3 The documented install command fails

The README instructs:

```bash
python setup.py develop --no_cuda_ext
```

This fails. `setup.py` calls `import torch` at line 9, and pip's PEP-517 build isolation hides torch
from the build environment. The `--no_cuda_ext` flag cannot survive a PEP-517 build either.

Since `--no_cuda_ext` means there are no extensions to build in the first place, the install reduces
to making `basicsr` importable. A `.pth` file in site-packages does that. `basicsr` has no
`__init__.py` and resolves as a namespace package, so this works.

---

## 3. Two traps that do not stop the code running

These are worse in one respect than the three above. A build failure is loud. These two run to
completion and hand you a number that is wrong.

### 3.1 The documented CPU fallback cannot run at UHD

The upstream README suggests that on machines where the CUDA kernels will not build you can drop in
a pure-PyTorch `selective_scan_ref` and alias `selective_scan_fn` to it, describing it as
"numerically equivalent, just slower."

Numerically equivalent it is. Viable at ultra-high definition it is not. At this model's largest
sequence length the reference path allocates two `(B, d_inner, L, d_state)` tensors, `deltaA` and
`deltaB_u`. With `B=1, d_inner=72, L=2,088,960, d_state=16` in fp32 that is **9.6 GB each, about
19 GB before anything else**, and it then runs a Python-level `for` loop over all 2,088,960
timesteps, appending to a list that is stacked at the end.

"Slower" undersells this by orders of magnitude. The correct reading is that the fallback is a
correctness reference only, and **any latency measured against it is not a measurement of
RetinexDual**. For scale, the real CUDA kernel runs that same scan in 29.2 ms.

A related hazard, and the reason `scripts/evaluate.py` checks for it: Python puts the working
directory ahead of site-packages, so a local directory named `mamba_ssm` silently shadows the
installed package. If that directory contains the reference fallback, everything imports, everything
runs, and every number afterwards is meaningless.

### 3.2 The metric average divides by the wrong count

In `inference_RetinexDual.py`, `num_img` increments in **both** branches of the ground-truth check
(lines 136 and 143), while `psnr_all`, `ssim_all` and `lpips_all` accumulate only in the branch
where ground truth was found:

```python
if gt_img is not None:
    ssim_all += ssim
    psnr_all += psnr
    lpips_all += lpips_value
    num_img += 1
else:
    num_img += 1          # <- counted, but nothing was added to the sums
...
print('avg_psnr:%f' % (psnr_all / num_img))
```

So any input image without a matching ground truth file is counted in the divisor but contributes
zero to the numerator, which drags the reported average down.

**Scope, stated carefully.** On the official UHD-LL testing set all 150 images are paired, so this
does not fire and it has no bearing on the paper's reported number. It bites when you point the
script at a partial download, a subset or a folder with mismatched filenames, which is exactly what
a reproducer does first. The same guard also means a silently truncated Google Drive download
produces a plausible-looking but deflated PSNR rather than an error. `scripts/evaluate.py` counts
only paired images and prints a warning when it finds unpaired ones.

---

## 4. Two properties of the released artifacts

Neither of these is a reproduction failure. Both are things a reproducer should know.

### 4.1 The released checkpoint predates the released code

Loading `UHD_LL.pth` reports **60 missing keys and 0 unexpected keys**. The released
`inference_RetinexDual.py` uses `strict=False`, so this passes in silence.

| | Tensors | Parameters |
|---|---|---|
| Model built by the released code | 826 | 4,747,035 |
| Released checkpoint | 766 | 4,725,531 |

The checkpoint's 4,725,531 parameters match the paper's stated 4.726M. The code builds 4.747M. All
60 extra tensors belong to 12 `SpectralGuidanceModule` instances in the illumination branch. The
conclusion is simply that **the released code is a later revision than the released checkpoint**.

This is harmless, and I verified that rather than assuming it. Every `SpectralGuidanceModule` ends
with `out = freq_features + self.alpha * (conditioned - freq_features)` where `alpha` is
zero-initialised. All 12 alphas load as exactly 0.0, and replacing every such module with an
identity gives bit-identical output, maximum absolute delta 0.0. The randomly initialised tensors
cannot reach the output.

**This says nothing about whether the paper's numbers are right.** It shows only that two released
artifacts come from different revisions.

### 4.2 Default inference is not deterministic

`RetinexDuelSambaFusionFinalization_arch.py:744` calls:

```python
cls_policy = F.gumbel_softmax(pred_route, hard=True, dim=-1)
```

There is no `if self.training` guard anywhere on this path, so `model.eval()` and
`torch.inference_mode()` do not suppress the sampling. `inference_RetinexDual.py:14` sets
`torch.manual_seed(123)` once at import, which is not sufficient, because warmup passes and repeated
iterations consume RNG state and successive forwards then route differently.

Five forward passes of one input, runs 2 to 5 compared against run 1:

| Mode | RNG reset per forward | Identical | Max pixel delta | Pairwise PSNR |
|---|---|---|---|---|
| Natural inference, as released | no | **no** | 0.210 to 0.466 | 51.50 dB |
| Exact RNG replay | yes | yes, bit-identical | 0.000 | inf |
| Deterministic argmax routing | n/a | yes, bit-identical | 0.000 | inf |

The RNG replay row is the decisive one. Restoring CPU and CUDA RNG state before each forward makes
output bit-identical, which isolates the variation entirely to sampling in the routing path and
rules out kernel-level nondeterminism. Had any existed, that row would still have differed.

The practical consequence, and the reason section 1 reports 5 seeds rather than one run: **the
released evaluation path produces run-dependent output and therefore run-dependent PSNR.** Any
comparison against it needs repeated evaluation and a stated variance.

To be clear about scope, this is a fact about the released implementation run on my hardware. It is
not evidence about how the paper's table was produced. The authors may have used a fixed seed, a
per-image RNG reset, a different commit or a separate evaluation wrapper.

---

## 5. Environment

Full detail, including checksums and the reasoning behind each version choice, is in
[`ENVIRONMENT.md`](ENVIRONMENT.md).

| Item | Value |
|---|---|
| GPU | RTX 4090, 24 GB, compute capability 8.9 |
| OS | Ubuntu 20.04.6 LTS, glibc 2.31 |
| Python | 3.11 |
| torch | 2.6.0+cu124 (deviation, see 2.2) |
| mamba_ssm | 2.2.4 (exactly as pinned) |
| causal_conv1d | 1.5.0.post8 (exactly as pinned) |
| Weights | `UHD_LL.pth`, sha256 `1977bb77...cae182`, 19,272,064 bytes |
| Dataset | UHD-LL testing set, 150 pairs, 0 corrupt, 150/150 paired |

Baseline forward pass at 3840x2176 measures 1347.89 ms median. Peak allocation is 11.2 GB, so
whole-image 4K inference needs a 16 GB card at minimum.

Harness sanity check before trusting any timing: an 8192-cubed fp16 matmul reaches 162.8 TFLOPS,
in line with this card's specification.

---

## 6. Reproducing this

```bash
git clone https://github.com/lemoon01110/Reproducing-RetinexDual
cd Reproducing-RetinexDual
bash setup_env.sh          # builds the working environment described in section 2
bash reproduce.sh          # full 150-image evaluation, 5 seeds
```

`reproduce.sh` writes `results/reproduction_seeds.csv` and
`results/reproduction_per_image.csv` and prints the comparison table from section 1.

You need to supply the UHD-LL testing set yourself. It is distributed through Google Drive, which
rate-limits folder downloads of this size, so an automated fetch is not reliable.
`scripts/check_data.py` prints the exact layout expected and verifies pairing and image dimensions
before anything runs.

---

## 7. Scope, and what this is not

- This is a reproduction of one task, low-light enhancement on UHD-LL. The paper covers four. I did
  not attempt deraining, deblurring or dehazing.
- Everything here is inference. No training was run and no claim about trainability is made.
- The three defects are packaging and documentation problems in a released repository. They are
  ordinary, they are the normal condition of research code, and they say nothing about the quality
  of the method. The method reproduces.
- Where a number came out against my expectation, in particular the SSIM gap, it is reported as
  unresolved instead of explained away.

## 8. Files

| Path | Contents |
|---|---|
| [`ENVIRONMENT.md`](ENVIRONMENT.md) | versions, checksums, provenance, every deviation and why |
| [`DETERMINISM.md`](DETERMINISM.md) | the nondeterminism audit in full |
| [`setup_env.sh`](setup_env.sh) | builds the working environment, and verifies the kernels load |
| [`reproduce.sh`](reproduce.sh) | checksums the weights, checks the data, runs the evaluation |
| [`scripts/evaluate.py`](scripts/evaluate.py) | the evaluation itself, 5 seeds over 150 images |
| [`scripts/check_data.py`](scripts/check_data.py) | dataset pairing, dimensions and decodability |
| [`scripts/make_figure.py`](scripts/make_figure.py) | renders the figure above from the committed CSVs |
| `results/reproduction_seeds.csv` | 5 rows, dataset-level PSNR and SSIM per seed |
| `results/reproduction_per_image.csv` | 150 rows, per-image mean and standard deviation |
| [`results/README.md`](results/README.md) | provenance and column meanings for both CSVs |

## Citation

The work being reproduced:

```bibtex
@inproceedings{kishawy2026retinexdual,
  title={RetinexDual: Retinex-based Dual Nature Approach for Generalized
         Ultra-High-Definition Image Restoration},
  author={Kishawy, Mohab and Hussein, Ali Abdellatif and Chen, Jun},
  booktitle={Proceedings of the 28th International Conference on Pattern Recognition (ICPR)},
  address={Lyon, France},
  month={August},
  year={2026}
}
```

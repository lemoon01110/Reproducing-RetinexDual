# Is RetinexDual Inference Deterministic

Short answer: no, not as released. This document shows why, isolates the cause, and explains what it
means for anyone trying to reproduce a PSNR number.

## Scope

Everything below is a fact about the **released implementation**
([ErrorLogic1211/RetinexDual](https://github.com/ErrorLogic1211/RetinexDual) at `9feec2c`) run on
the hardware described in [`ENVIRONMENT.md`](ENVIRONMENT.md).

It is **not** evidence about how the paper's published table was produced. The authors may have used
a fixed seed, a per-image RNG reset, a different commit or a separate evaluation wrapper. Nothing
here implies the published numbers are unreliable, and the reproduction in
[`README.md`](README.md) confirms the headline.

## Mechanism, read from the source

In `basicsr/models/archs/RetinexDuelSambaFusionFinalization_arch.py`:

```python
742:  pred_route = self.route(x)                                     # [B, HW, num_token]
744:  cls_policy = F.gumbel_softmax(pred_route, hard=True, dim=-1)   # draws random numbers
746:  prompt = torch.matmul(cls_policy, full_embedding).view(B, n, self.d_state)
748:  detached_index = torch.argmax(cls_policy.detach(), dim=-1, keepdim=False).view(B, n)
749:  x_sort_values, x_sort_indices = torch.sort(detached_index, dim=-1, stable=False)
```

`F.gumbel_softmax` is called unconditionally. There is no `if self.training` guard anywhere on this
path, so `model.eval()` and `torch.inference_mode()` do not suppress the sampling.

`inference_RetinexDual.py:14` sets `torch.manual_seed(123)` once at import. That is not sufficient.
Warmup passes and repeated iterations consume RNG state, so successive forward passes route
differently.

Note that `cls_policy` feeds two places, the `prompt` matmul on line 746 and the sort key on line
748. The Gumbel noise therefore does not cancel under the later `argmax`.

## Experiment

Regenerate this table with:

```bash
python scripts/determinism_audit.py --repo ~/RetinexDual \
    --image-dir <UHD-LL testing_set/input> --n-images 5
```

Five real UHD-LL test images, sampled evenly across the split rather than taken from the front, each
`1x3x2176x3840` after reflect-padding to a multiple of 128. Five forward passes per image, runs 2 to
5 each compared against run 1. Full warmup first, `cudnn.benchmark=True`. Deltas are on a [0, 1]
image scale.

| | Mode | RNG reset per forward | Outputs identical | Max pixel delta | Mean delta | Pairwise PSNR |
|---|---|---|---|---|---|---|
| **A** | Natural inference, as released | no | **no** | 0.008 to 0.135 | 4.95e-04 | **58.59 to 65.17 dB** |
| **B** | Exact RNG replay | yes | **yes, bit-identical** | 0.000 | 0.000 | inf |
| **C** | Deterministic argmax routing | n/a | **yes, bit-identical** | 0.000 | 0.000 | inf |

Per image, natural inference:

| image | max delta | mean delta | pairwise PSNR |
|---|---|---|---|
| `1003_UHD_LL.JPG` | 0.1354 | 7.87e-04 | 58.59 dB |
| `1453_UHD_LL.JPG` | 0.0249 | 3.94e-04 | 64.71 dB |
| `1778_UHD_LL.JPG` | 0.0186 | 3.98e-04 | 65.17 dB |
| `28_UHD_LL.JPG` | 0.0162 | 4.89e-04 | 62.88 dB |
| `674_UHD_LL.JPG` | 0.0131 | 4.06e-04 | 64.23 dB |

**Two earlier versions of this table were measured on a single input and both misrepresented the
spread.** The first used synthetic uniform-random noise and reported a max delta of 0.210 to 0.466
at 51.50 dB, far worse than any real image, because noise gives the router no spatial structure and
its decisions scatter more. The second used `1003_UHD_LL.JPG` alone, which the per-image table above
shows is the *most* variable of the five at 58.59 dB against 62.88 to 65.17 for the rest. Neither
was wrong as a measurement. Both were a single point presented as a general claim.

Note also that the max delta is itself stochastic, since the audit does not fix a seed. Repeated
runs on the same image give different maxima. The stable quantities are the mean delta, around
5e-04, and the identical or not verdict, which is what the argument actually rests on.

## Interpretation

**Experiment B is the one that carries the argument.** Restoring both CPU and CUDA RNG state before
each forward makes the output bit-identical. It follows that:

- Run-to-run variation is attributable **entirely to random sampling in the routing path**.
- There is **no kernel-level nondeterminism**. No atomics in the scatter or gather, no
  algorithm-selection drift, no nondeterministic selective-scan reduction. Had any of those existed,
  experiment B would still have differed.

Without experiment B, row A on its own is ambiguous, since nondeterministic reductions in a CUDA
kernel would produce a similar-looking result for an entirely different reason.

The perturbation is localised rather than spread evenly. Mean absolute delta is about 5e-04, well
under one 8-bit level, while the maximum reaches 0.135 on the worst image, roughly 34 levels. So
most pixels are untouched and a small number move a lot, which is what flipping a routing decision
for a subset of tokens would do.

## What this means for reproduction

1. **A single evaluation run is not a reproducible quantity.** Dataset-level PSNR is run-dependent.
   This is why [`README.md`](README.md) reports the mean over 5 independent seeds along with the
   spread, rather than a single number.
2. Measured across 5 seeds on the full 150-image test set, dataset PSNR moves by **0.0027 dB**
   standard deviation, and individual images move by **0.029 dB** on average. So the effect is
   small at dataset scale. Small is not zero, and any claimed improvement smaller than that spread
   is indistinguishable from noise.
3. Deterministic argmax routing changes inference semantics but touches no parameters and needs no
   retraining, so it is available to anyone who wants a reproducible evaluation path.

The defensible one-line statement: *the released evaluation path yields run-dependent output, and
therefore run-dependent PSNR and SSIM, under its default inference behaviour.*

# Results

## Provenance

These files are the recorded reproduction reported in [`../README.md`](../README.md) section 1.

They were produced by [`../scripts/evaluate.py`](../scripts/evaluate.py), the script shipped in this
repository, on the machine described in [`../ENVIRONMENT.md`](../ENVIRONMENT.md). Released
`UHD_LL.pth` weights, released inference path, unmodified routing, all 150 UHD-LL testing pairs,
once per seed for seeds 0 through 4.

Regenerate with:

```bash
REPO_DIR=~/RetinexDual DATA_DIR=~/data/UHD_LL/testing_set bash ../reproduce.sh
```

## Files

### `reproduction_seeds.csv`

Five rows, one per seed. Dataset-level means over 150 images.

| column | meaning |
|---|---|
| `seed` | value passed to `torch.manual_seed` before the pass |
| `n_images` | images scored, 150 in every row |
| `psnr_db` | mean PSNR over the 150 images, RGB, range [0, 255], no border crop |
| `ssim` | mean SSIM over the 150 images |

### `reproduction_per_image.csv`

150 rows, one per test image, aggregated across the 5 seeds.

| column | meaning |
|---|---|
| `image` | filename in the UHD-LL testing set |
| `n_seeds` | seeds contributing, 5 in every row |
| `psnr_mean_db` | mean PSNR for this image across seeds |
| `psnr_sd_db` | standard deviation across seeds, nonzero because inference is stochastic |
| `ssim_mean` | mean SSIM for this image across seeds |

### `reproduction_summary.json`

Machine-readable summary, including the torch version and GPU the run was made on.

## Summary

| quantity | value |
|---|---|
| PSNR, mean over seeds | 28.8199 dB |
| PSNR, standard deviation over seeds | 0.0027 dB |
| PSNR, range over seeds | 28.8173 to 28.8235 dB |
| PSNR reported in the paper | 28.79 dB |
| **PSNR difference** | **+0.030 dB** |
| SSIM, mean over seeds | 0.92217 |
| SSIM, standard deviation over seeds | 0.000016 |
| SSIM reported in the paper | 0.934 |
| **SSIM difference** | **-0.0118** |
| per-image PSNR range | 16.74 to 40.33 dB |
| mean per-image run-to-run standard deviation | 0.029 dB |

The `psnr_sd_db` column is why this reproduction reports a spread rather than a single number.
See [`../DETERMINISM.md`](../DETERMINISM.md).

## Cross-check against a second harness

These numbers were produced a second time by a separately written evaluation harness, built earlier
for a different experiment. It shares no code with `evaluate.py` beyond the repository's own
`calculate_psnr`, and it drives the model through a different call path, so it consumes the RNG
stream differently. Since routing is stochastic, that is a genuinely independent sample of the
released behaviour rather than a rerun of the same one.

| harness | seeds | mean PSNR | sd | range |
|---|---|---|---|---|
| `scripts/evaluate.py` (this repo) | 5 | 28.8199 | 0.0027 | 28.8173 to 28.8235 |
| separate earlier harness | 5 | 28.8236 | 0.0042 | 28.8180 to 28.8287 |

The means differ by 0.0036 dB. Welch's t-test on the two sets of seed-level means gives
**t = -1.63, p = 0.15**, so the difference is not statistically significant and the two harnesses
agree. Both also sit on the same side of the published value, about +0.03 dB.

This is worth stating because it rules out a specific failure mode. A reproduction that matches
only under the exact harness that produced it is weak evidence. Two independent implementations
landing in the same place is stronger.

One caveat, stated rather than buried: the second harness is not included in this repository, so
this row cannot be regenerated from what is published here. Only the first row can. It is recorded
as supporting context, not as a headline result.

## The SSIM gap

Scored over all 150 images by [`../scripts/ssim_protocol_probe.py`](../scripts/ssim_protocol_probe.py),
on one set of model outputs, to test whether the 0.0118 SSIM shortfall is a metric-definition
difference rather than a model difference:

| protocol | SSIM | vs repo helper | vs paper |
|---|---|---|---|
| `matlab_y` | 0.94656 | +0.02438 | +0.01256 |
| `matlab_y_border4` | 0.94652 | +0.02434 | +0.01252 |
| `repo_rgb_mean` | 0.92218 | 0.00000 | -0.01182 |

No convention tested reaches 0.934, and the published value falls between the two commonest ones.
The gap is not explained. See README section 1.

Worth noting that `repo_rgb_mean` here (0.92218) agrees with the main evaluation (0.92217) to five
decimals, which is a useful check that the two independent code paths compute the same thing.

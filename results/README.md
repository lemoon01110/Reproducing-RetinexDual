# Results

## Provenance

These two files are the recorded reproduction reported in [`../README.md`](../README.md) section 1.

They were produced on the machine described in [`../ENVIRONMENT.md`](../ENVIRONMENT.md) on
2026-08-09, running the released inference path with released `UHD_LL.pth` weights over all 150
UHD-LL testing pairs, once per seed, for seeds 0 through 4.

One point of provenance worth stating plainly. These rows were extracted from a larger routing
experiment that swept several substitutions for the token routing path. The rows kept here are the
arm in which **both the permutation and the prompt are the released learned ones**, which is to say
the unmodified released configuration. No routing substitution is applied in any row of these files.
`scripts/evaluate.py`, shipped in this repo, runs that same configuration directly and is the
supported way to regenerate them.

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
| `psnr_sd_db` | standard deviation across seeds, which is nonzero because inference is stochastic |
| `ssim_mean` | mean SSIM for this image across seeds |

## Summary

| quantity | value |
|---|---|
| PSNR, mean over seeds | 28.8236 dB |
| PSNR, standard deviation over seeds | 0.0042 dB |
| PSNR, range over seeds | 28.8180 to 28.8287 dB |
| PSNR reported in the paper | 28.79 dB |
| SSIM, mean over seeds | 0.92217 |
| SSIM reported in the paper | 0.934 |
| per-image PSNR range | 16.73 to 40.33 dB |
| mean per-image run-to-run standard deviation | 0.029 dB |

The `psnr_sd_db` column is the reason this reproduction reports a spread rather than a single
number. See [`../DETERMINISM.md`](../DETERMINISM.md).

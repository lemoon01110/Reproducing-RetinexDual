# Results

> Raw data for [**Reproducing RetinexDual**](../README.md). The report that interprets
> these files is [`../README.md`](../README.md).


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
| `ycbcr_mean` | 0.97731 | +0.05514 | +0.04331 |
| `msssim_y` | 0.97639 | +0.05421 | +0.04239 |
| `msssim_rgb` | 0.96132 | +0.03914 | +0.02732 |
| `matlab_y_float` | 0.94688 | +0.02471 | +0.01288 |
| `err_y_cly` | 0.94662 | +0.02444 | +0.01262 |
| `matlab_y` | 0.94656 | +0.02438 | +0.01256 |
| `matlab_y_border4` | 0.94652 | +0.02434 | +0.01252 |
| `matlab_rgb_float` | 0.92311 | +0.00094 | -0.01089 |
| `torch_rgb_01` | 0.92251 | +0.00034 | -0.01149 |
| `repo_rgb_mean` | 0.92218 | 0.00000 | -0.01182 |

No protocol lands within 0.0109 of the published 0.934. The band around it is 0.0234 wide and empty.
The file also carries three PSNR protocols, which pin the paper's colour space to RGB and thereby
rule out several explanations that would otherwise have to be tested separately.

These fall into two tight clusters, RGB-like around 0.922 and luma-like around 0.947, separated by
an empty band 0.0234 wide. The published 0.934 sits 46% of the way across that band, near neither.
`err_y_cly` is ERR's own protocol, which upstream credits for the benchmark, and it lands within
0.0002 of plain luma SSIM.

The `_float` rows score the model output before `tensor2img` rounds it to uint8. That accounts for
0.0009 of the 0.0118 gap, roughly 8%, so evaluating in float space is not the explanation.

No convention tested reaches 0.934, and the published value falls between the two commonest ones.
The gap is not explained. See README section 1.

Worth noting that `repo_rgb_mean` here (0.92218) agrees with the main evaluation (0.92217) to five
decimals, which is a useful check that the two independent code paths compute the same thing.

## JSON artifacts

The CSVs above back the reproduction. These three back the other measured tables in the report.
Before they existed, those tables had no source of truth and went stale repeatedly.
`scripts/check_report.py --tables` verifies the published tables against all three, and CI runs it.

| file | backs | produced by |
|---|---|---|
| `determinism.json` | the table in [`../DETERMINISM.md`](../DETERMINISM.md) and README section 4.2 | `scripts/determinism_audit.py --out` |
| `footprint_latency.json` | the latency column in README section 5 | `scripts/measure_footprint.py --mode latency --out` |
| `footprint_memory.json` | the working-set column in README section 5 | `scripts/measure_footprint.py --mode memory --out` |
| `ssim_protocols.json` | the SSIM protocol table in README section 1 | `scripts/ssim_protocol_probe.py --out` |
| `figure_data.json` | ties `assets/reproduction.png` to the CSVs it was drawn from | `scripts/make_figure.py --out-json` |
| `qualitative_picks.json` | records which images `assets/qualitative.png` shows, and why | `scripts/make_qualitative.py --out-json` |
| `timing.json` | the measured wall clock in README section 6 | `scripts/evaluate.py`, recorded in its summary |
| `checkpoint_audit.json` | the checkpoint and code counts in README section 4.1, and the SGM inertness proof | `scripts/verify_checkpoint.py --out` |
| `footprint_memory_expandable.json` | the allocator comparison in README section 5 | `measure_footprint.py --mode memory` under `expandable_segments` |
| `ceiling_bracket_default.json` | the OOM boundary in README section 5 | `scripts/find_ceiling.py --out` |
| `ceiling_bracket_expandable.json` | the same boundary under `expandable_segments` | `scripts/find_ceiling.py --out` |

How well each reproduces on a re-run, which is worth knowing before quoting one:

- `determinism.json` now reproduces **bit-exactly**, verified by re-running and comparing the whole
  file. Getting there needed a fixed seed *and* `cudnn.benchmark` off, because benchmark mode
  re-selects convolution algorithms per process. Earlier unseeded versions gave 0.008 to 0.135 and
  0.007 to 0.049 over the same images, so anything quoting that older span is quoting noise. See
  [`../DETERMINISM.md`](../DETERMINISM.md).
- `ssim_protocols.json` reproduces to five decimals on every protocol.
- `footprint_latency.json` moves by well under 1%, being a timing measurement.
- `footprint_memory.json`, `ceiling_bracket_*.json` reproduce to within about 0.01 GiB. Allocation is
  deterministic given the input size, so the residual is allocator bookkeeping rather than noise.

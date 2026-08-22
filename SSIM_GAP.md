# The SSIM Gap

> Part of [**Reproducing RetinexDual**](README.md), an independent reproduction of
> [arXiv:2508.04797](https://arxiv.org/abs/2508.04797) on the UHD-LL low-light benchmark.
> The paper's PSNR reproduces to 0.030 dB. Its SSIM does not, and this document is the
> attempt to find out why.
>
> Related: [`README.md`](README.md) section 1 for the summary,
> [`results/ssim_protocols.json`](results/ssim_protocols.json) for the raw measurements.

## Contents

- [The discrepancy](#the-discrepancy)
- [What was tested](#what-was-tested)
- [The result has a shape](#the-result-has-a-shape)
- [What this rules out](#what-this-rules-out)
- [What remains open](#what-remains-open)

## The discrepancy

**SSIM does not reproduce.** It comes out 0.0118 low, which is roughly 700 times the 0.000016
run-to-run spread, so this is a real difference and not noise.

The hypothesis worth testing was that this is a difference in how SSIM is defined rather than in
what the model produced, because **PSNR matches to 0.030 dB on the exact same output images**. A
model genuinely producing worse restorations would be expected to miss on both. "SSIM" also names a
family rather than one number: the repository's helper averages per-channel SSIM over RGB with a
MATLAB-style 11x11 Gaussian, while other common choices score luma only, crop a border, or use
scikit-image's 7x7 uniform default. Measured here, those choices differ from each other by 0.024,
which is twice the gap being explained, so the hypothesis was worth taking seriously.

[`scripts/ssim_protocol_probe.py`](scripts/ssim_protocol_probe.py) tests exactly this, scoring one
set of model outputs under nine conventions over all 150 images:

| protocol | SSIM | vs repo helper | vs paper |
|---|---|---|---|
| `ycbcr_mean` (per channel over Y, Cb, Cr) | 0.97731 | +0.05514 | +0.04331 |
| `matlab_y_float` (luma, before uint8 rounding) | 0.94688 | +0.02471 | +0.01288 |
| `err_y_cly` (**ERR's own protocol**, luma, replicate border) | 0.94662 | +0.02444 | +0.01262 |
| `matlab_y` (luma, 11x11 Gaussian) | 0.94656 | +0.02438 | **+0.01256** |
| `matlab_y_border4` (luma, 4px border crop) | 0.94652 | +0.02434 | +0.01252 |
| `matlab_rgb_float` (RGB, before uint8 rounding) | 0.92311 | +0.00094 | -0.01089 |
| `torch_rgb_01` (ERR's `cal_ssim.py`, [0,1] constants, zero padding) | 0.92251 | +0.00034 | -0.01149 |
| `repo_rgb_mean` (what this repo reports) | 0.92218 | 0.00000 | **-0.01182** |

All 150 images, one seed. SSIM's run-to-run spread is 0.000016, so a single seed is sufficient here
in a way it is not for PSNR. Backed by
[`results/ssim_protocols.json`](results/ssim_protocols.json), which `scripts/check_report.py`
verifies this table against.

**The hypothesis fails, and it fails in a specific and informative way.** The conventions do not
spread out smoothly. They fall into two tight clusters:

- **RGB-like**, three protocols spanning 0.92218 to 0.92311, a spread of 0.0009
- **luma-like**, four protocols spanning 0.94652 to 0.94688, a spread of 0.0004

Between them is an empty band **0.0234 wide**, and the published 0.934 sits almost exactly in the
middle of it, 46% of the way across. It is 0.0109 above the highest RGB result and 0.0125 below the
lowest luma one. So this is not a case of having guessed the wrong convention from a continuum.
Whatever produced 0.934 is not any member of either family.

Two of these deserve individual mention.

**ERR's protocol was the strongest candidate and it does not match.** The upstream README credits
[ERR](https://github.com/NJU-PCALab/ERR) for "the UHD restoration benchmarks and references", and
RetinexDual's results table sits alongside ERR's. Its `calculate_ssim` defaults to `crop_border=1`
and `test_y_channel=True`, and its `_ssim_cly` uses `BORDER_REPLICATE` without cropping the filtered
map, where the RetinexDual helper crops to valid. Implemented exactly, it lands within **0.0002** of
plain luma SSIM. At 4K the `[5:-5]` crop discards about 0.5% of pixels, so border handling cannot
move a dataset mean meaningfully. ERR ships a second, different SSIM in `basicsr/models/cal_ssim.py`
using `[0,1]` constants and zero padding, and that one lands in the RGB cluster instead. Neither
reaches 0.934.

**The paper's own table is inconsistent with the RGB reading.** It lists UHDFormer at 27.11 dB /
0.927 and ERR at 27.57 dB / 0.932. This reproduction measures 0.92218 at 28.82 dB, which would put
RetinexDual *below* UHDFormer on SSIM while being 1.7 dB better on PSNR. So the published column is
probably not per-channel RGB. But it is not luma either, which would read 0.947 here. That is the
part I cannot resolve.

What this rules out with reasonable confidence: the colour space, the border handling, the uint8
rounding, the SSIM constants, and ERR's implementation specifically. What it cannot rule out from
here: a different evaluation split, a downsampling step before scoring, a checkpoint other than the
released one, or an implementation I have not thought of.

The checkpoint is worth one note. It holds 4,725,531 parameters, matching the 4.726M the paper
states, so the released weights do appear to be the ones measured. Together with PSNR agreeing to
0.030 dB, that makes a model difference an unlikely explanation and keeps the question pointed at
the metric.

For completeness, since a reader will notice it: 0.934 sits near the midpoint of the two clusters. I
have no mechanism that would make an average of two metric families meaningful, and with nine
protocols measured the midpoint of some pair can be made to land almost anywhere, so I record it as
a coincidence rather than a finding.

**This is the one question I would put to the authors.** I am not claiming the paper is wrong.


Regenerate the table with:

```bash
python scripts/ssim_protocol_probe.py --repo ~/RetinexDual --data <UHD-LL testing_set>
```

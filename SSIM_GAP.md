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
- [Ten conventions, measured](#ten-conventions-measured)
- [The result has a shape](#the-result-has-a-shape)
- [ERR's protocol, the strongest candidate](#errs-protocol-the-strongest-candidate)
- [The colour space is pinned by PSNR](#the-colour-space-is-pinned-by-psnr)
- [Aggregation, and a tempting near-miss](#aggregation-and-a-tempting-near-miss)
- [What this rules out, and what remains open](#what-this-rules-out-and-what-remains-open)

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

## Ten conventions, measured

[`scripts/ssim_protocol_probe.py`](scripts/ssim_protocol_probe.py) scores one set of model outputs
under ten SSIM conventions over all 150 images:

| protocol | SSIM | vs repo helper | vs paper |
|---|---|---|---|
| `ycbcr_mean` (per channel over Y, Cb, Cr) | 0.97731 | +0.05514 | +0.04331 |
| `msssim_y` (multi-scale, luma) | 0.97639 | +0.05421 | +0.04239 |
| `msssim_rgb` (multi-scale, per channel) | 0.96132 | +0.03914 | +0.02732 |
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

## The result has a shape

**The hypothesis fails, and it fails in a specific and informative way.** The conventions do not
spread out smoothly. Three sit below the published value and seven above it, and the nearest on
either side leaves a gap:

- highest result below 0.934 is `matlab_rgb_float` at **0.92311**, short by 0.0109
- lowest result above 0.934 is `matlab_y_border4` at **0.94652**, over by 0.0125

**No protocol lands within 0.0109 of the published value**, and the band around it is 0.0234 wide
and empty. Underneath, the ten split into two tight families: RGB-like from 0.92218 to 0.92311, a
spread of 0.0009, and luma-like from 0.94652 to 0.94688, a spread of 0.0004. Everything else scores
higher still.

This matters because it changes what kind of answer is possible. If the conventions formed a
continuum, missing by 0.012 would just mean I had not found the right one yet. They do not. Whatever
produced 0.934 is not a member of either family, and the two obvious ways to leave a family, scoring
multi-scale or scoring all three YCbCr channels, both overshoot rather than landing in the gap.

## ERR's protocol, the strongest candidate

**It does not match.** The upstream README credits
[ERR](https://github.com/NJU-PCALab/ERR) for "the UHD restoration benchmarks and references", and
RetinexDual's results table sits alongside ERR's. Its `calculate_ssim` defaults to `crop_border=1`
and `test_y_channel=True`, and its `_ssim_cly` uses `BORDER_REPLICATE` without cropping the filtered
map, where the RetinexDual helper crops to valid. Implemented exactly, it lands within **0.0002** of
plain luma SSIM. At 4K the `[5:-5]` crop discards about 0.5% of pixels, so border handling cannot
move a dataset mean meaningfully. ERR ships a second, different SSIM in `basicsr/models/cal_ssim.py`
using `[0,1]` constants and zero padding, and that one lands in the RGB cluster instead. Neither
reaches 0.934.

### The paper's own table is inconsistent with the RGB reading

It lists UHDFormer at 27.11 dB /
0.927 and ERR at 27.57 dB / 0.932. This reproduction measures 0.92218 at 28.82 dB, which would put
RetinexDual *below* UHDFormer on SSIM while being 1.7 dB better on PSNR. So the published column is
probably not per-channel RGB. But it is not luma either, which would read 0.947 here. That is the
part I cannot resolve.

## The colour space is pinned by PSNR

Everything above works from the SSIM side. But **PSNR reproduced**, and that is itself evidence. Any
protocol choice that would move both metrics is already excluded by that agreement, so the
explanation has to be something that moves SSIM while leaving PSNR alone.

Measuring PSNR under the same conventions, over all 150 images:

| protocol | PSNR (dB) | vs paper 28.79 |
|---|---|---|
| `psnr_y_border1` | 31.4084 | +2.6184 |
| `psnr_y` (luma) | 31.4077 | +2.6177 |
| `psnr_rgb` (all three channels) | **28.8191** | **+0.0291** |

Luma PSNR runs **2.62 dB above** RGB. Had the paper computed PSNR on luma it would report about
31.4, not 28.79. **So the published PSNR is per-channel RGB.** That is not an inference from
convention, it is forced by a 2.6 dB separation against a 0.03 dB agreement.

Putting the two together:

- The paper computed PSNR in RGB. In RGB, this reproduction measures SSIM **0.92218**.
- Had it computed SSIM on luma instead, that would read **0.94656** here.
- It reports **0.934**, which is neither.

**No single colour-space choice produces both published numbers.** RGB explains the PSNR and
undershoots the SSIM by 0.012. Luma would explain neither, being 2.6 dB out on PSNR and 0.013 over
on SSIM. Mixing the two, RGB for PSNR and luma for SSIM, is unusual but not unheard of, and it still
does not land on 0.934.

This also disposes of one candidate I had left open. A downsampling step before scoring would move
PSNR as well, and PSNR agrees to 0.03 dB, so there is no such step.

## Aggregation, and a tempting near-miss

Everything so far asks what SSIM *means*. A separate question is how the 150 per-image scores were
combined. That matters here because the distribution is badly left-skewed: the mean is 0.92217 but
the median is 0.93220, because a handful of hard images drag the average down. The worst image
scores 0.71344.

Several ordinary aggregation choices land close to the published value, and one lands almost exactly
on it:

| aggregation | SSIM | vs paper | PSNR | vs paper |
|---|---|---|---|---|
| all 150 images | 0.92217 | -0.01183 | 28.8199 | +0.0299 |
| excluding worst 5% by SSIM | 0.92918 | -0.00482 | 29.1412 | +0.3512 |
| **excluding worst 10% by SSIM** | **0.93394** | **-0.00006** | **29.5636** | **+0.7736** |
| excluding worst 20% by SSIM | 0.94167 | +0.00767 | 30.0299 | +1.2399 |
| trimmed 10% each end | 0.92888 | -0.00512 | 29.0954 | +0.3054 |
| trimmed 20% each end | 0.93176 | -0.00224 | 29.1231 | +0.3331 |

**Excluding the worst 10% of images gives 0.93394 against a published 0.934, a difference of
0.00006.** That looks like a solved mystery, and it is not one.

The test that settles it is applying each aggregation to *both* metrics, dropping the same images
from each. The same exclusion that lands SSIM on 0.934 lifts PSNR to 29.56, which is **0.77 dB above
the published 28.79**, and PSNR reproduces exactly over the full 150. The images dragging SSIM down
are the same images dragging PSNR down. Pearson correlation between the two over 150 images is
**0.60**, and all fifteen of the worst-SSIM images also sit below median PSNR. So no exclusion can
repair one metric without breaking the other.

This is also a check on my own pipeline rather than the paper's. If those low-scoring images were
being mishandled here, my mean would be wrong and the published figure right. They are not: they
score badly on both metrics together, which is what genuinely difficult images look like, not what a
pairing or preprocessing bug looks like.

Reproduce with `python scripts/test_aggregation.py`, which needs no GPU and reads the committed
per-image CSV.

### Are the implementations themselves right

Everything above depends on my SSIM code being correct. If `ssim_matlab` were subtly wrong the
clustering would be an artifact of this repository rather than a fact about SSIM, so
`scripts/test_metrics.py` pins each implementation against an independent reference and CI runs it.
The MATLAB-style estimator agrees with scikit-image configured with a matched kernel to **3e-14**,
the colour transforms reproduce the BT.601 coefficients exactly, and every implementation returns
exactly 1.0 on identical inputs. The suite was checked against a deliberately corrupted copy to
confirm it fails rather than passing vacuously.

I record the 0.00006 near-miss because a reader will find it, and because it is a good illustration
of why a single matching number is not evidence. Test enough aggregations and one will land.

## What this rules out, and what remains open

What this rules out with reasonable confidence: the colour space, the border handling, the uint8
rounding, the SSIM constants, and ERR's implementation specifically. Also ruled out, both by the joint PSNR
constraint: a downsampling step before scoring, and any exclusion, trimming or subset choice.

What remains open: an SSIM implementation not among the nine tested, a checkpoint other than the
released one, or a test split that differs in a way that happens to leave PSNR unchanged while
moving SSIM by 0.012. That last one is possible but would be a coincidence.

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

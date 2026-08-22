# Memory, Latency and the Ceiling

> Part of [**Reproducing RetinexDual**](README.md), an independent reproduction of
> [arXiv:2508.04797](https://arxiv.org/abs/2508.04797) on the UHD-LL low-light benchmark.
> What one forward pass costs, what card you need, and where it stops working.
>
> Related: [`ENVIRONMENT.md`](ENVIRONMENT.md) for versions and hardware,
> [`results/footprint_memory.json`](results/footprint_memory.json) and
> [`results/footprint_latency.json`](results/footprint_latency.json) for the raw measurements.

## Contents

- [The sweep](#the-sweep)
- [Peak allocation is linear in pixel count](#peak-allocation-is-linear-in-pixel-count)
- [What card do you actually need](#what-card-do-you-actually-need)
- [Where it stops working](#where-it-stops-working)

## The sweep

Measured by [`scripts/measure_footprint.py`](scripts/measure_footprint.py), 5 groups of 5 iterations
after 5 warmup passes, fp32:

| Input | Padded | Mpix | CUDA-event median | Working set | GiB per Mpix | Reserved |
|---|---|---|---|---|---|---|
| 1280x768 | 1280x768 | 0.98 | 102.21 ms | 1.31 GiB | 1.337 | 9.37 GiB |
| 1920x1088 | 1920x1152 | 2.21 | 287.31 ms | 2.92 GiB | 1.320 | 17.68 GiB |
| 2560x1408 | 2560x1408 | 3.60 | 548.25 ms | 4.73 GiB | 1.313 | 18.20 GiB |
| **3840x2160** | **3840x2176** | 8.36 | **1347.73 ms** | **10.94 GiB** | 1.309 | **19.54 GiB** |

Latency and working set come from separate invocations (`--mode latency` and `--mode memory`)
because they need opposite settings of `cudnn.benchmark` and contaminate each other in one
process. Both are backed by [`results/footprint_latency.json`](results/footprint_latency.json) and
[`results/footprint_memory.json`](results/footprint_memory.json), which `scripts/check_report.py`
verifies this table against. Running both together cost about 1.5% on the 4K timing, which is
larger than the group-to-group spread, so the script warns when you do that.

Wall clock and CUDA-event timings agree to within 0.02% at every resolution, and group-to-group
spread at 4K is under 1 ms out of 1348 ms. An earlier, separately written harness measured the same
4K forward at 1347.89 ms, within 0.02% of the figure above.

The harness is sanity-checked first, on the principle that something which cannot recover a known
cost should not be trusted on an unknown one: an 8192-cubed fp16 matmul reaches 161.4 TFLOPS, in
line with this card's specification.

One caveat found while measuring, and the reason working set and reserved are reported separately.
With `cudnn.benchmark = True`, peak *allocated* came out non-monotonic in resolution (15.64 GiB at
1920x1152 against 15.28 GiB at 3840x2176), which cannot be a real working set. cuDNN's algorithm
search allocates trial workspaces that land in the peak statistic, and it declines the larger ones
once they stop fitting. Measured with benchmark off, the same sweep is linear to within 2%.

## Peak allocation is linear in pixel count

Fitting every memory measurement in
[`results/`](results/), 19 points spanning 0.98 to 16.91 Mpix, a 17x range:

```
peak allocated (GiB) = 1.305 x Mpix + 0.029      R2 = 1.000000
```

The largest residual is 3 MB. That is a tighter fit than the measurements deserve to produce and it
is worth saying why it holds so exactly: this network has no data-dependent allocation, so the
working set is a fixed function of input size.

**The intercept is the model itself.** 0.029 GiB is 31 MB, against 19 MB of fp32 weights
(4,747,035 parameters) plus optimiser-free buffers and CUDA context, so the constant term is what
sits on the card before any pixel arrives. The slope is what each megapixel costs.

The same constant was measured on a different GPU during earlier work on this model, an 8 GB RTX
5070 Laptop at compute capability 12.0, also at about 1.31 GiB per Mpix. Two cards and two
architectures agreeing suggests this is a property of the model rather than of one machine.
**That second measurement is not reproducible from this repository**, since it predates it and the
machine is not the one used here, so treat it as supporting context rather than as one of the
committed results.

`scripts/check_report.py --scaling` refits this line from the artifacts on every CI run and fails if
linearity breaks, so a future torch release that changes allocation behaviour cannot quietly
falsify the claim.

## What card do you actually need

The working set at 4K is 10.94 GiB, but the allocator reserves far more than it allocates, and
reserved is what decides whether the run fits. **One environment variable moves the answer by 8
GiB:**

| | reserved at 3840x2176 | ratio to working set | smallest card (derived) |
|---|---|---|---|
| default allocator | 19.67 GiB | 1.80x | 24 GB |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | **11.64 GiB** | **1.06x** | **16 GB** |

The reserved figures are measured. **The smallest-card column is derived from them**, by comparing
against usable capacity, and is not a measurement: the only card tested here is a 24 GB 4090. A
16 GB card should hold 11.64 GiB comfortably, but I have not run it.

Both rows measured with `cudnn.benchmark` off, so the only variable is the allocator. With benchmark
on, as the reproduction actually runs, the default allocator reserves 19.54 GiB, which is within
0.7% of the 19.67 above. The allocator is what moves this number, not the benchmark flag.

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

The default allocator suballocates from fixed-size segments and fragments badly at this working set,
holding 1.8x what it hands out. Expandable segments grow in place instead, bringing reserved to
within 6% of allocated across the whole sweep. **Whole-image 4K inference therefore fits on a 16 GB
card**, which the earlier "about 20 GiB" figure in this report wrongly ruled out. A 12 GB card is
marginal at 11.64 GiB reserved and is not something I have tested.

Smaller inputs, with expandable segments, all measured: 2560x1408 reserves 5.04 GiB, 1920x1088
reserves 3.11, and 1280x768 reserves 1.41. From those, an 8 GB card should handle everything up to
about 3.6 Mpix, which is again a derivation rather than a measurement.

Backed by [`results/footprint_memory.json`](results/footprint_memory.json) and
[`results/footprint_memory_expandable.json`](results/footprint_memory_expandable.json).

## Where it stops working

Stepping up in 128-row increments until it fails, rather than extrapolating:

| allocator | largest input that runs | allocated | reserved | first failure |
|---|---|---|---|---|
| default | 5248x2944, 15.45 Mpix | 20.19 GiB | 22.03 GiB | 16.91 Mpix |
| `expandable_segments:True` | **5504x3072, 16.91 Mpix** | 22.09 GiB | 22.44 GiB | 18.02 Mpix |

**An earlier version of this section said expandable segments do not move the ceiling. That was
wrong.** They raise it by about 9% in pixel count, from between 15.45 and 16.91 Mpix to between
16.91 and 18.02. The error came from probing at 20.36 and 33.18 Mpix, which sit above *both*
ceilings, so both returned OOM and the difference was invisible. Two points that agree tell you
nothing if both are outside the range where the effect lives. `scripts/find_ceiling.py` now walks
the boundary in 128-row steps for exactly this reason.

The mechanism is visible in the reserved column. Under the default allocator, reserved pins at
roughly 22 GiB from 11.8 Mpix upward and barely moves (22.46, 22.03, 22.30, 22.03), because it is
already saturating the card with fragmented segments. Under expandable segments reserved tracks
allocated within **0.35 GiB** at the ceiling against **1.84 GiB** for the default. That recovered
headroom is what buys the extra resolution.

So the corrected statement: **the allocator setting matters twice.** At 4K it saves 8 GiB of
reserved memory and moves the requirement from a 24 GB card to a 16 GB one. At the ceiling it buys
about one more resolution step. What it cannot do is help once the working set alone exceeds the
card, which at 1.305 GiB per Mpix happens around 18 Mpix on 24 GB.

Backed by [`results/ceiling_bracket_default.json`](results/ceiling_bracket_default.json) and
[`results/ceiling_bracket_expandable.json`](results/ceiling_bracket_expandable.json).

---


#!/usr/bin/env python3
"""Guard the report against rotting.

The substantive claims here need a 4K GPU and a licence-restricted dataset, so
they cannot be checked automatically. These three things can be:

  --links    every relative markdown link points at a file that exists
  --numbers  headline figures in the prose match the committed CSVs
  --style    no em dashes, en dashes or semicolons in prose

The numbers check is the one that matters. This report was edited many times and
each edit was a chance to leave a stale figure in one file while updating
another, which happened twice during writing.

Usage: python scripts/check_report.py [--links] [--numbers] [--style]
       python scripts/check_report.py            # all three
"""
import argparse
import csv
import pathlib
import re
import statistics as st
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = ["README.md", "ENVIRONMENT.md", "DETERMINISM.md", "SSIM_GAP.md",
        "FOOTPRINT.md", "results/README.md"]

# Lines that legitimately quote a superseded figure while saying so.
SUPERSEDED_MARKERS = (
    "earlier version", "superseded", "replaces", "against 0.210",
    "corrected", "overstated", "non-monotonic", "1368.87 ms against",
    "earlier, separately written harness", "separate earlier harness",
    "40-image pilot", "subset", "coincidence",
)


def read_docs():
    for name in DOCS:
        p = ROOT / name
        if p.exists():
            yield name, p.read_text()


def check_style():
    bad = []
    for name, text in read_docs():
        for i, line in enumerate(text.splitlines(), 1):
            for ch, label in (("—", "em dash"), ("–", "en dash"), (";", "semicolon")):
                if ch in line:
                    bad.append(f"{name}:{i} contains {label}: {line.strip()[:70]}")
    for b in bad:
        print(f"  FAIL {b}")
    print(f"style: {'FAIL' if bad else 'ok'} ({len(bad)} issues)")
    return not bad


def check_structure():
    """Catch the damage that moving prose between documents does.

    Splitting sections into their own files twice left a sentence cut in half,
    with a heading landing directly after a line of body text and no blank line
    between. Markdown renders that as a run-on and it reads as a truncation,
    which is exactly what it was. Both are cheap to detect.
    """
    bad = []
    for name, text in read_docs():
        lines = text.splitlines()
        # Comments inside fenced code look exactly like headings, so track fences.
        fenced, in_code = set(), False
        for i, line in enumerate(lines):
            if line.lstrip().startswith("```"):
                in_code = not in_code
                fenced.add(i)
            elif in_code:
                fenced.add(i)
        for i, line in enumerate(lines):
            if i in fenced or not line.startswith("#"):
                continue
            if i == 0:
                continue
            prev = lines[i - 1]
            if prev.strip():
                bad.append(f"{name}:{i + 1} heading directly follows body text, "
                           f"which usually means a sentence was cut: {prev.strip()[:60]}")
            elif i >= 2:
                # A paragraph ending without terminal punctuation right before a
                # heading is the signature of a truncated extraction.
                before = next((l for j, l in reversed(list(enumerate(lines[:i - 1])))
                                if l.strip() and j not in fenced), "")
                b = before.strip()
                if (b and not b.endswith((".", ":", "|", "`", ")", "]", "-", '"'))
                        and not b.startswith(("|", "#", ">", "-", "*"))):
                    bad.append(f"{name}:{i + 1} paragraph before this heading ends without "
                               f"punctuation: {b[-60:]}")
    for b in bad:
        print(f"  FAIL {b}")
    print(f"structure: {'FAIL' if bad else 'ok'} ({len(bad)} issues)")
    return not bad


def check_links():
    bad = []
    for name, text in read_docs():
        base = (ROOT / name).parent
        for m in re.finditer(r"\]\(([^)]+)\)", text):
            target = m.group(1).split("#")[0]
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            if not (base / target).exists():
                bad.append(f"{name}: broken link -> {target}")
    for b in bad:
        print(f"  FAIL {b}")
    print(f"links: {'FAIL' if bad else 'ok'} ({len(bad)} issues)")
    return not bad


def check_numbers():
    seeds_csv = ROOT / "results" / "reproduction_seeds.csv"
    per_img_csv = ROOT / "results" / "reproduction_per_image.csv"
    if not seeds_csv.exists():
        print("numbers: FAIL (results/reproduction_seeds.csv missing)")
        return False

    rows = list(csv.DictReader(seeds_csv.open()))
    psnr = [float(r["psnr_db"]) for r in rows]
    ssim = [float(r["ssim"]) for r in rows if r["ssim"]]
    truth = {
        "psnr_mean": st.mean(psnr),
        "psnr_sd": st.stdev(psnr) if len(psnr) > 1 else 0.0,
        "psnr_min": min(psnr),
        "psnr_max": max(psnr),
        "ssim_mean": st.mean(ssim) if ssim else None,
        "n_seeds": len(rows),
        "n_images": int(rows[0]["n_images"]),
    }

    per = list(csv.DictReader(per_img_csv.open())) if per_img_csv.exists() else []
    if per:
        pim = [float(r["psnr_mean_db"]) for r in per]
        truth["per_image_min"] = min(pim)
        truth["per_image_max"] = max(pim)
        truth["n_per_image"] = len(per)

    # Figures the prose is allowed to state, and what they must equal.
    expected = {
        f"{truth['psnr_mean']:.4f}": "dataset mean PSNR",
        f"{truth['psnr_sd']:.4f}": "PSNR standard deviation across seeds",
        f"{truth['psnr_min']:.4f}": "lowest seed PSNR",
        f"{truth['psnr_max']:.4f}": "highest seed PSNR",
    }

    readme = (ROOT / "README.md").read_text()
    problems = []

    for value, label in expected.items():
        if value not in readme:
            problems.append(f"README does not state the {label} ({value})")

    # Any 28.8xxx in the prose must be one of: a real seed value, the dataset
    # mean, or a line that flags itself as quoting a superseded figure.
    allowed = {f"{v:.4f}" for v in psnr} | {
        f"{truth['psnr_mean']:.4f}", f"{truth['psnr_min']:.4f}", f"{truth['psnr_max']:.4f}"
    }
    for name, text in read_docs():
        for i, line in enumerate(text.splitlines(), 1):
            if any(mark in line.lower() for mark in SUPERSEDED_MARKERS):
                continue
            for m in re.finditer(r"\b28\.8\d{2,4}\b", line):
                tok = m.group(0)
                if len(tok.split(".")[1]) < 4:
                    continue
                if tok not in allowed:
                    problems.append(f"{name}:{i} states {tok}, which is not in "
                                    f"reproduction_seeds.csv and is not marked superseded")

    if truth["ssim_mean"] is not None:
        s = f"{truth['ssim_mean']:.5f}"
        if s not in readme:
            problems.append(f"README does not state the mean SSIM ({s})")

    if per and f"{truth['n_per_image']}" not in readme:
        problems.append(f"README does not state the image count ({truth['n_per_image']})")

    for p in problems:
        print(f"  FAIL {p}")
    print(f"numbers: {'FAIL' if problems else 'ok'} "
          f"(PSNR {truth['psnr_mean']:.4f} +/- {truth['psnr_sd']:.4f} over "
          f"{truth['n_seeds']} seeds x {truth['n_images']} images)")
    return not problems


def no_contradiction(pattern, correct, label, problems, extra_allowed=()):
    """Every occurrence of a figure in this slot must equal the artifact value.

    Checking only that the correct value appears *somewhere* is not enough. These
    figures are restated across several documents, so a stale copy in one of them
    passes a presence check while contradicting the artifact. This scans for the
    shape of the number and requires each hit to be either the current value, an
    explicitly allowed alternative, or on a line that flags itself as quoting a
    superseded figure.
    """
    allowed = {correct} | set(extra_allowed)
    seen = False
    for name, text in read_docs():
        lines = text.splitlines()
        for i, line in enumerate(lines, 1):
            # A superseded-figure marker often sits a line or two above the number
            # it qualifies, because prose wraps. Check the surrounding paragraph
            # rather than the single line, or legitimate cross-references to an
            # earlier measurement get flagged as contradictions.
            lo = max(0, i - 4)
            hi = min(len(lines), i + 2)
            window = " ".join(lines[lo:hi]).lower()
            if any(m in window for m in SUPERSEDED_MARKERS):
                continue
            for m in re.finditer(pattern, line):
                seen = True
                if m.group(0) not in allowed:
                    problems.append(f"{name}:{i} states {label} as {m.group(0)}, "
                                    f"but the artifact says {correct}")
    if not seen:
        problems.append(f"no document states {label} ({correct})")


def check_tables():
    """Verify the determinism and footprint tables against their JSON artifacts.

    These two went stale three separate times while this report was being
    written, because unlike the reproduction they had no committed source of
    truth to check against. Now they do.
    """
    import json
    problems = []
    readme = (ROOT / "README.md").read_text()
    det_md = (ROOT / "DETERMINISM.md").read_text()
    # Tables move between documents as the report is restructured. Check the
    # union rather than pinning a figure to one file.
    all_docs = "\n".join(t for _, t in read_docs())

    det_p = ROOT / "results" / "determinism.json"
    if not det_p.exists():
        problems.append("results/determinism.json missing")
    else:
        d = json.loads(det_p.read_text())
        a = d["aggregate"]["A"]
        span = f"{a['max_lo']:.3f} to {a['max_hi']:.3f}"
        psnr = f"{a['psnr_lo']:.2f} to {a['psnr_hi']:.2f}"
        for doc, name in ((readme, "README.md"), (det_md, "DETERMINISM.md")):
            if span not in doc:
                problems.append(f"{name} does not state the max-delta span '{span}'")
            if psnr not in doc:
                problems.append(f"{name} does not state the pairwise PSNR span '{psnr}'")
        if d["aggregate"]["B"]["identical"] is not True:
            problems.append("determinism.json says RNG replay is NOT bit-identical, "
                            "which contradicts the whole argument")
        for img, per in d["per_image"].items():
            row = f"`{img}` | {per['A']['max_hi']:.4f}"
            if row not in det_md:
                problems.append(f"DETERMINISM.md per-image row for {img} does not match the artifact")

    for tag, fname in (("latency", "footprint_latency.json"), ("memory", "footprint_memory.json")):
        fp = ROOT / "results" / fname
        if not fp.exists():
            problems.append(f"results/{fname} missing")
            continue
        f = json.loads(fp.read_text())
        big = [r for r in f["rows"] if not r["oom"]][-1]
        if tag == "latency":
            v = f"{big['event_median']:.2f} ms"
            r = f"{big['peak_reserved_bench']:.2f} GiB"
            # Checked across the union of documents rather than pinned to one.
            # These figures are restated in several places and have moved between
            # them as the report was restructured, so following the content is
            # the right behaviour and forbidding the move is not.
            no_contradiction(r"\b1\d{3}\.\d{2} ms\b", v, "the 4K latency", problems)
            no_contradiction(r"\b19\.\d{2} GiB\b", r, "the 4K reserved memory", problems,
                             extra_allowed={"19.67 GiB"})
        else:
            v = f"{big['peak_alloc']:.2f} GiB"
            no_contradiction(r"\b10\.\d{2} GiB\b", v, "the 4K working set", problems)

    ssim_p = ROOT / "results" / "ssim_protocols.json"
    if not ssim_p.exists():
        problems.append("results/ssim_protocols.json missing")
    else:
        sj = json.loads(ssim_p.read_text())
        for proto, val in sj["protocols"].items():
            if proto.startswith("psnr_"):
                continue          # PSNR rows are checked against their own table
            v = f"{val:.5f}"
            if v not in all_docs:
                problems.append(f"no document states SSIM {v} for protocol '{proto}'")
        # PSNR under each convention is the sharpest constraint in SSIM_GAP.md,
        # so it gets the same treatment as the SSIM rows.
        for proto, val in sj["protocols"].items():
            if not proto.startswith("psnr_"):
                continue
            if f"{val:.4f}" not in all_docs:
                problems.append(f"no document states PSNR {val:.4f} for protocol '{proto}'")

        # Checkpoint audit. These numbers were quoted from notes for a long time
        # before anything re-measured them.
        ca = ROOT / "results" / "checkpoint_audit.json"
        if not ca.exists():
            problems.append("results/checkpoint_audit.json missing")
        else:
            c = json.loads(ca.read_text())
            for key, fmt in (("code_tensors", "{:,}"), ("code_params", "{:,}"),
                             ("ckpt_tensors", "{:,}"), ("ckpt_params", "{:,}"),
                             ("missing_keys", "{}"), ("sgm_modules", "{}")):
                v = fmt.format(c[key])
                if v not in all_docs:
                    problems.append(f"no document states {key} = {v}")
            if not c["verdict_inert"]:
                problems.append("checkpoint_audit.json says the SGM modules are NOT inert, "
                                "which contradicts README section 4.1")
            if c["sgm_forward_calls"] != 0:
                problems.append(f"SGM forward calls is {c['sgm_forward_calls']}, not 0")

        # Selective-scan latency, which was quoted at 29.2 ms from notes and is
        # actually 17.08 ms. Contradiction-checked so a stale copy cannot return.
        st_p = ROOT / "results" / "scan_timing.json"
        if not st_p.exists():
            problems.append("results/scan_timing.json missing")
        else:
            t = json.loads(st_p.read_text())
            no_contradiction(r"\b\d{2}\.\d{2} ms\b", f"{t['scan_median_ms']:.2f} ms",
                             "the selective-scan latency", problems)
            no_contradiction(r"\b9\.\d{2} GB\b", f"{t['ref_intermediate_gb']:.2f} GB",
                             "the reference intermediate size", problems)

        # glibc floors, measured from the wheels rather than quoted.
        ga = ROOT / "results" / "glibc_audit.json"
        if not ga.exists():
            problems.append("results/glibc_audit.json missing")
        else:
            g = json.loads(ga.read_text())
            if not g["claim_holds"]:
                problems.append("glibc_audit.json says the torch2.7/torch2.6 split does not hold, "
                                "which contradicts README section 2.2")
            for w in g["wheels"]:
                frag = f"GLIBC_{w['requires_glibc']}"
                if frag not in all_docs:
                    problems.append(f"no document states {frag} for {w['package']} {w['version']}")

        if sj["n_images"] < 150:
            problems.append(f"ssim_protocols.json is a partial run ({sj['n_images']} images). "
                            f"The published table must come from the full set.")

    # The committed reproduction.png could be stale relative to the CSVs and
    # nothing would notice, because CI re-renders to a temporary file and throws
    # it away. Comparing PNG bytes across matplotlib versions is not workable,
    # so make_figure.py records the values it plotted and those are checked here.
    fig_p = ROOT / "results" / "figure_data.json"
    seeds_p = ROOT / "results" / "reproduction_seeds.csv"
    per_p = ROOT / "results" / "reproduction_per_image.csv"
    if not fig_p.exists():
        problems.append("results/figure_data.json missing, so the committed figure "
                        "cannot be tied to the committed data")
    elif seeds_p.exists() and per_p.exists():
        fj = json.loads(fig_p.read_text())
        live = [float(r["psnr_db"]) for r in csv.DictReader(seeds_p.open())]
        if [round(v, 6) for v in fj["seed_psnr"]] != [round(v, 6) for v in live]:
            problems.append("figure_data.json does not match reproduction_seeds.csv, so "
                            "assets/reproduction.png was rendered from different data than "
                            "the repository now commits. Re-run scripts/make_figure.py.")
        if fj["n_images"] != len(list(csv.DictReader(per_p.open()))):
            problems.append("figure_data.json image count does not match "
                            "reproduction_per_image.csv")

    # Wall clock was quoted from impression rather than measurement for several
    # revisions, and was wrong by a factor of three. It now has an artifact.
    t_p = ROOT / "results" / "timing.json"
    if not t_p.exists():
        problems.append("results/timing.json missing")
    else:
        tj = json.loads(t_p.read_text())
        per_seed = f"{sum(tj['seed_times_s']) / len(tj['seed_times_s']) / 60:.1f} minutes per seed"
        total = f"{tj['wall_clock_s'] / 60:.1f} minutes for the full five"
        for frag in (per_seed, total, f"{tj['s_per_image']:.2f} s per image"):
            if frag not in readme:
                problems.append(f"README does not state '{frag}' from timing.json")

    # The qualitative figure needs a GPU to regenerate, so CI cannot rebuild it.
    # It can still check that the images shown were selected by the stated rule,
    # which is the part a reader would be right to be sceptical about.
    picks_p = ROOT / "results" / "qualitative_picks.json"
    per_img_csv = ROOT / "results" / "reproduction_per_image.csv"
    if picks_p.exists() and per_img_csv.exists():
        picks = json.loads(picks_p.read_text())
        rows = [r for r in csv.DictReader(per_img_csv.open()) if r["psnr_mean_db"]]
        rows.sort(key=lambda r: float(r["psnr_mean_db"]))
        want = {"worst": rows[0], "median": rows[len(rows) // 2], "best": rows[-1]}
        for pick in picks["picks"]:
            exp = want.get(pick["rank"])
            if exp is None:
                problems.append(f"qualitative_picks.json has unknown rank '{pick['rank']}'")
            elif exp["image"] != pick["image"]:
                problems.append(
                    f"qualitative figure claims {pick['rank']} is {pick['image']}, but the "
                    f"per-image CSV ranks {exp['image']} there. The selection is supposed to "
                    f"be by rank, not by eye.")
        for pick in picks["picks"]:
            v = f"{pick['psnr_mean_db']:.2f} dB"
            if pick["rank"] == "worst" and v.replace(" dB", "") not in readme:
                problems.append(f"README does not state the worst-case PSNR '{v}'")
    elif not picks_p.exists():
        problems.append("results/qualitative_picks.json missing")

    for p in problems:
        print(f"  FAIL {p}")
    print(f"tables: {'FAIL' if problems else 'ok'} "
          f"(determinism, footprint, SSIM and figure selection checked against results/*.json)")
    return not problems


def check_scaling():
    """Fit the memory scaling law across every committed measurement.

    README section 5 leans on peak allocation being linear in pixel count. That
    claim had no guard, so a future torch release changing allocation behaviour
    would quietly falsify it with nothing to notice. This refits the line from
    the artifacts on every CI run and fails if it stops being linear.
    """
    import json
    problems = []
    pts = set()
    for fp in sorted((ROOT / "results").glob("*.json")):
        try:
            data = json.loads(fp.read_text())
        except (ValueError, OSError):
            continue
        for r in data.get("rows", []):
            if r.get("oom") or "peak_alloc" not in r:
                continue
            # measure_footprint records the padded shape, find_ceiling records
            # Mpix directly. Accept either so every memory measurement in the
            # repository contributes to the fit.
            if "padded" in r:
                mpix = (r["padded"][2] * r["padded"][3]) / 1e6
            elif "mpix" in r:
                mpix = r["mpix"]
            else:
                continue
            pts.add((round(mpix, 4), round(r["peak_alloc"], 4)))

    if len(pts) < 4:
        problems.append(f"only {len(pts)} memory points available, need at least 4 to fit")
    else:
        pts = sorted(pts)
        n = len(pts)
        sx = sum(p[0] for p in pts)
        sy = sum(p[1] for p in pts)
        sxx = sum(p[0] ** 2 for p in pts)
        sxy = sum(p[0] * p[1] for p in pts)
        slope = (n * sxy - sx * sy) / (n * sxx - sx ** 2)
        icpt = (sy - slope * sx) / n
        ybar = sy / n
        ssr = sum((y - (slope * x + icpt)) ** 2 for x, y in pts)
        sst = sum((y - ybar) ** 2 for x, y in pts)
        r2 = 1 - ssr / sst
        resid = max(abs(y - (slope * x + icpt)) for x, y in pts)

        if r2 < 0.999:
            problems.append(f"memory is no longer linear in pixel count (R2 = {r2:.6f}, "
                            f"expected > 0.999). README section 5 depends on this.")
        if resid > 0.05:
            problems.append(f"largest residual {resid:.4f} GiB exceeds 0.05, so the linear "
                            f"model no longer describes the measurements")
        if not (1.25 <= slope <= 1.36):
            problems.append(f"scaling slope {slope:.4f} GiB/Mpix has moved outside the "
                            f"1.25 to 1.36 band the report states")
        if f"{slope:.3f}" not in (ROOT / "README.md").read_text():
            problems.append(f"README does not state the fitted slope {slope:.3f} GiB per Mpix")

        print(f"scaling: {'FAIL' if problems else 'ok'} "
              f"({slope:.4f} GiB/Mpix + {icpt:.4f}, R2 = {r2:.6f}, "
              f"{n} points over {pts[0][0]:.2f} to {pts[-1][0]:.2f} Mpix)")
        for p in problems:
            print(f"  FAIL {p}")
        return not problems

    for p in problems:
        print(f"  FAIL {p}")
    print("scaling: FAIL")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--links", action="store_true")
    ap.add_argument("--numbers", action="store_true")
    ap.add_argument("--style", action="store_true")
    ap.add_argument("--tables", action="store_true")
    ap.add_argument("--scaling", action="store_true")
    ap.add_argument("--structure", action="store_true")
    args = ap.parse_args()

    run_all = not (args.links or args.numbers or args.style or args.tables
                   or args.scaling or args.structure)
    ok = True
    if run_all or args.links:
        ok &= check_links()
    if run_all or args.structure:
        ok &= check_structure()
    if run_all or args.numbers:
        ok &= check_numbers()
    if run_all or args.tables:
        ok &= check_tables()
    if run_all or args.scaling:
        ok &= check_scaling()
    if run_all or args.style:
        ok &= check_style()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

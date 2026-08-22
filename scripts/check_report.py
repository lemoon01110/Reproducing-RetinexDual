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
DOCS = ["README.md", "ENVIRONMENT.md", "DETERMINISM.md", "results/README.md"]

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
            if v not in readme:
                problems.append(f"README does not state the 4K latency '{v}'")
            r = f"{big['peak_reserved_bench']:.2f} GiB"
            if r not in readme:
                problems.append(f"README does not state the 4K reserved memory '{r}'")
        else:
            v = f"{big['peak_alloc']:.2f} GiB"
            if v not in readme:
                problems.append(f"README does not state the 4K working set '{v}'")

    for p in problems:
        print(f"  FAIL {p}")
    print(f"tables: {'FAIL' if problems else 'ok'} "
          f"(determinism and footprint checked against results/*.json)")
    return not problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--links", action="store_true")
    ap.add_argument("--numbers", action="store_true")
    ap.add_argument("--style", action="store_true")
    ap.add_argument("--tables", action="store_true")
    args = ap.parse_args()

    run_all = not (args.links or args.numbers or args.style or args.tables)
    ok = True
    if run_all or args.links:
        ok &= check_links()
    if run_all or args.numbers:
        ok &= check_numbers()
    if run_all or args.tables:
        ok &= check_tables()
    if run_all or args.style:
        ok &= check_style()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Re-check every defect this report claims, against upstream as it is today.

The report makes specific factual claims about someone else's repository. Those
claims were true at commit 9feec2c. Upstream is live and may have fixed them, in
which case this report should say so rather than keep asserting a defect that no
longer exists. That is a fairness obligation, not just a maintenance one.

Each check fetches the real file from GitHub and tests the specific condition.
Network only, no GPU, no dataset.

  --pinned    check against the pinned commit (should always pass)
  --head      check against upstream master (may legitimately fail once fixed)

Exit code is 0 when the report's claims still hold.
"""
import argparse
import json
import re
import sys
import urllib.error
import urllib.request

PINNED = "9feec2c0814d740221db2323e5e815a4d455abb6"
RAW = "https://raw.githubusercontent.com/ErrorLogic1211/RetinexDual/{ref}/{path}"
RELEASES = "https://api.github.com/repos/{repo}/releases/tags/{tag}"


def fetch(url, as_json=False):
    req = urllib.request.Request(url, headers={"User-Agent": "reproducing-retinexdual"})
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = r.read().decode("utf-8", "replace")
    return json.loads(raw) if as_json else raw


def check_requirements_unsatisfiable(ref):
    """Defect 2.1: torch==2.7.1 pinned with kernels that publish no torch2.7 wheel."""
    txt = fetch(RAW.format(ref=ref, path="requirements.txt"))
    pins = dict(re.findall(r"^([A-Za-z0-9_.-]+)==([^\s]+)$", txt, re.M))
    torch_pin = pins.get("torch")
    findings = []

    if torch_pin != "2.7.1":
        findings.append(f"torch pin is now {torch_pin}, was 2.7.1")

    for pkg, repo in (("mamba_ssm", "state-spaces/mamba"),
                      ("causal_conv1d", "Dao-AILab/causal-conv1d")):
        ver = pins.get(pkg)
        if not ver:
            findings.append(f"{pkg} is no longer pinned")
            continue
        tag = "v" + ver
        try:
            rel = fetch(RELEASES.format(repo=repo, tag=tag), as_json=True)
        except urllib.error.HTTPError as e:
            findings.append(f"could not read {repo} release {tag}: HTTP {e.code}")
            continue
        torches = sorted({m.group(0) for a in rel.get("assets", [])
                          for m in [re.search(r"torch\d+\.\d+", a["name"])] if m})
        wanted = "torch" + ".".join(torch_pin.split(".")[:2]) if torch_pin else None
        if wanted and wanted in torches:
            findings.append(f"{pkg} {ver} now publishes a {wanted} wheel, "
                            f"so the pin set may be satisfiable")
    return findings


def check_setup_py_imports_torch(ref):
    """Defect 2.3: setup.py imports torch at module scope, so PEP-517 build isolation breaks it."""
    txt = fetch(RAW.format(ref=ref, path="setup.py"))
    lines = txt.splitlines()
    for i, line in enumerate(lines[:40], 1):
        if re.match(r"^\s*import torch\s*$", line):
            return [] if i == 9 else [f"setup.py imports torch at line {i}, report says line 9"]
    return ["setup.py no longer imports torch at module scope, so defect 2.3 may be fixed"]


def check_readme_recommends_ref_fallback(ref):
    """Defect 3.1: the README suggests selective_scan_ref as 'numerically equivalent, just slower'."""
    txt = fetch(RAW.format(ref=ref, path="README.md"))
    findings = []
    if "selective_scan_ref" not in txt:
        findings.append("README no longer mentions selective_scan_ref, defect 3.1 may be fixed")
    if "numerically equivalent" not in txt:
        findings.append("README no longer calls the fallback 'numerically equivalent'")
    return findings


def check_metric_divisor_bug(ref):
    """Defect 3.2: num_img increments in both branches, metric sums only in the GT branch."""
    txt = fetch(RAW.format(ref=ref, path="inference_RetinexDual.py"))
    incs = len(re.findall(r"^\s*num_img \+= 1\s*$", txt, re.M))
    psnr_incs = len(re.findall(r"^\s*psnr_all \+= ", txt, re.M))
    findings = []
    if incs < 2:
        findings.append(f"num_img now increments {incs} time(s), not 2, so defect 3.2 may be fixed")
    if psnr_incs != 1:
        findings.append(f"psnr_all accumulates in {psnr_incs} place(s), report assumes 1")
    if "psnr_all / num_img" not in txt.replace("(", "").replace(")", ""):
        findings.append("the average is no longer computed as psnr_all / num_img")
    return findings


def check_gumbel_unguarded(ref):
    """Section 4.2: gumbel_softmax is called with no eval-mode guard."""
    path = "basicsr/models/archs/RetinexDuelSambaFusionFinalization_arch.py"
    txt = fetch(RAW.format(ref=ref, path=path))
    lines = txt.splitlines()
    findings = []
    hits = [i for i, l in enumerate(lines, 1) if "gumbel_softmax" in l]
    if not hits:
        return ["gumbel_softmax is gone from the arch file, section 4.2 may be obsolete"]
    for i in hits:
        window = "\n".join(lines[max(0, i - 12):i])
        if "self.training" in window:
            findings.append(f"line {i} now sits under a self.training guard, "
                            f"so inference may be deterministic")
    if 744 not in hits:
        findings.append(f"gumbel_softmax is at line(s) {hits}, report cites 744")
    return findings


def check_pin_still_described(ref):
    """Is the pinned commit still master HEAD, and does the report say so correctly?

    ENVIRONMENT.md described 9feec2c as 'master HEAD'. Upstream moved three
    commits ahead and the claim quietly became false. Nothing caught it, because
    every defect still held and every artifact still matched.
    """
    import pathlib
    head = fetch("https://api.github.com/repos/ErrorLogic1211/RetinexDual/commits/master",
                 as_json=True)["sha"]
    env = pathlib.Path(__file__).resolve().parent.parent / "ENVIRONMENT.md"
    text = env.read_text() if env.exists() else ""
    findings = []
    if head == PINNED:
        return findings
    # Upstream has moved. The report must not still call the pin master HEAD, and
    # should name the commit it has moved to.
    for i, line in enumerate(text.splitlines(), 1):
        if PINNED[:7] in line and "master HEAD" in line and "was" not in line:
            findings.append(f"ENVIRONMENT.md:{i} still calls the pinned commit master HEAD, "
                            f"but HEAD is now {head[:8]}")
    if head[:8] not in text:
        findings.append(f"upstream moved to {head[:8]} and ENVIRONMENT.md does not mention it")
    return findings


CHECKS = [
    ("pinned commit is described accurately", check_pin_still_described),
    ("2.1 requirements.txt is unsatisfiable", check_requirements_unsatisfiable),
    ("2.3 setup.py imports torch at line 9", check_setup_py_imports_torch),
    ("3.1 README recommends the reference fallback", check_readme_recommends_ref_fallback),
    ("3.2 metric average divides by the wrong count", check_metric_divisor_bug),
    ("4.2 gumbel_softmax has no eval-mode guard", check_gumbel_unguarded),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", action="store_true",
                    help="check upstream master instead of the pinned commit. A failure "
                         "here means upstream fixed something and the report should say so")
    ap.add_argument("--pinned", action="store_true", help="check the pinned commit (default)")
    args = ap.parse_args()

    ref = "master" if args.head else PINNED
    label = "upstream master" if args.head else f"pinned {PINNED[:7]}"
    print(f"Re-checking the report's claims against {label}\n")

    failed = 0
    for name, fn in CHECKS:
        try:
            findings = fn(ref)
        except Exception as e:
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
            failed += 1
            continue
        if findings:
            failed += 1
            print(f"  CHANGED {name}")
            for f in findings:
                print(f"          {f}")
        else:
            print(f"  holds   {name}")

    print()
    if failed and args.head:
        print(f"{failed} claim(s) no longer hold on master. That is not a bug in this")
        print("repository. It means upstream has moved and the report should note which")
        print("defects were subsequently fixed, so it does not keep asserting them.")
    elif failed:
        print(f"{failed} claim(s) do not hold at the pinned commit. That IS a problem:")
        print("the report describes a specific commit and should describe it accurately.")
    else:
        print("All claims still hold.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

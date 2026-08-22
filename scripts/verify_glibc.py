#!/usr/bin/env python3
"""Verify the glibc floor claimed for each CUDA kernel wheel.

README section 2.2 says the torch2.7 builds of mamba_ssm and causal_conv1d need
GLIBC_2.32 and therefore cannot load on Ubuntu 20.04, while the torch2.6 builds
of the same pinned versions need only GLIBC_2.14 and can. That is the argument
for moving torch rather than the kernels, so it should be measured rather than
asserted.

Method: download each wheel, extract the compiled extension, and read the
versioned symbol references from its ELF dynamic symbol table. The highest
GLIBC_x.y a binary references is the oldest glibc that can load it.

No GPU needed, and no need to install anything. Downloads are large, so this is
not run in CI. Run it once and commit the artifact.

Usage: python scripts/verify_glibc.py --out results/glibc_audit.json
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

WHEELS = [
    ("mamba_ssm", "2.3.0", "torch2.7",
     "https://github.com/state-spaces/mamba/releases/download/v2.3.0/"
     "mamba_ssm-2.3.0+cu12torch2.7cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"),
    ("mamba_ssm", "2.2.4", "torch2.6",
     "https://github.com/state-spaces/mamba/releases/download/v2.2.4/"
     "mamba_ssm-2.2.4+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"),
    ("causal_conv1d", "1.6.0", "torch2.7",
     "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.0/"
     "causal_conv1d-1.6.0+cu12torch2.7cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"),
    ("causal_conv1d", "1.5.0.post8", "torch2.6",
     "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.0.post8/"
     "causal_conv1d-1.5.0.post8+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"),
]

VER = re.compile(r"GLIBC_(\d+)\.(\d+)")


def max_glibc(so_path):
    """Highest GLIBC_x.y referenced by this shared object."""
    for tool in (["readelf", "--dyn-syms", "-W", str(so_path)],
                 ["objdump", "-T", str(so_path)],
                 ["nm", "-D", "--with-symbol-versions", str(so_path)]):
        try:
            out = subprocess.run(tool, capture_output=True, text=True, timeout=180).stdout
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        found = [(int(a), int(b)) for a, b in VER.findall(out)]
        if found:
            hi = max(found)
            return f"{hi[0]}.{hi[1]}", tool[0], len(found)
    return None, None, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--host-glibc", default="2.31",
                    help="the glibc the report's target machine has")
    args = ap.parse_args()

    host = tuple(int(x) for x in args.host_glibc.split("."))
    rows = []

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for pkg, ver, torch_tag, url in WHEELS:
            whl = td / f"{pkg}-{ver}-{torch_tag}.whl"
            print(f"[get ] {pkg} {ver} ({torch_tag})", flush=True)
            try:
                urllib.request.urlretrieve(url, whl)
            except Exception as e:
                print(f"       download failed: {type(e).__name__}: {e}")
                rows.append({"package": pkg, "version": ver, "torch": torch_tag,
                             "error": str(e)})
                continue

            with zipfile.ZipFile(whl) as z:
                sos = [n for n in z.namelist() if n.endswith(".so")]
                if not sos:
                    print("       no .so inside the wheel")
                    continue
                target = max(sos, key=lambda n: z.getinfo(n).file_size)
                z.extract(target, td)

            so = td / target
            req, tool, count = max_glibc(so)
            loads = req is not None and tuple(int(x) for x in req.split(".")) <= host
            rows.append({
                "package": pkg, "version": ver, "torch": torch_tag,
                "so": Path(target).name,
                "so_bytes": so.stat().st_size,
                "requires_glibc": req,
                "symbol_refs": count,
                "read_with": tool,
                "loads_on_host": loads,
            })
            print(f"       {Path(target).name}: needs GLIBC_{req}, "
                  f"{'loads' if loads else 'DOES NOT load'} on glibc {args.host_glibc}")
            so.unlink()

    print(f"\n| Wheel | Requires | Loads on glibc {args.host_glibc} |")
    print("|---|---|---|")
    for r in rows:
        if "error" in r:
            print(f"| `{r['package']}` {r['version']} / {r['torch']} | download failed | |")
            continue
        print(f"| `{r['package']}` {r['version']} / {r['torch']} | GLIBC_{r['requires_glibc']} | "
              f"{'yes' if r['loads_on_host'] else 'no'} |")

    good = [r for r in rows if "error" not in r]
    claim_holds = (all(not r["loads_on_host"] for r in good if r["torch"] == "torch2.7")
                   and all(r["loads_on_host"] for r in good if r["torch"] == "torch2.6")
                   and len(good) == len(WHEELS))
    print()
    print("Claim holds: the torch2.7 builds cannot load and the torch2.6 builds can."
          if claim_holds else
          "Claim does NOT hold as stated. README section 2.2 needs revisiting.")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"host_glibc": args.host_glibc, "claim_holds": claim_holds, "wheels": rows}, indent=2))
        print(f"wrote {args.out}")
    return 0 if claim_holds else 1


if __name__ == "__main__":
    sys.exit(main())

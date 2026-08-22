#!/usr/bin/env bash
# Build a working RetinexDual inference environment.
#
# The upstream requirements.txt cannot be installed as written. It pins
# torch==2.7.1 alongside mamba_ssm==2.2.4 and causal_conv1d==1.5.0.post8, and
# neither kernel package publishes a torch2.7 wheel (both stop at torch2.6).
# Bumping the kernels instead pulls in wheels that need GLIBC_2.32, which
# Ubuntu 20.04 does not have.
#
# Resolution: pin torch to 2.6.0 and keep the repository's exact pinned kernel
# versions. Only torch moves. See README section 2 and ENVIRONMENT.md.
set -euo pipefail

ENV_NAME="${ENV_NAME:-retinexdual-repro}"
PY_VER="3.11"
REPO_DIR="${REPO_DIR:-$HOME/RetinexDual}"

# torch 2.6.0 linux wheels are pre-cxx11-ABI, so the kernel wheels must carry the
# matching cxx11abiFALSE tag. A mismatch here produces an undefined-symbol error
# at import, not a clean failure at install.
ABI_TAG="cxx11abiFALSE"
CP_TAG="cp311-cp311"
MAMBA_WHL="https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4+cu12torch2.6${ABI_TAG}-${CP_TAG}-linux_x86_64.whl"
CCONV_WHL="https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.0.post8/causal_conv1d-1.5.0.post8+cu12torch2.6${ABI_TAG}-${CP_TAG}-linux_x86_64.whl"

command -v conda >/dev/null || { echo "conda not found on PATH"; exit 1; }
eval "$(conda shell.bash hook)"

echo "==> glibc on this machine"
ldd --version | head -1

if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
  echo "==> creating conda env ${ENV_NAME} (python ${PY_VER})"
  conda create -y -n "${ENV_NAME}" "python=${PY_VER}"
fi
conda activate "${ENV_NAME}"

echo "==> installing torch 2.6.0 + cu124"
pip install --quiet torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

echo "==> verifying ABI tag matches the wheels we are about to install"
ACTUAL_ABI=$(python -c "import torch; print('cxx11abiTRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'cxx11abiFALSE')")
if [[ "${ACTUAL_ABI}" != "${ABI_TAG}" ]]; then
  echo "torch reports ${ACTUAL_ABI} but this script targets ${ABI_TAG}."
  echo "Edit ABI_TAG at the top of this script and re-run."
  exit 1
fi
echo "    ${ACTUAL_ABI}, matches"

echo "==> installing the pinned CUDA kernels (torch2.6 builds, need only GLIBC_2.14)"
pip install --quiet "${CCONV_WHL}"
pip install --quiet "${MAMBA_WHL}"

echo "==> installing the remaining runtime dependencies"
# mamba_ssm's top-level __init__ imports transformers, even though only
# selective_scan_interface is used here.
pip install --quiet \
  "numpy" "opencv-python==4.10.0.84" "scikit_image==0.25.1" "einops==0.8.1" \
  "timm==1.0.15" "lpips==0.1.4" "torchmetrics" "transformers" \
  "natsort==8.4.0" "PyYAML==6.0.2" "tqdm==4.67.1" "lmdb==1.4.1" "yacs==0.1.8" \
  "addict" "future" "requests" "scipy"

echo "==> making basicsr importable"
# `python setup.py develop --no_cuda_ext` fails: setup.py does `import torch` at
# line 9 and pip's PEP-517 build isolation hides torch. Since --no_cuda_ext means
# there are no extensions to build, a .pth file is equivalent. basicsr has no
# __init__.py and resolves as a namespace package.
if [[ -d "${REPO_DIR}" ]]; then
  SITE=$(python -c "import site; print(site.getsitepackages()[0])")
  echo "${REPO_DIR}" > "${SITE}/retinexdual.pth"
  echo "    wrote ${SITE}/retinexdual.pth -> ${REPO_DIR}"
else
  echo "    REPO_DIR ${REPO_DIR} not found, skipping."
  echo "    Clone https://github.com/ErrorLogic1211/RetinexDual and re-run, or set REPO_DIR."
fi

echo "==> verifying the CUDA kernels actually load"
python - <<'PY'
import torch
import mamba_ssm.ops.selective_scan_interface as ssi
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

print(f"    torch      {torch.__version__}")
print(f"    mamba_ssm  {ssi.__file__}")
assert "site-packages" in ssi.__file__, "a local mamba_ssm directory is shadowing the real package"

if not torch.cuda.is_available():
    raise SystemExit("    CUDA not available")
print(f"    gpu        {torch.cuda.get_device_name(0)}")

# Correctness check against the pure-PyTorch reference at a small, tractable size.
torch.manual_seed(0)
b, d, l, n = 1, 8, 64, 16
u     = torch.randn(b, d, l, device="cuda")
delta = torch.rand(b, d, l, device="cuda")
A     = -torch.rand(d, n, device="cuda") - 0.5
B     = torch.randn(b, n, l, device="cuda")
C     = torch.randn(b, n, l, device="cuda")
D     = torch.randn(d, device="cuda")

fast = selective_scan_fn(u, delta, A, B, C, D, delta_softplus=True)
ref  = selective_scan_ref(u, delta, A, B, C, D, delta_softplus=True)
diff = (fast - ref).abs().max().item()
print(f"    selective_scan_fn vs ref: max abs diff {diff:.2e}")
assert diff < 1e-3, "CUDA kernel disagrees with the reference implementation"
print("    OK")
PY

echo
echo "Done. Activate with:  conda activate ${ENV_NAME}"

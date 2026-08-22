#!/usr/bin/env bash
# Reproduce RetinexDual's UHD-LL headline number.
#
#   REPO_DIR=~/RetinexDual DATA_DIR=~/data/UHD_LL/testing_set bash reproduce.sh
#
# Expects the environment built by setup_env.sh to be active.
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/RetinexDual}"
DATA_DIR="${DATA_DIR:-$HOME/data/UHD_LL/testing_set}"
SEEDS="${SEEDS:-0 1 2 3 4}"
OUT_DIR="${OUT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/results}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> repo    ${REPO_DIR}"
echo "==> data    ${DATA_DIR}"
echo "==> seeds   ${SEEDS}"
echo "==> out     ${OUT_DIR}"
echo

if [[ ! -d "${REPO_DIR}" ]]; then
  echo "Upstream checkout not found at ${REPO_DIR}."
  echo "  git clone https://github.com/ErrorLogic1211/RetinexDual ${REPO_DIR}"
  echo "  cd ${REPO_DIR} && git checkout 9feec2c0814d740221db2323e5e815a4d455abb6"
  exit 1
fi

WEIGHTS="${REPO_DIR}/pretrained_weights/UHD_LL.pth"
if [[ ! -f "${WEIGHTS}" ]]; then
  echo "Weights not found at ${WEIGHTS}."
  echo "  Download UHD_LL.pth from https://huggingface.co/ErrorLogic/RetinexDual"
  exit 1
fi

echo "==> checking the weight file"
EXPECTED_SHA="1977bb774cefb360bcb6edecdf4606568fa59f53aab8717fbe13bc35bacae182"
ACTUAL_SHA=$(sha256sum "${WEIGHTS}" | cut -d' ' -f1)
if [[ "${ACTUAL_SHA}" != "${EXPECTED_SHA}" ]]; then
  echo "    WARNING sha256 mismatch."
  echo "    expected ${EXPECTED_SHA}"
  echo "    actual   ${ACTUAL_SHA}"
  echo "    The upstream file may have been re-uploaded. Numbers may not match."
else
  echo "    sha256 matches"
fi

echo
echo "==> checking the dataset"
python "${HERE}/scripts/check_data.py" --data "${DATA_DIR}"

echo
echo "==> running the evaluation"
# shellcheck disable=SC2086
python "${HERE}/scripts/evaluate.py" \
  --repo "${REPO_DIR}" \
  --data "${DATA_DIR}" \
  --weights "${WEIGHTS}" \
  --seeds ${SEEDS} \
  --out "${OUT_DIR}"

echo
echo "Wrote:"
echo "  ${OUT_DIR}/reproduction_seeds.csv"
echo "  ${OUT_DIR}/reproduction_per_image.csv"
echo "  ${OUT_DIR}/reproduction_summary.json"

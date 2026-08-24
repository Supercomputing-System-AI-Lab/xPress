#!/bin/bash
# One-shot setup for XPress serving in vLLM.
# Clones upstream vLLM at the pinned base commit, applies the XPress patch, and installs
# with the prebuilt wheel (the patch is Python-only -> no CUDA compilation needed).
#
#   ./setup_vllm.sh [target_dir]     # default: ~/vllm-xpress-src
#
# Prereqs: a CUDA 12.x-capable node; conda env with python 3.11 recommended:
#   conda create -n vllm-xpress python=3.11 -y && conda activate vllm-xpress
set -euo pipefail

BASE_COMMIT=82ae4164ee016d4daecd2033c26f5c0827984a80
DIR="${1:-$HOME/vllm-xpress-src}"
HERE="$(cd "$(dirname "$0")" && pwd)"

git clone https://github.com/vllm-project/vllm.git "$DIR"
cd "$DIR"
git checkout "$BASE_COMMIT"
# XPress source files (browsable in this repo under vllm_xpress/src/) + the
# small registration patch for the 5 upstream files that must be touched.
cp -r "$HERE/src/vllm/." vllm/
git apply "$HERE/registration.patch"
echo "[setup] XPress installed on upstream vllm @ ${BASE_COMMIT:0:9}"

# Python-only changes on a pinned base -> reuse the prebuilt wheel for that commit.
VLLM_USE_PRECOMPILED=1 pip install -e .
pip install datasets   # for the benchmark's dataset loaders

echo ""
echo "[setup] verifying the XPress changes are present and active..."
python "$HERE/verify_install.py"

echo ""
echo "[setup] done. Quick test (private checkpoint repo needs 'hf auth login' first):"
echo "  export FLASHINFER_DISABLE_VERSION_CHECK=1"
echo "  python $HERE/bench_vllm_accept.py --dataset gsm8k --max-samples 128"

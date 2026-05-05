#!/usr/bin/env bash
# Start Jupyter with LD_LIBRARY_PATH set so TensorFlow sees CUDA/cuDNN on WSL.
# Usage: bash scripts/run_jupyter_wsl_gpu.sh
# Then open the printed URL in your browser.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "${ROOT}/venv/bin/activate"
PYV="$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")"
NV="${VIRTUAL_ENV}/lib/python${PYV}/site-packages/nvidia"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${NV}/cuda_runtime/lib:${NV}/cuda_nvrtc/lib:${NV}/cublas/lib:${NV}/cudnn/lib:${NV}/cufft/lib:${NV}/curand/lib:${NV}/cusolver/lib:${NV}/cusparse/lib:${NV}/nccl/lib:${NV}/nvjitlink/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec jupyter notebook "$@"

#!/usr/bin/env bash
# Install project deps into UV_PROJECT_ENVIRONMENT with PyTorch cu124 wheels
# (cluster A100 nodes: driver CUDA 12.4; uv sync alone resolves torch 2.x+cu130).

set -euo pipefail

: "${UV_PROJECT_ENVIRONMENT:?UV_PROJECT_ENVIRONMENT must be set}"

PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"
TORCH_VERSION="2.6.0"
TORCHVISION_VERSION="0.21.0"
TORCHAUDIO_VERSION="2.6.0"

_venv_python() {
  echo "${UV_PROJECT_ENVIRONMENT}/bin/python"
}

_cuda124_venv_ok() {
  "$(_venv_python)" -c "
import torch
if not torch.cuda.is_available():
    raise SystemExit(1)
if not str(torch.version.cuda or '').startswith('12.'):
    raise SystemExit(1)
torch.zeros(1, device='cuda')
" 2>/dev/null
}

if ! _cuda124_venv_ok; then
  rm -rf "${UV_PROJECT_ENVIRONMENT}"
fi

if [[ ! -x "$(_venv_python)" ]]; then
  uv venv "${UV_PROJECT_ENVIRONMENT}"
fi

uv sync --group project-dev

# uv sync resolves torch>=2.5 to cu130+; reinstall cu124 stack after full resolve.
uv pip install \
  --python "$(_venv_python)" \
  --reinstall-package torch \
  --reinstall-package torchvision \
  --reinstall-package torchaudio \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}" \
  --index-url "${PYTORCH_INDEX_URL}"

if ! _cuda124_venv_ok; then
  echo "ERROR: venv at ${UV_PROJECT_ENVIRONMENT} failed CUDA 12.4 smoke test" >&2
  "$(_venv_python)" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
  exit 1
fi

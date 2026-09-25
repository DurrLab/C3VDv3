#!/usr/bin/env bash
# Clone the upstream LTX-2 repo at the pinned commit, overlay this release's
# modified source files on top, and sync the Python environment with uv.
#
# Run from this sim2real/ directory:
#   bash setup_overlay.sh
set -euo pipefail

LTX_COMMIT="${LTX_COMMIT:-ae855f8}"
LTX_REMOTE="${LTX_REMOTE:-https://github.com/Lightricks/LTX-2.git}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

if [[ ! -d LTX-2/.git ]]; then
  echo "Cloning ${LTX_REMOTE} ..."
  git clone "${LTX_REMOTE}" LTX-2
fi

(
  cd LTX-2
  git fetch --tags origin
  git checkout "${LTX_COMMIT}"
)

echo "Applying overlay ..."
cp -r "${PROJECT_ROOT}/ltx_overlay/packages/." "${PROJECT_ROOT}/LTX-2/packages/"

echo
echo "Overlay applied. Next:"
echo "  cd LTX-2 && uv sync"
echo
echo "Then use the launchers under ${PROJECT_ROOT}/run/."

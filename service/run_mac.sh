#!/bin/bash
# Run GMNet service natively on Mac (with MPS GPU acceleration)
#
# Prerequisites:
#   pip install -r backend/requirements.txt
#
# Usage: ./run_mac.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Weights are bundled in the submodule
CHECKPOINTS="$REPO_DIR/vendor/GMNet/checkpoints"

if [ ! -f "$CHECKPOINTS/G_realworld.pth" ] && [ ! -f "$CHECKPOINTS/G_synthetic.pth" ]; then
    echo ""
    echo "============================================================"
    echo "  No GMNet weights found!"
    echo ""
    echo "  Initialize the git submodule:"
    echo "    git submodule update --init"
    echo ""
    echo "  Weights should be at:"
    echo "    $CHECKPOINTS/G_realworld.pth"
    echo "    $CHECKPOINTS/G_synthetic.pth"
    echo "============================================================"
    echo ""
    exit 1
fi

echo "Found weights:"
ls -lh "$CHECKPOINTS"/G_*.pth 2>/dev/null
echo ""

export GMNET_CHECKPOINTS_DIR="$CHECKPOINTS"
export MAX_MEGAPIXELS=50
export JOB_TTL_HOURS=24

echo "Starting GMNet on http://localhost:8002"
echo "Backend: PyTorch (MPS/CPU auto-detect)"
echo ""

cd "$SCRIPT_DIR/backend"
exec uvicorn app.main:app --host 0.0.0.0 --port 8002 --workers 1

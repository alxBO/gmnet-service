#!/bin/bash
# Deploy GMNet on a Vast.ai GPU instance
#
# === Vast.ai instance setup ===
#
# 1. Choose a GPU instance (any CUDA GPU, model is only ~7 MB)
# 2. Use a PyTorch template image (e.g. pytorch/pytorch:2.x-cuda12.x-runtime)
# 3. In "Docker options", add:  -p 8002:8002
# 4. Set disk space to at least 5 GB
#
# === On the instance ===
#
# SSH in, then:
#   git clone --recurse-submodules <repo-url>
#   cd gmnet-service/service
#   ./deploy_vastai.sh
#
# === Access ===
#
# Option A: Click "Open" on the instance card (Cloudflare tunnel, HTTPS)
# Option B: Use direct IP:port from "IP Port Info" popup
#           (or use env var VAST_TCP_PORT_8002 for the external port)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== GMNet Vast.ai Deployment ==="
echo ""

# 1. Install Python dependencies
echo "[1/2] Installing dependencies..."
pip install -q -r "$SCRIPT_DIR/backend/requirements.txt"

# 2. Check weights
CHECKPOINTS="$REPO_DIR/vendor/GMNet/checkpoints"
if [ ! -f "$CHECKPOINTS/G_realworld.pth" ] && [ ! -f "$CHECKPOINTS/G_synthetic.pth" ]; then
    echo ""
    echo "============================================================"
    echo "  No GMNet weights found!"
    echo ""
    echo "  Make sure you cloned with --recurse-submodules:"
    echo "    git submodule update --init"
    echo "============================================================"
    echo ""
    exit 1
fi

echo "  Found weights:"
ls -lh "$CHECKPOINTS"/G_*.pth 2>/dev/null

# 3. Start the service
echo "[2/2] Starting service on port 8002..."
echo ""

# Show access info if running on Vast.ai
if [ -n "$VAST_TCP_PORT_8002" ]; then
    echo "Direct access: http://$(hostname -I | awk '{print $1}'):$VAST_TCP_PORT_8002"
fi
echo "Local: http://0.0.0.0:8002"
echo ""

cd "$SCRIPT_DIR/backend"
export GMNET_CHECKPOINTS_DIR="$CHECKPOINTS"
export MAX_MEGAPIXELS=50
export JOB_TTL_HOURS=24

exec uvicorn app.main:app --host 0.0.0.0 --port 8002 --workers 1

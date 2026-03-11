# GMNet Service

Web service for SDR-to-HDR conversion using [GMNet](https://github.com/qtlark/GMNet) (ICLR 2025).
GMNet predicts a **gain map** from an SDR image and reconstructs an HDR result in EXR format.

## Quick Start (Mac)

```bash
git clone --recurse-submodules <repo-url>
cd gmnet-service/service
pip install -r backend/requirements.txt
./run_mac.sh
# Open http://localhost:8002
```

## Quick Start (Vast.ai GPU)

```bash
git clone --recurse-submodules <repo-url>
cd gmnet-service/service
./deploy_vastai.sh
```

## Features

- **Two models**: `realworld` (natural photos) and `synthetic` (rendered content), hot-swappable
- **Adjustable parameters**: peak luminance (2–32), downscaling factor (1x/2x/4x)
- **Real-time progress**: SSE-based queue with position tracking and cancellation
- **Image analysis**: input histogram, dynamic range, clipping; output HDR luminance stats
- **A/B comparison**: interactive SDR vs HDR slider with client-side tone mapping
- **EXR export**: download full-resolution float32 HDR output
- **Batch processing**: upload and process multiple images sequentially
- **In-memory processing**: no temporary files stored on disk (except model weights)
- **MPS + CUDA**: Apple Silicon (MPS with CPU fallback) and NVIDIA GPU support

## Architecture

```
Client (HTML/JS)  ──SSE──>  FastAPI (port 8002)  ──>  FIFO Queue  ──>  GMNet (GPU)
                                │                                          │
                           Upload/Analysis                          Gain Map → HDR
                           Tone Mapping (client)                    EXR (on download)
```

## Parameters

| Parameter | Values | Default | Description |
|-----------|--------|---------|-------------|
| Model | `realworld`, `synthetic` | `realworld` | Pretrained weight set |
| Peak | 2.0 – 32.0 | 8.0 | Peak luminance multiplier. Higher = more HDR range |
| Scale | 1, 2, 4 | 1 | Input downscaling factor. Higher = faster, less detail |

**How Peak works**: the gain map encodes per-pixel luminance boost. The `peak` parameter controls the
maximum possible boost: `HDR = SDR_linear × 2^(gainmap × log₂(peak)) / peak`. A peak of 8.0 means
up to 8× luminance expansion in bright areas.

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Service status, available models, device, queue size |
| `/api/upload` | POST | Upload SDR image (multipart), returns analysis + job_id |
| `/api/generate/{job_id}` | POST | Enqueue HDR generation with parameters |
| `/api/cancel/{job_id}` | POST | Cancel queued or running job |
| `/api/status/{job_id}` | GET | SSE progress stream |
| `/api/result/{job_id}` | GET | Result metadata and HDR analysis |
| `/api/hdr-raw/{job_id}` | GET | Raw float32 binary for client-side tone mapping |
| `/api/download/{job_id}` | GET | Download EXR file |

### Example

```bash
# Upload
curl -X POST http://localhost:8002/api/upload -F "file=@photo.jpg"
# {"job_id":"abc123", "width":1920, "height":1080, ...}

# Generate with custom peak
curl -X POST http://localhost:8002/api/generate/abc123 \
  -H "Content-Type: application/json" \
  -d '{"model":"realworld", "peak":12.0, "scale":1}'

# Download EXR
curl -O http://localhost:8002/api/download/abc123
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GMNET_CHECKPOINTS_DIR` | `/app/checkpoints` | Path to `G_*.pth` weight files |
| `GMNET_VENDOR_DIR` | auto-detected | Path to `vendor/GMNet/codes` |
| `MAX_MEGAPIXELS` | `50` | Maximum upload resolution in megapixels |
| `JOB_TTL_HOURS` | `24` | Hours before completed jobs are cleaned from memory |

## Deployment

### Mac (native)

Requires Python 3.10+ and PyTorch with MPS support (Apple Silicon).

```bash
cd service
pip install -r backend/requirements.txt
./run_mac.sh
```

The service auto-detects MPS. If an operation is unsupported on MPS, it falls back to CPU
transparently. Runs on port **8002**.

### Vast.ai (CUDA)

Choose any GPU instance (the model is only ~7 MB, so even a small GPU works).
Use a PyTorch template image and expose port 8002.

```bash
cd service
./deploy_vastai.sh
```

The script installs dependencies and starts the server. No weight download is needed — they're
bundled in the git submodule (~15 MB total).

## Troubleshooting

**"No GMNet weights found"**: Run `git submodule update --init` to pull the vendor repo.

**MPS errors on Mac**: Some PyTorch operations may not be available on MPS. The service
automatically falls back to CPU for the forward pass when this happens.

**Large images are slow**: Use Scale=2 or Scale=4 to reduce the input resolution for the
neural network. The gain map is upscaled back to full resolution.

**OpenEXR install fails**: On Mac, `brew install openexr` first. On Linux, `apt install libopenexr-dev`.

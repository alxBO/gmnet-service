"""GMNet inference pipeline — standalone loader that bypasses the original
training framework so we only depend on PyTorch + the network definition."""

import gc
import logging
import os
import sys
import types
from collections import OrderedDict
from threading import Lock

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vendor path setup — we need to import GMNet from vendor/GMNet/codes
# ---------------------------------------------------------------------------

def _vendor_codes_dir() -> str:
    """Resolve the vendor/GMNet/codes directory."""
    env = os.environ.get("GMNET_VENDOR_DIR")
    if env and os.path.isdir(env):
        return env
    # auto-detect relative to this file
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "..", "..", "..", "vendor", "GMNet", "codes"),
        os.path.join(here, "..", "..", "..", "..", "vendor", "GMNet", "codes"),
    ]
    for c in candidates:
        p = os.path.normpath(c)
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(
        "Cannot find vendor/GMNet/codes. Set GMNET_VENDOR_DIR or clone "
        "the submodule: git submodule update --init"
    )


def _setup_vendor_imports():
    """Add vendor path to sys.path and stub out gpu_memory_log (requires pynvml)."""
    codes_dir = _vendor_codes_dir()
    if codes_dir not in sys.path:
        sys.path.insert(0, codes_dir)

    # Stub out gpu_memory_log so we don't need pynvml at runtime
    stub = types.ModuleType("utils.gpu_memory_log")
    stub.gpu_memory_log = lambda *a, **kw: None
    sys.modules.setdefault("utils.gpu_memory_log", stub)


_setup_vendor_imports()

from models.modules.GMNet import GMNet  # noqa: E402
from models.modules.arch_util import initialize_weights, make_layer  # noqa: E402


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def _detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class GMNetPipeline:
    """Loads GMNet weights and runs SDR -> HDR inference entirely in-memory."""

    AVAILABLE_MODELS = ("synthetic", "realworld")

    def __init__(self, checkpoints_dir: str):
        self.device = _detect_device()
        self._lock = Lock()
        self._current_model: str | None = None
        self._net: GMNet | None = None
        self.checkpoints_dir = checkpoints_dir

        # Discover available weights
        self.available_models: list[str] = []
        for name in self.AVAILABLE_MODELS:
            path = os.path.join(checkpoints_dir, f"G_{name}.pth")
            if os.path.isfile(path):
                self.available_models.append(name)
                logger.info("Found model: %s (%s)", name, path)

        if not self.available_models:
            raise FileNotFoundError(
                f"No GMNet weights found in {checkpoints_dir}. "
                "Expected G_synthetic.pth and/or G_realworld.pth"
            )

        # Load first available model
        self._load_model(self.available_models[0])
        logger.info("GMNet pipeline ready on %s", self.device)

    def _load_model(self, model_name: str):
        """Load or hot-swap model weights."""
        if self._current_model == model_name and self._net is not None:
            return

        path = os.path.join(self.checkpoints_dir, f"G_{model_name}.pth")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Weights not found: {path}")

        logger.info("Loading GMNet model: %s", model_name)

        # Cleanup previous model
        if self._net is not None:
            del self._net
            self._net = None
            gc.collect()
            self._clear_device_cache()

        net = GMNet(in_nc=3, out_nc=1, nf=64, nb=16, act_type="relu")

        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        clean = OrderedDict()
        for k, v in state_dict.items():
            clean[k.removeprefix("module.")] = v
        net.load_state_dict(clean, strict=False)

        net.to(self.device)
        net.eval()
        self._net = net
        self._current_model = model_name

    def _clear_device_cache(self):
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps":
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    def run(
        self,
        img_bytes: bytes,
        progress_cb,
        model_name: str = "realworld",
        scale: int = 1,
        peak: float = 8.0,
    ) -> np.ndarray:
        """Run full inference pipeline. Returns float32 RGB HDR image (H, W, 3).

        Args:
            img_bytes: Raw image bytes (PNG/JPEG).
            progress_cb: Callable(stage, progress, message).
            model_name: "synthetic" or "realworld".
            scale: Downsampling factor for LQ input (1 = no downscaling).
            peak: Peak luminance multiplier for gain map reconstruction.

        Returns:
            HDR image as float32 numpy array (H, W, 3) in linear RGB.
        """
        with self._lock:
            return self._run_locked(img_bytes, progress_cb, model_name, scale, peak)

    def _run_locked(self, img_bytes, progress_cb, model_name, scale, peak):
        # Hot-swap model if needed
        if self._current_model != model_name:
            progress_cb("loading_model", 0.02, f"Loading {model_name} model...")
            self._load_model(model_name)

        progress_cb("preprocessing", 0.05, "Decoding and preprocessing image...")

        # Decode image
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Cannot decode image")

        img = img_bgr.astype(np.float32) / 255.0

        # Original SDR in linear RGB (for HDR reconstruction)
        img_rgb = img[:, :, ::-1].copy()  # BGR -> RGB
        img_linear = np.power(img_rgb, 2.2)

        # LQ: downscaled input
        if scale > 1:
            img_lq = cv2.resize(img, None, fx=1.0 / scale, fy=1.0 / scale,
                                interpolation=cv2.INTER_CUBIC).clip(0, 1)
        else:
            img_lq = img.copy()
        img_lq = img_lq[:, :, ::-1].copy()  # BGR -> RGB

        # MN: full-resolution conditioning
        img_mn = img_rgb.copy()

        # To tensors (C, H, W)
        lq_t = torch.from_numpy(np.ascontiguousarray(img_lq.transpose(2, 0, 1))).float()
        mn_t = torch.from_numpy(np.ascontiguousarray(img_mn.transpose(2, 0, 1))).float()

        progress_cb("inference", 0.20, "Running GMNet inference...")

        # Move to device and run
        lq_t = lq_t.unsqueeze(0).to(self.device)
        mn_t = mn_t.unsqueeze(0).to(self.device)

        with torch.no_grad():
            try:
                gain_map, q_gain_map = self._net((lq_t, mn_t))
            except Exception as e:
                if self.device.type == "mps":
                    # MPS fallback to CPU for unsupported ops
                    logger.warning("MPS inference failed (%s), falling back to CPU", e)
                    cpu_net = self._net.to("cpu")
                    gain_map, q_gain_map = cpu_net(
                        (lq_t.cpu(), mn_t.cpu())
                    )
                    self._net.to(self.device)
                    gain_map = gain_map.cpu()
                    q_gain_map = q_gain_map.cpu()
                else:
                    raise

        progress_cb("reconstruction", 0.70, "Reconstructing HDR from gain map...")

        # Get quantized gain map as numpy (H, W)
        gm = q_gain_map.detach().squeeze().float().cpu().numpy()
        gm = np.clip(gm, 0, 1)

        # Resize gain map to match original resolution (network output may differ by a few pixels)
        h, w = img_linear.shape[:2]
        if gm.shape[0] != h or gm.shape[1] != w:
            gm = cv2.resize(gm, (w, h), interpolation=cv2.INTER_LINEAR)

        # Expand to (H, W, 1) for broadcasting
        gm = gm[..., np.newaxis]

        # Reconstruct HDR: sdr_linear * 2^(gainmap * log2(peak)) / peak
        hdr = img_linear * np.power(2.0, gm * np.log2(peak)) / peak

        hdr = hdr.astype(np.float32)

        # Cleanup
        del lq_t, mn_t, gain_map, q_gain_map
        gc.collect()
        self._clear_device_cache()

        return hdr

    def close(self):
        with self._lock:
            if self._net is not None:
                del self._net
                self._net = None
            gc.collect()
            self._clear_device_cache()


def save_exr(path: str, img: np.ndarray):
    """Save float32 RGB image as EXR."""
    import OpenEXR
    import Imath

    img = img.astype(np.float32)
    h, w = img.shape[:2]
    header = OpenEXR.Header(w, h)
    float_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))

    if img.ndim == 2:
        header["channels"] = {"Y": float_chan}
        out = OpenEXR.OutputFile(path, header)
        out.writePixels({"Y": img.tobytes()})
    else:
        c = img.shape[2]
        names = ["R", "G", "B", "A"][:c]
        header["channels"] = {n: float_chan for n in names}
        out = OpenEXR.OutputFile(path, header)
        out.writePixels({n: img[:, :, i].tobytes() for i, n in enumerate(names)})
    out.close()

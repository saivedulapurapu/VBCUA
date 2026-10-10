"""Chooses the compute device for Whisper and Sentence-BERT from settings.DEVICE."""
import logging

from backend.app.config import settings

logger = logging.getLogger(__name__)

VALID_DEVICES = ("cpu", "cuda", "auto")


def cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_device() -> str:
    """Returns "cuda" or "cpu". "auto" uses CUDA when available; "cuda" falls back to CPU if it is not."""
    requested = (settings.DEVICE or "cpu").strip().lower()
    if requested not in VALID_DEVICES:
        logger.warning(f"Unknown DEVICE={settings.DEVICE!r}; using cpu")
        return "cpu"
    if requested == "cpu":
        return "cpu"
    if cuda_available():
        return "cuda"
    if requested == "cuda":
        logger.warning("DEVICE=cuda but CUDA is not available (CPU-only PyTorch or no GPU); using cpu")
    return "cpu"


def asr_device_label(device: str) -> str:
    """Stored per attempt: Whisper runs in fp16 on CUDA and fp32 on CPU."""
    return "cuda-fp16" if device == "cuda" else "cpu-fp32"

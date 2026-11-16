"""Image sanity checks and PIL <-> OpenCV conversions."""

import cv2
import numpy as np
from loguru import logger

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None


def _is_pil(image):
    return Image is not None and isinstance(image, Image.Image)


def to_bgr(image):
    """
    Normalize input to an OpenCV BGR uint8 HxWx3 array.

    Accepts: PIL Image (L/RGB/RGBA), ndarray HxW, HxWx1, HxWx3, HxWx4.
    Assumes ndarray 3-channel is already BGR (OpenCV convention).
    """
    try:
        if image is None:
            raise ValueError("image is None")

        if _is_pil(image):
            logger.info(f"to_bgr: PIL mode={image.mode} size={image.size}")
            rgb = np.asarray(image.convert("RGB"))
            if rgb.size == 0:
                raise ValueError("empty PIL image")
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        arr = np.asarray(image)
        if arr.size == 0:
            raise ValueError("empty ndarray")
        logger.info(f"to_bgr: ndarray shape={arr.shape} dtype={arr.dtype}")

        if arr.ndim == 2:
            return cv2.cvtColor(arr.astype(np.uint8, copy=False), cv2.COLOR_GRAY2BGR)
        if arr.ndim == 3 and arr.shape[2] == 1:
            return cv2.cvtColor(arr[:, :, 0].astype(np.uint8, copy=False), cv2.COLOR_GRAY2BGR)
        if arr.ndim == 3 and arr.shape[2] == 3:
            out = arr
            if out.dtype != np.uint8:
                out = np.clip(out, 0, 255).astype(np.uint8)
            return np.ascontiguousarray(out)
        if arr.ndim == 3 and arr.shape[2] == 4:
            bgr = arr[:, :, :3]
            if bgr.dtype != np.uint8:
                bgr = np.clip(bgr, 0, 255).astype(np.uint8)
            return np.ascontiguousarray(bgr)

        raise ValueError(f"unsupported image shape {getattr(arr, 'shape', None)}")
    except Exception as e:
        logger.exception(f"to_bgr failed: {e}")
        raise


def bgr_to_pil(image_bgr):
    """Convert OpenCV BGR uint8 array to a Pillow RGB Image."""
    try:
        if Image is None:
            raise ImportError("Pillow is required for bgr_to_pil")
        if image_bgr is None or not isinstance(image_bgr, np.ndarray) or image_bgr.size == 0:
            raise ValueError("bgr_to_pil needs a non-empty BGR ndarray")
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
    except Exception as e:
        logger.exception(f"bgr_to_pil failed: {e}")
        raise


def bgr_to_original_pil(image_bgr):
    """Pillow RGB copy for handoff 'original'."""
    return bgr_to_pil(image_bgr)

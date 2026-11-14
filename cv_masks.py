"""Classical CV alternative for finding cable-like masks."""

import cv2
import numpy as np


def _odd_ksize(ksize):
    """GaussianBlur requires a positive odd kernel size."""
    k = max(1, int(ksize))
    if k % 2 == 0:
        k += 1
    return k


def _yolo_union_mask(yolo_masks, shape_hw):
    """OR all YOLO instance masks into one binary image sized to the input."""
    h, w = shape_hw
    union = np.zeros((h, w), dtype=np.uint8)
    if not yolo_masks:
        return union
    for raw in yolo_masks:
        try:
            mask = (np.asarray(raw) > 0.5).astype(np.uint8)
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            union = np.maximum(union, mask)
        except Exception:
            continue
    return union


def detect_cv_masks(
    image,
    yolo_masks,
    min_width,
    gaussian_ksize,
    binary_threshold,
    logger=None,
):
    """
    Find CV masks that YOLO missed.

    Pipeline: gray -> Gaussian blur -> binary INV -> contours -> width filter
    -> drop anything overlapping a YOLO mask.

    Returns a list of uint8 binary masks (H x W), possibly empty.
    """
    try:
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            if logger is not None:
                logger.error("detect_cv_masks: invalid image")
            return []

        h, w = image.shape[:2]
        ksize = _odd_ksize(gaussian_ksize)
        thresh = int(binary_threshold)
        min_w = float(min_width)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        blur = cv2.GaussianBlur(gray, (ksize, ksize), 0)
        _, binary = cv2.threshold(blur, thresh, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        yolo_union = _yolo_union_mask(yolo_masks, (h, w))

        kept = []
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            if bw < min_w:
                continue
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 1, thickness=-1)
            if yolo_union.any() and np.logical_and(mask, yolo_union).any():
                continue
            kept.append(mask)

        if logger is not None:
            logger.info(
                f"detect_cv_masks: {len(kept)} new mask(s) "
                f"(ksize={ksize}, thresh={thresh}, min_w>={min_w:.0f})"
            )
        return kept
    except Exception as e:
        if logger is not None:
            logger.error(f"detect_cv_masks failed: {e}")
        return []


def overlay_cv_masks(image, cv_masks, color=(0, 255, 0), alpha=0.45):
    """Draw filled CV masks in green on a BGR image copy."""
    try:
        out = image.copy()
        if not cv_masks:
            return out
        overlay = out.copy()
        for mask in cv_masks:
            m = np.asarray(mask)
            if m.shape[:2] != out.shape[:2]:
                m = cv2.resize(m, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_NEAREST)
            overlay[m > 0] = color
        return cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0)
    except Exception:
        try:
            return image.copy()
        except Exception:
            return image

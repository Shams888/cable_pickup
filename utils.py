"""Small helpers for model inference and pickup points."""

import cv2
import numpy as np
from loguru import logger as _log


def filter_result_by_box_size(result, min_width, logger=None):
    """
    Drop detections whose longest box side is below the given threshold.
    """
    log = logger if logger is not None else _log
    try:
        n_before = 0 if result.boxes is None else len(result.boxes)

        if n_before > 0:
            xyxy = result.boxes.xyxy.detach().cpu().numpy()
            keep = [
                i for i, (x1, y1, x2, y2) in enumerate(xyxy)
                if max(x2 - x1, y2 - y1) >= min_width
            ]
            n_dropped = n_before - len(keep)
            if n_dropped:
                log.info(
                    f"infer: dropped {n_dropped}/{n_before} mask(s) "
                    f"(need max(w,h)>={min_width})"
                )
            result = result[keep] if keep else result[[]]

        return result
    except Exception as e:
        log.exception(f"filter_result_by_box_size failed: {e}")
        return result


def _mask_pca_axis(mask):
    """Return (pts Nx2 xy float32, axis, proj, t0, t1, mask_u8) or None."""
    m = (np.asarray(mask) > 0).astype(np.uint8)
    if m.ndim != 2 or not m.any():
        return None
    ys, xs = np.where(m > 0)
    if len(xs) < 8:
        return None
    pts = np.column_stack([xs, ys]).astype(np.float32)
    _mean, eigvecs = cv2.PCACompute(pts, mean=None)
    axis = eigvecs[0].astype(np.float64)
    if axis[0] < 0 or (abs(axis[0]) < 1e-6 and axis[1] < 0):
        axis = -axis
    proj = (pts - pts.mean(axis=0)) @ axis
    t0, t1 = float(proj.min()), float(proj.max())
    if (t1 - t0) < 1e-3:
        return None
    return pts, axis, proj, t0, t1, m


def _point_at_projection(pts, proj, dist, target, band_px):
    """Medial pixel (max DT) among points near projection target."""
    band = max(float(band_px), 1.0)
    near = np.abs(proj - target) <= band
    if not np.any(near):
        near = np.abs(proj - target) <= np.min(np.abs(proj - target)) + 1e-6
    cand = pts[near]
    h, w = dist.shape
    cx = np.clip(np.rint(cand[:, 0]).astype(int), 0, w - 1)
    cy = np.clip(np.rint(cand[:, 1]).astype(int), 0, h - 1)
    best = int(np.argmax(dist[cy, cx]))
    return float(cand[best, 0]), float(cand[best, 1])


def _radius_profile(pts, proj, dist, t0, t1, step_px=2.0, band_px=4.0):
    """Sample max DT (local radius) along the major axis."""
    span = t1 - t0
    n = max(8, int(np.ceil(span / max(step_px, 1.0))) + 1)
    samples = np.linspace(t0, t1, n)
    radii = np.zeros(n, dtype=np.float64)
    for i, t in enumerate(samples):
        near = np.abs(proj - t) <= band_px
        if not np.any(near):
            continue
        cand = pts[near]
        h, w = dist.shape
        cx = np.clip(np.rint(cand[:, 0]).astype(int), 0, w - 1)
        cy = np.clip(np.rint(cand[:, 1]).astype(int), 0, h - 1)
        radii[i] = float(dist[cy, cx].max())
    return samples, radii


def _gray_along(samples, pts, proj, gray, band_px=4.0):
    """Mean grayscale along axis samples (255 = white label-ish)."""
    if gray is None:
        return np.zeros(len(samples), dtype=np.float64)
    g = gray
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    out = np.zeros(len(samples), dtype=np.float64)
    h, w = g.shape[:2]
    for i, t in enumerate(samples):
        near = np.abs(proj - t) <= band_px
        if not np.any(near):
            continue
        cand = pts[near]
        cx = np.clip(np.rint(cand[:, 0]).astype(int), 0, w - 1)
        cy = np.clip(np.rint(cand[:, 1]).astype(int), 0, h - 1)
        out[i] = float(g[cy, cx].mean())
    return out


def _find_clip_index(samples, radii, t0, t1, fraction, clip_window, wire_ref):
    """
    Clip = strongest local radius peak near mid-length (not just global max,
    which can latch onto a hanging label / connector).
    """
    span = t1 - t0
    mid = t0 + float(np.clip(fraction, 0.0, 1.0)) * span
    half = float(np.clip(clip_window, 0.05, 0.49)) * span
    in_win = (samples >= mid - half) & (samples <= mid + half)
    idxs = np.where(in_win)[0]
    if len(idxs) == 0:
        return int(np.argmax(radii))

    thresh = max(wire_ref * 1.35, wire_ref + 3.0)
    # Local peaks inside the window.
    peaks = []
    for i in idxs:
        if i <= 0 or i >= len(radii) - 1:
            continue
        if radii[i] >= radii[i - 1] and radii[i] >= radii[i + 1] and radii[i] >= thresh:
            peaks.append(i)
    if peaks:
        # Prefer peak closest to mid; break ties with taller peak.
        return min(peaks, key=lambda i: (abs(samples[i] - mid), -radii[i]))

    # Fallback: max radius in window.
    return int(idxs[np.argmax(radii[idxs])])


def _walk_off_bulge(samples, radii, clip_i, direction, wire_ref):
    """
    Walk from clip along samples until radius returns near wire thickness.
    direction: -1 or +1 in sample index space.
    """
    limit = max(wire_ref * 1.2, wire_ref + 2.0)
    i = clip_i
    last = clip_i
    while 0 <= i < len(samples):
        last = i
        if radii[i] <= limit and abs(i - clip_i) >= 1:
            return i
        i += direction
    return last


def pickup_point(
    mask,
    fraction=0.5,
    band_px=4.0,
    offset_px=45.0,
    clip_window=0.25,
    image=None,
):
    """
    Rotation-stable pickup: clip landmark, then offset onto wire body.

    - Find mid-window radius peak (mounting clip), not label/connector.
    - Walk off the bulge onto normal wire thickness on both sides.
    - Step `offset_px` further; prefer the side that looks like dark wire
      (not white label) and stays wire-thin.
    """
    try:
        parsed = _mask_pca_axis(mask)
        if parsed is None:
            m = (np.asarray(mask) > 0).astype(np.uint8)
            if not m.any():
                _log.info("pickup_point: empty mask")
                return None
            ys, xs = np.where(m > 0)
            _log.info("pickup_point: PCA failed, using centroid")
            return float(xs.mean()), float(ys.mean())

        pts, axis, proj, t0, t1, m = parsed
        dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
        samples, radii = _radius_profile(pts, proj, dist, t0, t1, step_px=2.0, band_px=band_px)
        gray_prof = _gray_along(samples, pts, proj, image, band_px=band_px)

        positive = radii > 0
        wire_ref = float(np.percentile(radii[positive], 40)) if np.any(positive) else 1.0
        clip_i = _find_clip_index(samples, radii, t0, t1, fraction, clip_window, wire_ref)

        # Leave the clip bulge on each side, then apply a fixed extra offset.
        step = float(np.mean(np.diff(samples))) if len(samples) > 1 else 2.0
        extra = max(1, int(round(float(offset_px) / max(step, 1e-3))))
        span = t1 - t0
        margin = 0.12 * span

        def label_side_score(direction):
            """Prefer the side that has a white sticker between clip and that end."""
            if direction < 0:
                lo, hi = 0, clip_i
            else:
                lo, hi = clip_i, len(samples) - 1
            if hi <= lo:
                return 0.0
            # Near-clip band on that side (where labels usually sit).
            near = slice(lo, hi + 1)
            g = gray_prof[near]
            r = radii[near]
            if g.size == 0:
                return 0.0
            # Bright & not thick like a connector => sticker evidence.
            bright = g > 150
            wireish = r < wire_ref * 1.5
            return float(np.sum(bright & wireish)) + 0.05 * float(g.max())

        # Choose side: toward the sticker (same side as yellow in your refs).
        dir_pref = -1 if label_side_score(-1) >= label_side_score(+1) else +1

        def build_pick(direction):
            edge_i = _walk_off_bulge(samples, radii, clip_i, direction, wire_ref)
            pick_i = int(np.clip(edge_i + direction * extra, 0, len(samples) - 1))
            t = float(samples[pick_i])
            t = float(np.clip(t, t0 + margin, t1 - margin))
            pick_i = int(np.argmin(np.abs(samples - t)))
            # If we landed on white label, step back toward the clip onto dark wire.
            for _ in range(extra + 8):
                if gray_prof[pick_i] < 150 and radii[pick_i] <= wire_ref * 1.35:
                    break
                pick_i = int(np.clip(pick_i - direction, 0, len(samples) - 1))
            return pick_i, float(samples[pick_i])

        pick_i, pick_t = build_pick(dir_pref)
        # If preferred side still looks bad (on bulge/label), try the other.
        bad = gray_prof[pick_i] >= 150 or radii[pick_i] > wire_ref * 1.5
        if bad:
            alt_i, alt_t = build_pick(-dir_pref)
            pick_i, pick_t = alt_i, alt_t

        pt = _point_at_projection(pts, proj, dist, pick_t, band_px)
        _log.info(f"pickup_point: uv={pt} clip_i={clip_i} dir={dir_pref}")
        return pt
    except Exception as e:
        _log.exception(f"pickup_point failed: {e}")
        raise


def pickup_points_for_masks(masks, fraction=0.5, offset_px=45.0, image=None):
    """Return list of (x, y) or None for each mask."""
    try:
        return [
            pickup_point(m, fraction=fraction, offset_px=offset_px, image=image)
            for m in (masks or [])
        ]
    except Exception as e:
        _log.exception(f"pickup_points_for_masks failed: {e}")
        raise


def draw_pickup_points(image, points, radius=6, color=(0, 255, 255)):
    """Draw pickup markers (BGR, default yellow) on a copy of the image."""
    try:
        out = image.copy()
        r = max(1, int(radius))
        for pt in points or []:
            if pt is None:
                continue
            x, y = int(round(pt[0])), int(round(pt[1]))
            cv2.circle(out, (x, y), r, color, -1, cv2.LINE_AA)
            cv2.circle(out, (x, y), r + 2, (0, 0, 0), 1, cv2.LINE_AA)
        return out
    except Exception as e:
        _log.exception(f"draw_pickup_points failed: {e}")
        raise

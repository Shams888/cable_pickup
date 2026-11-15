"""Classical CV + mask join/suppress helpers for cable segmentation."""

import cv2
import numpy as np


def _odd_ksize(ksize):
    """GaussianBlur requires a positive odd kernel size."""
    k = max(1, int(ksize))
    if k % 2 == 0:
        k += 1
    return k


def _resize_mask(mask, shape_hw):
    """Binary uint8 mask resized to (h, w)."""
    h, w = shape_hw
    m = (np.asarray(mask) > 0.5).astype(np.uint8)
    if m.shape[:2] != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    return m


def _yolo_union_mask(yolo_masks, shape_hw):
    """OR all instance masks into one binary image sized to the input."""
    h, w = shape_hw
    union = np.zeros((h, w), dtype=np.uint8)
    if not yolo_masks:
        return union
    for raw in yolo_masks:
        try:
            union = np.maximum(union, _resize_mask(raw, shape_hw))
        except Exception:
            continue
    return union


def _piece_ends(contour):
    """Two ends of a piece along its main direction, plus that direction."""
    vx, vy, x0, y0 = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    direction = np.array([vx, vy], dtype=np.float64)
    points = contour.reshape(-1, 2).astype(np.float64)
    t = (points - np.array([x0, y0], dtype=np.float64)) @ direction
    ends = [
        points[t < t.min() + 5].mean(axis=0),
        points[t > t.max() - 5].mean(axis=0),
    ]
    return ends, direction


def _angle_deg(a, b):
    """Angle in degrees between two vectors (direction-insensitive via abs cos)."""
    cos = abs(float(a @ b)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    return float(np.degrees(np.arccos(min(cos, 1.0))))


def _ends_should_join(ends_i, dir_i, ends_j, dir_j, max_gap_px, max_join_angle_deg, max_lateral_px):
    """True if a facing-end pair looks like a label gap on one cable."""
    max_gap = float(max_gap_px)
    max_ang = float(max_join_angle_deg)
    max_lat = float(max_lateral_px)
    for a in ends_i:
        for b in ends_j:
            gap = b - a
            dist = float(np.linalg.norm(gap))
            if dist > max_gap or dist < 1e-6:
                continue
            if _angle_deg(gap, dir_i) > max_ang or _angle_deg(gap, dir_j) > max_ang:
                continue
            # Reject parallel neighbors: gap should not be mostly sideways.
            # Lateral = component of gap perpendicular to average cable axis.
            axis = dir_i + dir_j
            if float(np.dot(dir_i, dir_j)) < 0:
                axis = dir_i - dir_j
            n = np.linalg.norm(axis)
            if n < 1e-6:
                continue
            axis = axis / n
            lateral = abs(float(gap[0] * (-axis[1]) + gap[1] * axis[0]))
            if lateral > max_lat:
                continue
            return True, (a, b)
    return False, None


def _bridge_pieces_on_mask(
    mask, pieces, max_gap_px, max_join_angle_deg, join_width_px, max_lateral_px=40
):
    """Draw thin end-aligned bridges between piece contours onto mask (in-place)."""
    if len(pieces) < 2:
        return
    info = [_piece_ends(c) for c in pieces]
    width = max(1, int(join_width_px))
    for i in range(len(pieces)):
        for j in range(i + 1, len(pieces)):
            ends_i, dir_i = info[i]
            ends_j, dir_j = info[j]
            ok, pair = _ends_should_join(
                ends_i, dir_i, ends_j, dir_j, max_gap_px, max_join_angle_deg, max_lateral_px
            )
            if ok and pair is not None:
                a, b = pair
                cv2.line(
                    mask,
                    (int(a[0]), int(a[1])),
                    (int(b[0]), int(b[1])),
                    255,
                    width,
                )


def _contour_passes_size(contour, min_cable_area, min_length):
    (_cx, _cy), (rw, rh), _ang = cv2.minAreaRect(contour)
    return cv2.contourArea(contour) >= float(min_cable_area) and max(rw, rh) >= float(min_length)


def join_instance_masks(
    masks,
    shape_hw,
    max_gap_px=120,
    max_join_angle_deg=25,
    join_width_px=15,
    min_piece_area=300,
    min_cable_area=8000,
    min_length=500,
    scores=None,
    max_lateral_px=40,
):
    """
    Join instance masks whose facing ends line up across a small gap (label split).

    Pairwise union-find only — never OR all masks onto one canvas (that mixes
    parallel cables that merely touch or slightly overlap).

    Returns (joined_masks, joined_scores). Scores for a merged blob = max of members.
    """
    h, w = shape_hw
    if not masks:
        return [], []

    binary = []
    piece_scores = []
    pieces = []
    for i, raw in enumerate(masks):
        m = _resize_mask(raw, (h, w))
        if int(m.sum()) < int(min_piece_area):
            continue
        ctrs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not ctrs:
            continue
        binary.append(m)
        piece_scores.append(float(scores[i]) if scores is not None else 1.0)
        pieces.append(max(ctrs, key=cv2.contourArea))

    n = len(binary)
    if n == 0:
        return [], []

    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    info = [_piece_ends(c) for c in pieces]
    for i in range(n):
        for j in range(i + 1, n):
            ends_i, dir_i = info[i]
            ends_j, dir_j = info[j]
            ok, _pair = _ends_should_join(
                ends_i, dir_i, ends_j, dir_j, max_gap_px, max_join_angle_deg, max_lateral_px
            )
            if ok:
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    out_masks, out_scores = [], []
    width = max(1, int(join_width_px))
    for members in groups.values():
        canvas = np.zeros((h, w), dtype=np.uint8)
        score = 0.0
        for i in members:
            canvas = np.maximum(canvas, (binary[i] * 255).astype(np.uint8))
            score = max(score, piece_scores[i])

        # Bridges only inside this group (label gaps), not across all cables.
        if len(members) > 1:
            group_pieces = [pieces[i] for i in members]
            _bridge_pieces_on_mask(
                canvas,
                group_pieces,
                max_gap_px,
                max_join_angle_deg,
                width,
                max_lateral_px,
            )

        contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        # One group -> one mask (largest CC if bridge failed to connect).
        c = max(contours, key=cv2.contourArea)
        if not _contour_passes_size(c, min_cable_area, min_length):
            continue
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [c], -1, 1, thickness=-1)
        out_masks.append(mask)
        out_scores.append(score if score > 0 else 1.0)
    return out_masks, out_scores


def suppress_overlapping_masks(masks, scores=None, overlap_thresh=0.1):
    """
    Greedy NMS on mask pixels (not boxes).

    Two masks overlap if intersection / min(area_i, area_j) >= overlap_thresh.
    Keep higher score (or larger area if scores tie / missing).
    """
    if not masks:
        return [], []

    prepared = []
    for i, raw in enumerate(masks):
        m = (np.asarray(raw) > 0).astype(np.uint8)
        area = int(m.sum())
        if area == 0:
            continue
        score = float(scores[i]) if scores is not None else float(area)
        prepared.append((m, score, area))

    # Higher score first; break ties with larger area.
    order = sorted(range(len(prepared)), key=lambda i: (prepared[i][1], prepared[i][2]), reverse=True)
    kept_masks, kept_scores = [], []
    thresh = float(overlap_thresh)

    for i in order:
        m, score, area = prepared[i]
        overlaps = False
        for kept in kept_masks:
            inter = int(np.logical_and(m, kept).sum())
            if inter == 0:
                continue
            smaller = min(area, int(kept.sum()))
            if smaller > 0 and (inter / smaller) >= thresh:
                overlaps = True
                break
        if not overlaps:
            kept_masks.append(m)
            kept_scores.append(score)

    return kept_masks, kept_scores


def _find_cable_contours(
    image,
    gaussian_ksize,
    binary_threshold,
    min_piece_area,
    max_gap_px,
    max_join_angle_deg,
    join_width_px,
    min_cable_area,
    min_length,
):
    """
    Gray -> blur -> thresh -> pieces -> end-aligned thin bridges -> whole cables.
    Returns a list of contours (possibly empty).
    """
    ksize = _odd_ksize(gaussian_ksize)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    blur = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    _, binary = cv2.threshold(blur, int(binary_threshold), 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pieces = [c for c in contours if cv2.contourArea(c) >= float(min_piece_area)]
    if not pieces:
        return []

    mask = np.zeros(binary.shape, dtype=np.uint8)
    cv2.drawContours(mask, pieces, -1, 255, cv2.FILLED)
    _bridge_pieces_on_mask(mask, pieces, max_gap_px, max_join_angle_deg, join_width_px)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if _contour_passes_size(c, min_cable_area, min_length)]


def detect_cv_masks(
    image,
    yolo_masks,
    min_width,
    gaussian_ksize,
    binary_threshold,
    min_piece_area=300,
    max_gap_px=120,
    max_join_angle_deg=25,
    join_width_px=15,
    min_cable_area=8000,
    logger=None,
):
    """
    Find CV masks that YOLO missed.

    Pipeline: end-aligned piece join (thin bridges) -> area/length filter
    -> drop anything overlapping a YOLO mask.

    Returns a list of uint8 binary masks (H x W), possibly empty.
    """
    try:
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            if logger is not None:
                logger.error("detect_cv_masks: invalid image")
            return []

        h, w = image.shape[:2]
        min_w = float(min_width)

        cables = _find_cable_contours(
            image,
            gaussian_ksize,
            binary_threshold,
            min_piece_area,
            max_gap_px,
            max_join_angle_deg,
            join_width_px,
            min_cable_area,
            min_w,
        )
        yolo_union = _yolo_union_mask(yolo_masks, (h, w))

        kept = []
        for contour in cables:
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 1, thickness=-1)
            if yolo_union.any() and np.logical_and(mask, yolo_union).any():
                continue
            kept.append(mask)

        if logger is not None:
            logger.info(
                f"detect_cv_masks: {len(kept)} new mask(s) "
                f"(of {len(cables)} cables, ksize={_odd_ksize(gaussian_ksize)}, "
                f"thresh={int(binary_threshold)}, gap={int(max_gap_px)}, "
                f"min_len>={min_w:.0f})"
            )
        return kept
    except Exception as e:
        if logger is not None:
            logger.error(f"detect_cv_masks failed: {e}")
        return []


def overlay_masks(image, masks, color=(0, 255, 0), alpha=0.45, scores=None):
    """Draw filled masks on a BGR image copy; optional conf text at centroid."""
    try:
        out = image.copy()
        if not masks:
            return out
        overlay = out.copy()
        for idx, mask in enumerate(masks):
            m = np.asarray(mask)
            if m.shape[:2] != out.shape[:2]:
                m = cv2.resize(m, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_NEAREST)
            overlay[m > 0] = color
            if scores is not None and idx < len(scores):
                ys, xs = np.where(m > 0)
                if len(xs) > 0:
                    cx, cy = int(xs.mean()), int(ys.mean())
                    cv2.putText(
                        overlay,
                        f"{scores[idx]:.2f}",
                        (cx, max(0, cy - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
        return cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0)
    except Exception:
        try:
            return image.copy()
        except Exception:
            return image


def overlay_cv_masks(image, cv_masks, color=(0, 255, 0), alpha=0.45):
    """Draw filled CV masks in green on a BGR image copy."""
    return overlay_masks(image, cv_masks, color=color, alpha=alpha)

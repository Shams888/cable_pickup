#!/usr/bin/env python3
"""
trace_export.py - record an exact, step-by-step trace of MODEL.infer() for the
Pipeline Explorer (pipeline_explorer.html).

This script does NOT change your detection or pickup code. It:
  1. calls the same helpers from cv_masks.py and utils.py, in the same order as infer(),
  2. records what goes into and comes out of every step (masks, arrays, coordinates,
     settings, branch decisions),
  3. checks that every stage gives the same answer as your real functions
     (join_instance_masks, suppress_overlapping_masks, detect_cv_masks, pickup_point,
     and finally MODEL.infer itself).

Usage (run from the project folder):
    python3 trace_export.py --image Sheet_Data_1_folder/image84.png
    python3 trace_export.py --image Sheet_Data_1_folder/image84.png \
                            --image Sheet_Data_1_folder/image85.png \
                            --embed pipeline_explorer.html

Each image gives traces/trace_<name>.json. Load it in the explorer with "Load trace",
or use --embed to put the traces inside pipeline_explorer.html so it opens with them.
"""

import argparse
import base64
import inspect
import json
import os
import re

import cv2
import numpy as np

import cv_masks as cm
import utils as U
from model import MODEL


# ---------------------------------------------------------------------------
# small encoders
# ---------------------------------------------------------------------------
def rle(mask):
    """Row-major run lengths, starting with a run of zeros (may be 0)."""
    flat = (np.asarray(mask) > 0).astype(np.uint8).ravel()
    if flat.size == 0:
        return []
    change = np.flatnonzero(np.diff(flat)) + 1
    bounds = np.concatenate([[0], change, [flat.size]])
    runs = np.diff(bounds).astype(int).tolist()
    if flat[0] == 1:
        runs = [0] + runs
    return runs


def mask_obj(mask):
    m = np.asarray(mask) > 0
    ys, xs = np.where(m)
    if xs.size:
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    else:
        bbox = [0, 0, 0, 0]
    return {"shape": list(m.shape), "rle": rle(m), "area": int(m.sum()), "bbox": bbox}


def b64img(img, ext=".jpg", quality=88):
    if ext == ".jpg":
        ok, buf = cv2.imencode(ext, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        mime = "image/jpeg"
    else:
        ok, buf = cv2.imencode(ext, img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        mime = "image/png"
    if not ok:
        return None
    return f"data:{mime};base64," + base64.b64encode(buf.tobytes()).decode()


def P(p, nd=2):
    return [round(float(p[0]), nd), round(float(p[1]), nd)]


def R(x, nd=3):
    return round(float(x), nd)


def poly(contour, eps=1.0):
    """Simplified outline for drawing only."""
    approx = cv2.approxPolyDP(contour, eps, True)
    return approx.reshape(-1, 2).astype(int).tolist()


# ---------------------------------------------------------------------------
# pair checks: mirror of _ends_should_join + the centroid test used by
# join_instance_masks and _bridge_pieces_on_mask
# ---------------------------------------------------------------------------
def explain_ends(ends_i, dir_i, ends_j, dir_j, max_gap_px, max_join_angle_deg, max_lateral_px):
    max_gap, max_ang, max_lat = float(max_gap_px), float(max_join_angle_deg), float(max_lateral_px)
    checks = []
    for ai, a in enumerate(ends_i):
        for bi, b in enumerate(ends_j):
            gap = b - a
            dist = float(np.linalg.norm(gap))
            c = {"a": ai, "b": bi, "pa": P(a), "pb": P(b), "dist": R(dist, 2)}
            checks.append(c)
            if dist > max_gap or dist < 1e-6:
                c["fail"] = "gap"
                continue
            ang_i, ang_j = cm._angle_deg(gap, dir_i), cm._angle_deg(gap, dir_j)
            c["ang_i"], c["ang_j"] = R(ang_i, 2), R(ang_j, 2)
            if ang_i > max_ang or ang_j > max_ang:
                c["fail"] = "angle"
                continue
            axis = dir_i + dir_j
            if float(np.dot(dir_i, dir_j)) < 0:
                axis = dir_i - dir_j
            n = np.linalg.norm(axis)
            if n < 1e-6:
                c["fail"] = "axis"
                continue
            axis = axis / n
            lateral = abs(float(gap[0] * (-axis[1]) + gap[1] * axis[0]))
            c["lateral"] = R(lateral, 2)
            if lateral > max_lat:
                c["fail"] = "lateral"
                continue
            c["fail"] = None
            return True, (a, b), checks
    return False, None, checks


def trace_pairs(info, lengths, cents, min_length, max_gap_px, max_join_angle_deg, max_lateral_px,
                ids=None, compact_after=None):
    """Same order of tests as join_instance_masks / _bridge_pieces_on_mask."""
    n = len(info)
    ids = ids if ids is not None else list(range(n))
    long_thresh = max(float(min_length) * 0.75, 400.0)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            rec = {"i": ids[i], "j": ids[j]}
            if lengths[i] >= long_thresh and lengths[j] >= long_thresh:
                rec["result"] = "both_long"
                pairs.append(rec)
                continue
            ends_i, dir_i = info[i]
            ends_j, dir_j = info[j]
            ok, pair, checks = explain_ends(ends_i, dir_i, ends_j, dir_j,
                                            max_gap_px, max_join_angle_deg, max_lateral_px)
            real_ok, _ = cm._ends_should_join(ends_i, dir_i, ends_j, dir_j,
                                              max_gap_px, max_join_angle_deg, max_lateral_px)
            assert bool(ok) == bool(real_ok), "mirror of _ends_should_join disagrees"
            min_dist = min(c["dist"] for c in checks)
            rec["min_dist"] = min_dist
            keep_detail = compact_after is None or min_dist <= compact_after
            if keep_detail:
                rec["checks"] = checks
            if not ok:
                rec["result"] = "ends"
                pairs.append(rec)
                continue
            axis = dir_i + dir_j
            if float(np.dot(dir_i, dir_j)) < 0:
                axis = dir_i - dir_j
            nrm = np.linalg.norm(axis)
            if nrm < 1e-6:
                rec["result"] = "axis"
                pairs.append(rec)
                continue
            axis = axis / nrm
            delta = cents[j] - cents[i]
            cent_lat = abs(float(delta[0] * (-axis[1]) + delta[1] * axis[0]))
            rec["cent_lat"] = R(cent_lat, 2)
            rec["cent_lat_max"] = R(max(float(max_lateral_px) * 1.5, 50.0), 2)
            if cent_lat > max(float(max_lateral_px) * 1.5, 50.0):
                rec["result"] = "centroid_lateral"
                pairs.append(rec)
                continue
            rec["result"] = "join"
            rec["pair"] = [P(pair[0]), P(pair[1])]
            rec["checks"] = checks
            pairs.append(rec)
    return pairs


def piece_record(contour, k):
    ends, direction = cm._piece_ends(contour)
    return {
        "id": k,
        "ends": [P(ends[0]), P(ends[1])],
        "dir": [R(direction[0], 5), R(direction[1], 5)],
        "length": R(cm._contour_length(contour), 2),
        "centroid": P(cm._contour_centroid(contour)),
        "contour_area": R(cv2.contourArea(contour), 1),
        "poly": poly(contour),
    }, (ends, direction)


# ---------------------------------------------------------------------------
# Stage 1: join_instance_masks
# ---------------------------------------------------------------------------
def trace_join(masks, shape_hw, scores, max_gap_px, max_join_angle_deg, join_width_px,
               min_piece_area, min_cable_area, min_length, max_lateral_px=40):
    h, w = shape_hw
    T = {"items": [], "pieces": [], "pairs": [], "groups": [], "outputs": []}
    binary, piece_scores, pieces = [], [], []
    for i, raw in enumerate(masks):
        m = cm._resize_mask(raw, (h, w))
        item = {"raw": i, "area": int(m.sum()), "resized": mask_obj(m)}
        if int(m.sum()) < int(min_piece_area):
            item["result"] = "too_small"
            T["items"].append(item)
            continue
        ctrs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not ctrs:
            item["result"] = "no_contour"
            T["items"].append(item)
            continue
        item["result"] = "piece"
        item["piece"] = len(pieces)
        item["n_contours"] = len(ctrs)
        T["items"].append(item)
        binary.append(m)
        piece_scores.append(float(scores[i]) if scores is not None else 1.0)
        pieces.append(max(ctrs, key=cv2.contourArea))

    info, lengths, cents = [], [], []
    for k, c in enumerate(pieces):
        rec, inf = piece_record(c, k)
        rec["raw"] = [it["raw"] for it in T["items"] if it.get("piece") == k][0]
        rec["score"] = R(piece_scores[k], 4)
        T["pieces"].append(rec)
        info.append(inf)
        lengths.append(cm._contour_length(c))
        cents.append(cm._contour_centroid(c))

    T["long_thresh"] = R(max(float(min_length) * 0.75, 400.0), 2)
    T["pairs"] = trace_pairs(info, lengths, cents, min_length, max_gap_px,
                             max_join_angle_deg, max_lateral_px)

    n = len(binary)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for p in T["pairs"]:
        if p["result"] == "join":
            ri, rj = find(p["i"]), find(p["j"])
            if ri != rj:
                parent[rj] = ri
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    out_masks, out_scores = [], []
    width = max(1, int(join_width_px))
    for members in groups.values():
        g = {"members": members, "bridges": []}
        canvas = np.zeros((h, w), dtype=np.uint8)
        score = 0.0
        for i in members:
            canvas = np.maximum(canvas, (binary[i] * 255).astype(np.uint8))
            score = max(score, piece_scores[i])
        if len(members) > 1:
            before = canvas.copy()
            group_pieces = [pieces[i] for i in members]
            cm._bridge_pieces_on_mask(canvas, group_pieces, max_gap_px, max_join_angle_deg,
                                      width, max_lateral_px, min_length)
            sub_pairs = trace_pairs([info[i] for i in members], [lengths[i] for i in members],
                                    [cents[i] for i in members], min_length, max_gap_px,
                                    max_join_angle_deg, max_lateral_px, ids=members)
            mirror = before.copy()
            for p in sub_pairs:
                if p["result"] == "join":
                    a, b = p["pair"]
                    cv2.line(mirror, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), 255, width)
                    g["bridges"].append({"i": p["i"], "j": p["j"], "a": a, "b": b, "width": width})
            assert np.array_equal(mirror, canvas), "bridge mirror differs from _bridge_pieces_on_mask"
        contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        g["n_contours"] = len(contours)
        if not contours:
            g["result"] = "no_contour"
            T["groups"].append(g)
            continue
        c = max(contours, key=cv2.contourArea)
        (_cx, _cy), (rw, rh), _ang = cv2.minAreaRect(c)
        g["area"] = R(cv2.contourArea(c), 1)
        g["length"] = R(max(rw, rh), 2)
        g["poly"] = poly(c)
        g["score"] = R(score if score > 0 else 1.0, 4)
        if not cm._contour_passes_size(c, min_cable_area, min_length):
            g["result"] = "too_small_area" if cv2.contourArea(c) < float(min_cable_area) else "too_short"
            T["groups"].append(g)
            continue
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [c], -1, 1, thickness=-1)
        g["result"] = "kept"
        g["output"] = len(out_masks)
        g["mask"] = mask_obj(mask)
        T["groups"].append(g)
        out_masks.append(mask)
        out_scores.append(score if score > 0 else 1.0)

    real_m, real_s = cm.join_instance_masks(
        masks, (h, w), max_gap_px=max_gap_px, max_join_angle_deg=max_join_angle_deg,
        join_width_px=join_width_px, min_piece_area=min_piece_area, min_cable_area=min_cable_area,
        min_length=min_length, scores=scores, max_lateral_px=max_lateral_px)
    T["verified"] = (len(real_m) == len(out_masks)
                     and all(np.array_equal(a, b) for a, b in zip(real_m, out_masks))
                     and np.allclose(real_s, out_scores))
    return real_m, real_s, T


# ---------------------------------------------------------------------------
# Stage 1b: suppress_overlapping_masks
# ---------------------------------------------------------------------------
def trace_suppress(masks, scores, overlap_thresh):
    T = {"thresh": float(overlap_thresh), "order": [], "steps": []}
    prepared, src = [], []
    for i, raw in enumerate(masks):
        m = (np.asarray(raw) > 0).astype(np.uint8)
        area = int(m.sum())
        if area == 0:
            continue
        s = float(scores[i]) if scores is not None else float(area)
        prepared.append((m, s, area))
        src.append(i)
    order = sorted(range(len(prepared)), key=lambda i: (prepared[i][1], prepared[i][2]), reverse=True)
    T["order"] = [src[i] for i in order]
    kept, kept_src = [], []
    for i in order:
        m, s, area = prepared[i]
        st = {"input": src[i], "score": R(s, 4), "area": area, "compared": []}
        overlaps = False
        for k, kmask in enumerate(kept):
            inter = int(np.logical_and(m, kmask).sum())
            cmp_ = {"kept_input": kept_src[k], "inter": inter}
            st["compared"].append(cmp_)
            if inter == 0:
                continue
            smaller = min(area, int(kmask.sum()))
            cmp_["smaller"] = smaller
            cmp_["ratio"] = R(inter / smaller, 4) if smaller else 0
            if smaller > 0 and (inter / smaller) >= float(overlap_thresh):
                overlaps = True
                st["suppressed_by"] = kept_src[k]
                break
        st["result"] = "suppressed" if overlaps else "kept"
        if not overlaps:
            kept.append(m)
            kept_src.append(src[i])
        T["steps"].append(st)
    real_m, real_s = cm.suppress_overlapping_masks(masks, scores, overlap_thresh=overlap_thresh)
    T["kept_inputs"] = kept_src
    T["verified"] = len(real_m) == len(kept) and all(np.array_equal(a, b) for a, b in zip(real_m, kept))
    return real_m, real_s, T


# ---------------------------------------------------------------------------
# Stage 2: detect_cv_masks / _find_cable_contours
# ---------------------------------------------------------------------------
def trace_cv(image, yolo_masks, min_width, gaussian_ksize, binary_threshold, min_piece_area,
             max_gap_px, max_join_angle_deg, join_width_px, min_cable_area):
    h, w = image.shape[:2]
    min_w = float(min_width)
    T = {"min_w": min_w}
    ksize = cm._odd_ksize(gaussian_ksize)
    T["ksize_in"], T["ksize"] = gaussian_ksize, ksize
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    blur = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    _, binary = cv2.threshold(blur, int(binary_threshold), 255, cv2.THRESH_BINARY_INV)
    T["gray_img"] = b64img(gray, ".jpg", 85)
    T["blur_img"] = b64img(blur, ".jpg", 85)
    T["binary_img"] = b64img(binary, ".png")
    T["binary_fg"] = int((binary > 0).sum())

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    T["n_contours"] = len(contours)
    pieces = [c for c in contours if cv2.contourArea(c) >= float(min_piece_area)]
    T["n_small"] = len(contours) - len(pieces)
    T["small_polys"] = [poly(c, 1.5) for c in contours if cv2.contourArea(c) < float(min_piece_area)
                        and cv2.contourArea(c) >= 20][:400]
    T["pieces"], info, lengths, cents = [], [], [], []
    for k, c in enumerate(pieces):
        rec, inf = piece_record(c, k)
        T["pieces"].append(rec)
        info.append(inf)
        lengths.append(cm._contour_length(c))
        cents.append(cm._contour_centroid(c))
    T["long_thresh"] = R(max(min_w * 0.75, 400.0), 2)

    if not pieces:
        T["cables"], T["candidates"] = [], []
    else:
        mask = np.zeros(binary.shape, dtype=np.uint8)
        cv2.drawContours(mask, pieces, -1, 255, cv2.FILLED)
        before = mask.copy()
        cm._bridge_pieces_on_mask(mask, pieces, max_gap_px, max_join_angle_deg, join_width_px,
                                  min_length=min_w)
        # _find_cable_contours does not pass max_lateral_px -> default 40
        T["pairs"] = trace_pairs(info, lengths, cents, min_w, max_gap_px, max_join_angle_deg, 40,
                                 compact_after=3 * float(max_gap_px))
        width = max(1, int(join_width_px))
        mirror = before.copy()
        T["bridges"] = []
        for p in T["pairs"]:
            if p["result"] == "join":
                a, b = p["pair"]
                cv2.line(mirror, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), 255, width)
                T["bridges"].append({"i": p["i"], "j": p["j"], "a": a, "b": b, "width": width})
        assert np.array_equal(mirror, mask), "CV bridge mirror differs from _bridge_pieces_on_mask"
        T["bridged_img"] = b64img(mask, ".png")

        contours2, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        T["cables"] = []
        passing = []
        for c in contours2:
            (_cx, _cy), (rw, rh), _ang = cv2.minAreaRect(c)
            area = float(cv2.contourArea(c))
            rec = {"area": R(area, 1), "length": R(max(rw, rh), 2), "poly": poly(c, 1.5)}
            ok = cm._contour_passes_size(c, min_cable_area, min_w)
            rec["result"] = "pass" if ok else ("too_small_area" if area < float(min_cable_area) else "too_short")
            if area >= 1000 or ok:
                T["cables"].append(rec)
            if ok:
                rec["candidate"] = len(passing)
                passing.append(c)
        T["n_cable_contours"] = len(contours2)

        union = cm._yolo_union_mask(yolo_masks, (h, w))
        T["union"] = mask_obj(union)
        T["candidates"] = []
        for k, c in enumerate(passing):
            cmask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(cmask, [c], -1, 1, thickness=-1)
            overlap = int(np.logical_and(cmask, union).sum())
            hit = bool(union.any() and overlap > 0)
            T["candidates"].append({"mask": mask_obj(cmask), "overlap_px": overlap,
                                    "result": "rejected_overlap" if hit else "kept"})

    real = cm.detect_cv_masks(image, yolo_masks, min_width, gaussian_ksize, binary_threshold,
                              min_piece_area, max_gap_px, max_join_angle_deg, join_width_px,
                              min_cable_area, None)
    kept = [c["mask"] for c in T["candidates"] if c["result"] == "kept"]
    T["verified"] = len(real) == len(kept) and all(
        mask_obj(a)["rle"] == b["rle"] for a, b in zip(real, kept))
    return real, T


# ---------------------------------------------------------------------------
# Stage 3: pickup_point (mirrors the nested label_side_score / build_pick)
# ---------------------------------------------------------------------------
def trace_pickup(mask, fraction, offset_px, image, band_px=4.0, clip_window=0.25):
    T = {"fraction": fraction, "offset_px": float(offset_px), "band_px": band_px,
         "clip_window": clip_window}
    parsed = U._mask_pca_axis(mask)
    if parsed is None:
        m = (np.asarray(mask) > 0).astype(np.uint8)
        ys, xs = np.where(m > 0)
        T["fallback"] = True
        T["point"] = [float(xs.mean()), float(ys.mean())] if xs.size else None
        return T
    pts, axis, proj, t0, t1, m = parsed
    mean = pts.mean(axis=0)
    _mean_cv, eig = cv2.PCACompute(pts, mean=None)
    raw_axis = eig[0].astype(np.float64)
    T["pca"] = {"n_pts": int(len(pts)), "mean": P(mean, 3), "pca_mean": P(_mean_cv[0], 3),
                "axis_raw": [R(raw_axis[0], 6), R(raw_axis[1], 6)],
                "axis": [R(axis[0], 6), R(axis[1], 6)],
                "flipped": bool(not np.allclose(raw_axis, axis)),
                "t0": R(t0, 3), "t1": R(t1, 3)}

    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    ys, xs = np.where(m > 0)
    x0, y0 = max(0, int(xs.min()) - 2), max(0, int(ys.min()) - 2)
    x1, y1 = min(m.shape[1], int(xs.max()) + 3), min(m.shape[0], int(ys.max()) + 3)
    crop = dist[y0:y1, x0:x1]
    dmax = float(crop.max()) if crop.size else 1.0
    T["dist"] = {"bbox": [x0, y0, x1, y1], "max": R(dmax, 3),
                 "img": b64img(np.clip(crop / max(dmax, 1e-6) * 255, 0, 255).astype(np.uint8), ".png")}

    samples, radii = U._radius_profile(pts, proj, dist, t0, t1, step_px=2.0, band_px=band_px)
    gray_prof = U._gray_along(samples, pts, proj, image, band_px=band_px)
    n = len(samples)
    T["samples"] = [R(s, 3) for s in samples]
    T["radii"] = [R(v, 3) for v in radii]
    T["gray"] = [R(v, 2) for v in gray_prof]

    # a few loop iterations in detail (radius + gray)
    gimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    demo = []
    for i in sorted(set([max(1, n // 8), n // 2, min(n - 2, (7 * n) // 8)])):
        t = samples[i]
        near = np.abs(proj - t) <= band_px
        cand = pts[near]
        rec = {"i": int(i), "t": R(t), "count": int(near.sum())}
        if cand.size:
            cx = np.clip(np.rint(cand[:, 0]).astype(int), 0, dist.shape[1] - 1)
            cy = np.clip(np.rint(cand[:, 1]).astype(int), 0, dist.shape[0] - 1)
            b = int(np.argmax(dist[cy, cx]))
            rec["best"] = [int(cx[b]), int(cy[b])]
            rec["radius"] = R(dist[cy, cx].max())
            rec["gray"] = R(gimg[cy, cx].mean(), 2)
        demo.append(rec)
    T["demo"] = demo

    positive = radii > 0
    wire_ref = float(np.percentile(radii[positive], 40)) if np.any(positive) else 1.0
    T["wire_ref"] = R(wire_ref, 4)
    T["n_positive"] = int(positive.sum())

    clip_i = U._find_clip_index(samples, radii, t0, t1, fraction, clip_window, wire_ref)
    span = t1 - t0
    mid = t0 + float(np.clip(fraction, 0.0, 1.0)) * span
    half = float(np.clip(clip_window, 0.05, 0.49)) * span
    in_win = (samples >= mid - half) & (samples <= mid + half)
    idxs = np.where(in_win)[0]
    thresh = max(wire_ref * 1.35, wire_ref + 3.0)
    peaks = []
    for i in idxs:
        if i <= 0 or i >= len(radii) - 1:
            continue
        if radii[i] >= radii[i - 1] and radii[i] >= radii[i + 1] and radii[i] >= thresh:
            peaks.append(int(i))
    if len(idxs) == 0:
        how, mirror_i = "no_window", int(np.argmax(radii))
    elif peaks:
        how, mirror_i = "peak", min(peaks, key=lambda i: (abs(samples[i] - mid), -radii[i]))
    else:
        how, mirror_i = "window_max", int(idxs[np.argmax(radii[idxs])])
    assert int(mirror_i) == int(clip_i), "mirror of _find_clip_index disagrees"
    T["clip"] = {"mid": R(mid), "half": R(half), "win_lo": int(idxs[0]) if len(idxs) else None,
                 "win_hi": int(idxs[-1]) if len(idxs) else None, "thresh": R(thresh, 4),
                 "peaks": peaks, "how": how, "clip_i": int(clip_i)}

    step = float(np.mean(np.diff(samples))) if len(samples) > 1 else 2.0
    extra = max(1, int(round(float(offset_px) / max(step, 1e-3))))
    margin = 0.12 * span
    T["step"], T["extra"], T["span"], T["margin"] = R(step, 4), extra, R(span), R(margin)

    def side(direction):
        if direction < 0:
            lo, hi = 0, clip_i
        else:
            lo, hi = clip_i, len(samples) - 1
        rec = {"direction": direction, "lo": int(lo), "hi": int(hi)}
        if hi <= lo:
            rec["score"] = 0.0
            return 0.0, rec
        g = gray_prof[lo:hi + 1]
        r = radii[lo:hi + 1]
        if g.size == 0:
            rec["score"] = 0.0
            return 0.0, rec
        bright = g > 150
        wireish = r < wire_ref * 1.5
        score = float(np.sum(bright & wireish)) + 0.05 * float(g.max())
        rec.update({"n_bright": int(bright.sum()), "n_wireish": int(wireish.sum()),
                    "n_both": int(np.sum(bright & wireish)), "g_max": R(g.max(), 2),
                    "score": R(score, 4)})
        return score, rec

    s_neg, rec_neg = side(-1)
    s_pos, rec_pos = side(+1)
    dir_pref = -1 if s_neg >= s_pos else +1
    T["side"] = {"neg": rec_neg, "pos": rec_pos, "dir_pref": dir_pref}

    def build(direction):
        rec = {"direction": direction}
        limit = max(wire_ref * 1.2, wire_ref + 2.0)
        i, last, path = clip_i, clip_i, []
        while 0 <= i < len(samples):
            last = i
            path.append(int(i))
            if radii[i] <= limit and abs(i - clip_i) >= 1:
                break
            i += direction
        edge_i = U._walk_off_bulge(samples, radii, clip_i, direction, wire_ref)
        assert int(edge_i) == int(last), "mirror of _walk_off_bulge disagrees"
        pick_i0 = int(np.clip(edge_i + direction * extra, 0, len(samples) - 1))
        t_raw = float(samples[pick_i0])
        t = float(np.clip(t_raw, t0 + margin, t1 - margin))
        pick_i1 = int(np.argmin(np.abs(samples - t)))
        pick_i, back, broke = pick_i1, [], False
        for _ in range(extra + 8):
            g, rr = float(gray_prof[pick_i]), float(radii[pick_i])
            ok = g < 150 and rr <= wire_ref * 1.35
            back.append([int(pick_i), R(g, 2), R(rr, 3), bool(ok)])
            if ok:
                broke = True
                break
            pick_i = int(np.clip(pick_i - direction, 0, len(samples) - 1))
        rec.update({"limit": R(limit, 4), "walk": path, "edge_i": int(edge_i), "pick_i0": pick_i0,
                    "t_raw": R(t_raw), "t_clamped": R(t), "clamped": bool(abs(t - t_raw) > 1e-9),
                    "pick_i1": pick_i1, "backtrack": back, "found_wire": broke,
                    "pick_i": int(pick_i), "pick_t": R(samples[pick_i])})
        return pick_i, float(samples[pick_i]), rec

    pick_i, pick_t, first = build(dir_pref)
    tries = [first]
    bad = gray_prof[pick_i] >= 150 or radii[pick_i] > wire_ref * 1.5
    T["bad"] = {"gray": R(gray_prof[pick_i], 2), "radius": R(radii[pick_i], 3),
                "radius_limit": R(wire_ref * 1.5, 4), "is_bad": bool(bad)}
    if bad:
        pick_i, pick_t, second = build(-dir_pref)
        tries.append(second)
    T["tries"] = tries
    T["pick_i"], T["pick_t"] = int(pick_i), R(pick_t)

    band = max(float(band_px), 1.0)
    near = np.abs(proj - pick_t) <= band
    fb = not np.any(near)
    if fb:
        near = np.abs(proj - pick_t) <= np.min(np.abs(proj - pick_t)) + 1e-6
    cand = pts[near]
    cx = np.clip(np.rint(cand[:, 0]).astype(int), 0, dist.shape[1] - 1)
    cy = np.clip(np.rint(cand[:, 1]).astype(int), 0, dist.shape[0] - 1)
    b = int(np.argmax(dist[cy, cx]))
    point = U._point_at_projection(pts, proj, dist, pick_t, band_px)
    T["final"] = {"band": band, "fallback": bool(fb), "count": int(len(cand)),
                  "best_dist": R(dist[cy[b], cx[b]]), "point": P(point, 3)}
    T["point"] = [float(point[0]), float(point[1])]

    real = U.pickup_point(mask, fraction=fraction, offset_px=offset_px, image=image)
    T["verified"] = real is not None and np.allclose(real, point)
    return T


# ---------------------------------------------------------------------------
def sources():
    out = {}

    def add(name, obj):
        lines, start = inspect.getsourcelines(obj)
        out[name] = {"file": os.path.basename(inspect.getsourcefile(obj)), "start": start,
                     "code": "".join(lines)}

    add("MODEL.process", MODEL.process)
    add("MODEL.infer", MODEL.infer)
    for n in ["_odd_ksize", "_resize_mask", "_yolo_union_mask", "_piece_ends", "_angle_deg",
              "_ends_should_join", "_bridge_pieces_on_mask", "_contour_passes_size",
              "_contour_length", "_contour_centroid", "join_instance_masks",
              "suppress_overlapping_masks", "_find_cable_contours", "detect_cv_masks",
              "overlay_masks", "overlay_cv_masks"]:
        add(n, getattr(cm, n))
    for n in ["filter_result_by_box_size", "_mask_pca_axis", "_point_at_projection",
              "_radius_profile", "_gray_along", "_find_clip_index", "_walk_off_bulge",
              "pickup_point", "pickup_points_for_masks", "draw_pickup_points"]:
        add(n, getattr(U, n))
    return out


def letterbox_check(result, raw_masks, h, w):
    """Compare _resize_mask(masks.data) with Ultralytics' own polygons (masks.xy)."""
    rows = []
    try:
        xy = result.masks.xy
    except Exception:
        return rows
    for i, raw in enumerate(raw_masks):
        try:
            a = cm._resize_mask(raw, (h, w)).astype(bool)
            b = np.zeros((h, w), np.uint8)
            if len(xy[i]) >= 3:
                cv2.fillPoly(b, [np.asarray(xy[i]).astype(np.int32)], 1)
            b = b.astype(bool)
            inter, uni = np.logical_and(a, b).sum(), np.logical_or(a, b).sum()
            ya, yb = np.where(a)[0], np.where(b)[0]
            rows.append({"raw": i, "iou": R(inter / uni if uni else 0, 4),
                         "dy": R(ya.mean() - yb.mean(), 2) if ya.size and yb.size else None})
        except Exception:
            continue
    return rows


def trace_image(model, path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(path)
    h, w = image.shape[:2]
    model.reload_params_if_changed()
    s = {k: getattr(model, k) for k in [
        "model_confidence", "mask_width_threshold_px", "cv_gaussian_ksize", "cv_binary_threshold",
        "cv_min_piece_area", "cv_max_gap_px", "cv_max_join_angle_deg", "cv_join_width_px",
        "cv_min_cable_area", "mask_overlap_thresh", "enable_pickup", "pickup_fraction",
        "pickup_offset_px", "pickup_marker_radius"]}

    results = model.model.predict(image, conf=model.model_confidence, verbose=False)
    result = results[0]
    raw_masks, raw_scores, raw_boxes = [], [], []
    if result.masks is not None:
        raw_masks = list(result.masks.data.detach().cpu().numpy())
        if result.boxes is not None and len(result.boxes):
            raw_scores = [float(c) for c in result.boxes.conf.detach().cpu().numpy()]
            raw_boxes = result.boxes.xyxy.detach().cpu().numpy().round(1).tolist()
        else:
            raw_scores = [1.0] * len(raw_masks)
    raw = [{"score": R(raw_scores[i], 4), "box": raw_boxes[i] if raw_boxes else None,
            "native": mask_obj(np.asarray(raw_masks[i]) > 0.5)} for i in range(len(raw_masks))]

    yolo_masks, yolo_scores, Tj = trace_join(
        raw_masks, (h, w), raw_scores if raw_scores else None, s["cv_max_gap_px"],
        s["cv_max_join_angle_deg"], s["cv_join_width_px"], s["cv_min_piece_area"],
        s["cv_min_cable_area"], s["mask_width_threshold_px"])
    yolo_masks, yolo_scores, Ts = trace_suppress(yolo_masks, yolo_scores, s["mask_overlap_thresh"])

    cv_only, Tc = trace_cv(image, yolo_masks, s["mask_width_threshold_px"], s["cv_gaussian_ksize"],
                           s["cv_binary_threshold"], s["cv_min_piece_area"], s["cv_max_gap_px"],
                           s["cv_max_join_angle_deg"], s["cv_join_width_px"], s["cv_min_cable_area"])

    all_masks = list(yolo_masks) + list(cv_only)
    cables = []
    for k, mk in enumerate(all_masks):
        src = "yolo" if k < len(yolo_masks) else "cv"
        c = {"index": k, "source": src, "mask": mask_obj(mk),
             "score": R(yolo_scores[k], 4) if src == "yolo" else None}
        if s["enable_pickup"]:
            c["pickup"] = trace_pickup(mk, s["pickup_fraction"], s["pickup_offset_px"], image)
        cables.append(c)

    # End-to-end check against the real infer()
    ann = cm.overlay_masks(image, yolo_masks, color=(255, 0, 0), alpha=0.45, scores=yolo_scores)
    ann = cm.overlay_cv_masks(ann, cv_only)
    if s["enable_pickup"]:
        ann = U.draw_pickup_points(ann, [c["pickup"]["point"] for c in cables],
                                   radius=s["pickup_marker_radius"], color=(0, 255, 255))
    real = model.infer(image)
    diff = int(np.any(real != ann, axis=2).sum()) if real is not None and real.shape == ann.shape else -1

    return {
        "version": 1,
        "kind": "exact",
        "name": os.path.splitext(os.path.basename(path))[0],
        "image_path": path,
        "w": w, "h": h,
        "settings": {k: (v if not isinstance(v, np.generic) else v.item()) for k, v in s.items()},
        "image": b64img(image, ".jpg", 90),
        "result": b64img(real if real is not None else ann, ".jpg", 90),
        "yolo": {"masks_data_shape": list(np.asarray(raw_masks).shape) if raw_masks else None,
                 "raw": raw, "letterbox": letterbox_check(result, raw_masks, h, w),
                 "join": Tj, "suppress": Ts,
                 "final_scores": [R(x, 4) for x in yolo_scores]},
        "cv": Tc,
        "cables": cables,
        "checks": {"join": Tj["verified"], "suppress": Ts["verified"], "cv": Tc["verified"],
                   "pickup": all(c.get("pickup", {}).get("verified", True) for c in cables),
                   "infer_pixels_different": diff},
        "sources": sources(),
    }


def embed(html_path, traces):
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()
    payload = json.dumps(traces, separators=(",", ":")).replace("</", "<\\/")
    pattern = re.compile(r'(<script id="embedded-traces" type="application/json">)(.*?)(</script>)', re.S)
    if not pattern.search(html):
        raise ValueError(f"No embedded-traces block found in {html_path}")
    html = pattern.sub(lambda m: m.group(1) + payload + m.group(3), html, count=1)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    ap = argparse.ArgumentParser(description="Export pipeline traces for pipeline_explorer.html")
    ap.add_argument("--image", action="append", required=True, help="Input photo (repeatable)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="traces", help="Folder for trace_<name>.json")
    ap.add_argument("--embed", default=None, help="Also embed the traces into this explorer HTML")
    args = ap.parse_args()

    model = MODEL(yaml_path=args.config)
    os.makedirs(args.out, exist_ok=True)
    traces = []
    for path in args.image:
        t = trace_image(model, path)
        out = os.path.join(args.out, f"trace_{t['name']}.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(t, f, separators=(",", ":"))
        model.logger.info(f"Wrote {out} ({os.path.getsize(out) / 1e6:.2f} MB) checks={t['checks']}")
        traces.append(t)
    if args.embed:
        embed(args.embed, traces)
        model.logger.info(f"Embedded {len(traces)} trace(s) into {args.embed}")


if __name__ == "__main__":
    main()

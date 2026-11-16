#!/usr/bin/env python3
"""
make_figures.py - draw every figure of the pipeline report from the exact traces.

Inputs : traces/trace_image84.json, traces/trace_image85.json (from trace_export.py)
         + the original photos they point to.
Outputs: docs/pipeline_report/figures/*.png and figures/numbers.json
         (every derived number quoted in the report).

Nothing here changes the pipeline: intermediate arrays are recomputed with the
same helpers from cv_masks.py / utils.py on the masks recorded in the trace.

Run from the project folder:  python3 docs/pipeline_report/make_figures.py
"""
import json
import os
import sys

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle, Arc
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
import cv_masks as cm  # noqa: E402
import utils as U  # noqa: E402

OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "font.family": "DejaVu Serif", "mathtext.fontset": "cm", "font.size": 8.5,
    "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.linewidth": 0.6,
    "savefig.dpi": 220, "figure.dpi": 100,
})
YOLO, CVG, AMB, RED, YEL = (0.18, 0.33, 0.90), (0.10, 0.62, 0.30), (0.95, 0.66, 0.0), (0.86, 0.20, 0.18), (1.0, 0.92, 0.0)
PAL = [(1.0, 0.39, 0.28), (1.0, 0.70, 0.0), (0.0, 0.68, 0.94), (0.69, 0.32, 0.87), (1.0, 0.18, 0.57), (0.0, 0.78, 0.75)]
NUM = {}


# --------------------------------------------------------------------------- helpers
def rle(o):
    h, w = o["shape"]
    a = np.zeros(h * w, np.uint8)
    p, v = 0, 0
    for r in o["rle"]:
        if v:
            a[p:p + r] = 1
        p += r
        v ^= 1
    return a.reshape(h, w)


def rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("wrote", name)


def show(ax, im, cmap=None, **kw):
    ax.imshow(im, cmap=cmap, interpolation="nearest", **kw)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.5)


def zoom(ax, box, pad=0, clamp=True):
    x0, y0, x1, y1 = box
    x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
    if clamp:
        if y0 < 0:
            y1, y0 = y1 - y0, 0
        if x0 < 0:
            x1, x0 = x1 - x0, 0
        x1, y1 = min(x1, 1280), min(y1, 720)
    ax.set_xlim(x0 - 0.5, x1 - 0.5)
    ax.set_ylim(y1 - 0.5, y0 - 0.5)


def tint(ax, mask, color, alpha=0.45):
    rgba = np.zeros(mask.shape + (4,), np.float32)
    rgba[mask > 0] = (*color, alpha)
    ax.imshow(rgba, interpolation="nearest")


def tag(ax, x, y, s, color="white", fs=7, ha="left", va="bottom", bg=(0, 0, 0, 0.6)):
    ax.text(x, y, s, color=color, fontsize=fs, ha=ha, va=va,
            bbox=dict(boxstyle="round,pad=0.18", fc=bg, ec="none"))


def panel(ax, letter):
    ax.text(0.012, 0.975, f"({letter})", transform=ax.transAxes, fontsize=8.5, fontweight="bold",
            color="black", va="top", ha="left", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))


def polyline(ax, pts, color, lw=1.2, ls="-", closed=True, alpha=1):
    p = np.asarray(pts, float)
    if closed:
        p = np.vstack([p, p[:1]])
    ax.plot(p[:, 0], p[:, 1], color=color, lw=lw, ls=ls, alpha=alpha)


def bbox_of(o, pad=0):
    x0, y0, x1, y1 = o["bbox"]
    return [x0 - pad, y0 - pad, x1 + pad, y1 + pad]


def union_box(*boxes):
    b = np.array(boxes, float)
    return [b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()]


def pts_box(pts, pad):
    p = np.asarray(pts, float)
    return [p[:, 0].min() - pad, p[:, 1].min() - pad, p[:, 0].max() + pad, p[:, 1].max() + pad]


# --------------------------------------------------------------------------- load
def load(name):
    T = json.load(open(os.path.join(ROOT, "traces", f"trace_{name}.json")))
    img = cv2.imread(os.path.join(ROOT, T["image_path"]))
    S = T["settings"]
    yolo = [c for c in T["cables"] if c["source"] == "yolo"]
    cvc = [c for c in T["cables"] if c["source"] == "cv"]
    ym = [rle(c["mask"]) for c in yolo]
    cvm = [rle(c["mask"]) for c in cvc]
    ann = cm.overlay_masks(img, ym, color=(255, 0, 0), alpha=0.45, scores=[c["score"] for c in yolo])
    ann = cm.overlay_cv_masks(ann, cvm)
    ann = U.draw_pickup_points(ann, [c["pickup"]["point"] for c in T["cables"]], radius=S["pickup_marker_radius"], color=(0, 255, 255))
    T["_img"], T["_ann"] = img, ann
    return T


T = load("image84")
T85 = load("image85")
S = T["settings"]
IMG, ANN = T["_img"], T["_ann"]
H, W = IMG.shape[:2]
GRAY = cv2.cvtColor(IMG, cv2.COLOR_BGR2GRAY)
J, CV, Y = T["yolo"]["join"], T["cv"], T["yolo"]
CAB = next(c for c in T["cables"] if c["source"] == "cv")          # the top, OpenCV-only cable
K = CAB["pickup"]
CMASK = rle(CAB["mask"])
cand_kept = next(k for k, c in enumerate(CV["candidates"]) if c["result"] == "kept")
KEPT_MASK = rle(CV["candidates"][cand_kept]["mask"])
TOP_PIECES = [p["id"] for p in CV["pieces"] if KEPT_MASK[int(round(p["centroid"][1])), int(round(p["centroid"][0]))]]
NUM["top_pieces"] = TOP_PIECES
NUM["settings"] = S

# exact recomputation for the pickup cable
PTS, AXIS, PROJ, T0, T1, M = U._mask_pca_axis(CMASK)
MEAN = PTS.mean(axis=0)
DIST = cv2.distanceTransform(M, cv2.DIST_L2, 5)
NRM = np.array([-AXIS[1], AXIS[0]])
TMAP = np.full(CMASK.shape, np.nan, np.float32)
TMAP[PTS[:, 1].astype(int), PTS[:, 0].astype(int)] = PROJ
SAMP, RAD, GPROF = np.array(K["samples"]), np.array(K["radii"]), np.array(K["gray"])
CB = bbox_of(CAB["mask"])


def at_t(t, u=0.0):
    return MEAN + t * AXIS + u * NRM


def band_poly(t, b, L=60):
    a, c = at_t(t - b), at_t(t + b)
    return np.array([a + L * NRM, a - L * NRM, c - L * NRM, c + L * NRM])


def straight_strip(img, half=42, step=1.0):
    ts = np.arange(T0, T1 + step, step)
    us = np.arange(-half, half + 1, 1.0)
    tt, uu = np.meshgrid(ts, us)
    mx = (MEAN[0] + tt * AXIS[0] + uu * NRM[0]).astype(np.float32)
    my = (MEAN[1] + tt * AXIS[1] + uu * NRM[1]).astype(np.float32)
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), ts, us


STRIP, STS, SUS = straight_strip(rgb(IMG))
NUM["pickup"] = {k: K[k] for k in ["wire_ref", "n_positive", "clip", "step", "extra", "span", "margin", "side", "bad", "pick_i", "pick_t", "final", "point", "pca"]}
NUM["pickup"]["n_samples"] = len(SAMP)
NUM["pickup"]["tries"] = [{k: v for k, v in tr.items() if k != "walk"} | {"walk_len": len(tr["walk"])} for tr in K["tries"]]
NUM["pickup"]["demo"] = K["demo"]

# =========================================================================== FIGURES
# F01 overview -------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.15))
show(ax[0], rgb(IMG)); ax[0].set_title("Input photo (image84, 720 × 1280)")
show(ax[1], rgb(ANN)); ax[1].set_title("Output of infer(): blue = YOLO, green = OpenCV, yellow = pickup")
panel(ax[0], "a"); panel(ax[1], "b")
fig.tight_layout(w_pad=0.6)
save(fig, "f01_overview.png")

# F02 block diagram ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 2.5))
ax.set_xlim(0, 100); ax.set_ylim(0, 34); ax.axis("off")


def box(x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=(*color, 0.10), ec=color, lw=1.1))
    ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=8.2, fontweight="bold", color="black")
    ax.text(x + w / 2, y + h - 5.6, body, ha="center", va="top", fontsize=6.3, color="0.15", linespacing=1.3)


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, lw=1.0, color="0.3"))


box(0.5, 19, 12, 14, "Photo", "BGR array\n720×1280×3", (0.4, 0.4, 0.4))
box(16.5, 19, 26, 14, "Stage 1 · YOLO", "model.predict\n→ masks + scores\njoin · suppress", YOLO)
box(46.5, 19, 26, 14, "Stage 2 · OpenCV", "gray → blur → threshold\ncontours → bridges\nkeep if no YOLO overlap", CVG)
box(76.5, 19, 23, 14, "Overlay", "blue = YOLO\ngreen = OpenCV\nα = 0.45", (0.4, 0.4, 0.4))
box(16.5, 1, 56, 14, "Stage 3 · Pickup point (every mask)", "PCA axis → t → distance transform → profiles\n→ clip → sticker side → walk off bulge\n→ +45 px → checks → centre pixel", AMB)
box(76.5, 1, 23, 14, "Result", "annotated image\n+ yellow markers", (0.4, 0.4, 0.4))
arrow(13.4, 26, 16.1, 26); arrow(43.0, 26, 46.1, 26); arrow(73.0, 26, 76.1, 26)
arrow(59.5, 18.6, 59.5, 15.4); arrow(73.0, 8, 76.1, 8)
save(fig, "f02_pipeline.png")

# F03 YOLO raw masks ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 4.1))
show(ax, rgb(IMG))
for i, it in enumerate(J["items"]):
    tint(ax, rle(it["resized"]), PAL[i], 0.5)
    b = Y["raw"][i]["box"]
    tag(ax, b[0] - 8, (b[1] + b[3]) / 2, f"mask {i}\ns = {Y['raw'][i]['score']:.3f}", color="white", fs=7, ha="right", va="center", bg=(*PAL[i], 0.9))
cb = bbox_of(CV["candidates"][cand_kept]["mask"], 4)
ax.add_patch(Rectangle((cb[0], max(cb[1], 1)), cb[2] - cb[0], cb[3] - max(cb[1], 1), fill=False, ec=CVG, lw=1.6, ls="--"))
tag(ax, cb[2] - 4, cb[3] + 2, "no YOLO mask here", color="white", fs=7.5, ha="right", va="top", bg=(*CVG, 0.9))
save(fig, "f03_yolo_raw.png")
NUM["raw_scores"] = [r["score"] for r in Y["raw"]]
NUM["masks_data_shape"] = Y["masks_data_shape"]
NUM["letterbox"] = Y["letterbox"]

# F04 resize ----------------------------------------------------------------------------
fig = plt.figure(figsize=(7.2, 2.35))
gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.25, 0.9], wspace=0.12)
a0, a1, a2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
nat = rle(Y["raw"][1]["native"])
show(a0, nat, cmap="gray"); a0.set_title(f"masks.data[1]: {nat.shape[0]} × {nat.shape[1]}", fontsize=8)
res = rle(J["items"][1]["resized"])
show(a1, rgb(IMG)); tint(a1, res, PAL[1], 0.55); a1.set_title(f"after _resize_mask: {H} × {W}", fontsize=8)
rng = np.random.default_rng(3)
small = (rng.random((4, 4)) > 0.45).astype(np.uint8)
big = np.kron(small, np.ones((2, 2), np.uint8))
a2.axis("off")
ia = a2.inset_axes([0.0, 0.25, 0.38, 0.5]); ib = a2.inset_axes([0.5, 0.05, 0.5, 0.9])
for axx, arr, ttl in [(ia, small, "4 × 4"), (ib, big, "8 × 8")]:
    axx.imshow(arr, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axx.set_xticks(np.arange(-0.5, arr.shape[1], 1), minor=True); axx.set_yticks(np.arange(-0.5, arr.shape[0], 1), minor=True)
    axx.grid(which="minor", color=(0.6, 0.6, 0.6), lw=0.4); axx.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
    axx.set_title(ttl, fontsize=7.5)
a2.annotate("", xy=(0.49, 0.5), xytext=(0.4, 0.5), xycoords="axes fraction", arrowprops=dict(arrowstyle="-|>", color="0.3"))
a2.text(0.5, 0.97, "nearest neighbour ×2:\ncopy, never blend", transform=a2.transAxes, ha="center", va="top", fontsize=7.5)
panel(a0, "a"); panel(a1, "b")
save(fig, "f04_resize.png")

# F05 piece geometry (exact, on one YOLO piece) ------------------------------------
pc = J["pieces"][0]
m0 = rle(J["items"][pc["raw"]]["resized"])
ctrs, _ = cv2.findContours(m0, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cont = max(ctrs, key=cv2.contourArea)
vx, vy, x0, y0 = cv2.fitLine(cont, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
dvec = np.array([vx, vy], float)
cpts = cont.reshape(-1, 2).astype(float)
tt = (cpts - np.array([x0, y0])) @ dvec
lo_sel, hi_sel = tt < tt.min() + 5, tt > tt.max() - 5
(ends, _d) = cm._piece_ends(cont)
rect = cv2.boxPoints(cv2.minAreaRect(cont))
(_c, (rw, rh), _a) = cv2.minAreaRect(cont)
mom = cv2.moments(cont)
cen = np.array([mom["m10"] / mom["m00"], mom["m01"] / mom["m00"]])
fig = plt.figure(figsize=(7.2, 2.7))
gs = fig.add_gridspec(1, 2, width_ratios=[2.3, 1], wspace=0.06)
a0, a1 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
for axx in (a0, a1):
    show(axx, rgb(IMG)); tint(axx, m0, PAL[0], 0.35)
    polyline(axx, cpts, "white", lw=0.9)
    polyline(axx, rect, (0.2, 0.9, 1.0), lw=1.0, ls="--")
    L = 0.6 * (tt.max() - tt.min())
    axx.plot([x0 - dvec[0] * L, x0 + dvec[0] * L], [y0 - dvec[1] * L, y0 + dvec[1] * L], color=AMB, lw=1.1, ls=(0, (6, 3)))
    axx.scatter(cpts[lo_sel, 0], cpts[lo_sel, 1], s=4, color=RED, zorder=3)
    axx.scatter(cpts[hi_sel, 0], cpts[hi_sel, 1], s=4, color=RED, zorder=3)
    axx.scatter([e[0] for e in ends], [e[1] for e in ends], s=34, color=AMB, edgecolor="black", lw=0.6, zorder=4)
    axx.scatter([cen[0]], [cen[1]], s=40, marker="X", color="white", edgecolor="black", lw=0.6, zorder=4)
zoom(a0, pts_box(cpts, 25))
zoom(a1, [ends[0][0] - 45, ends[0][1] - 45, ends[0][0] + 35, ends[0][1] + 45])
tag(a0, cpts[:, 0].min() + 5, cpts[:, 1].max() + 22, f"minAreaRect (cyan): {max(rw, rh):.1f} × {min(rw, rh):.1f} px", color=(0.2, 0.9, 1.0), fs=7, va="bottom")
tag(a0, x0 + dvec[0] * 120, y0 + dvec[1] * 120 - 10, f"fitLine d = ({dvec[0]:.4f}, {dvec[1]:.4f})", color=AMB, fs=7)
tag(a0, cen[0] + 10, cen[1] + 30, "centroid", fs=7, va="top")
tag(a1, ends[0][0] - 40, ends[0][1] + 40, "points with t < t_min + 5 (red)\n→ averaged into the end (amber)", fs=6.5, va="bottom")
panel(a0, "a"); panel(a1, "b")
save(fig, "f05_piece_geometry.png")
NUM["piece0"] = {"dir": dvec.tolist(), "p0": [float(x0), float(y0)], "ends": [list(map(float, e)) for e in ends], "length": float(max(rw, rh)),
                 "width": float(min(rw, rh)), "centroid": cen.tolist(), "n_lo": int(lo_sel.sum()), "n_hi": int(hi_sel.sum()), "n_contour": len(cpts), "area": float(cv2.contourArea(cont))}

# F06 schematic of the join tests --------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 2.6))
ax.set_aspect("equal"); ax.axis("off")
di = np.array([1, 0.05]); di /= np.linalg.norm(di)
dj = np.array([1, 0.11]); dj /= np.linalg.norm(dj)
A0, Aend = np.array([0, 0]), np.array([4.2, 0.21])
Bst, B1 = np.array([7.0, 1.45]), np.array([11.2, 1.91])


def bar(p, q, c):
    v = q - p; n = np.array([-v[1], v[0]]) / np.linalg.norm(v) * 0.32
    ax.add_patch(Polygon([p + n, q + n, q - n, p - n], closed=True, fc=(*c, 0.25), ec=c, lw=1.0))


bar(A0, Aend, YOLO); bar(Bst, B1, YOLO)
g = Bst - Aend
axv = (di + dj) / np.linalg.norm(di + dj)
nv = np.array([-axv[1], axv[0]])
ax.add_patch(FancyArrowPatch(Aend, Bst, arrowstyle="-|>", mutation_scale=10, lw=1.6, color=RED))
ax.add_patch(FancyArrowPatch(A0 + di * 1.0, A0 + di * 2.6, arrowstyle="-|>", mutation_scale=9, lw=1.2, color="black"))
ax.add_patch(FancyArrowPatch(Bst + dj * 1.6, Bst + dj * 3.2, arrowstyle="-|>", mutation_scale=9, lw=1.2, color="black"))
along = Aend + axv * (g @ axv)
ax.plot([Aend[0], along[0]], [Aend[1], along[1]], color="0.4", lw=0.9, ls="--")
ax.plot([along[0], Bst[0]], [along[1], Bst[1]], color=CVG, lw=1.6)
ax.add_patch(Arc(Aend, 3.0, 3.0, angle=0, theta1=np.degrees(np.arctan2(di[1], di[0])), theta2=np.degrees(np.arctan2(g[1], g[0])), color=AMB, lw=1.2))
ax.plot([Aend[0], Aend[0] + di[0] * 2.0], [Aend[1], Aend[1] + di[1] * 2.0], color=AMB, lw=0.8, ls=":")
ax.scatter(*Aend, s=28, color=AMB, edgecolor="black", zorder=4); ax.scatter(*Bst, s=28, color=AMB, edgecolor="black", zorder=4)
ci, cj = (A0 + Aend) / 2, (Bst + B1) / 2
ax.scatter(*ci, marker="X", s=40, color="white", edgecolor="black", zorder=4); ax.scatter(*cj, marker="X", s=40, color="white", edgecolor="black", zorder=4)
ax.text(*(Aend + np.array([0.0, -0.6])), r"$\mathbf{a}$", fontsize=11, ha="center")
ax.text(*(Bst + np.array([-0.2, 0.45])), r"$\mathbf{b}$", fontsize=11, ha="center")
ax.text(*(A0 + di * 1.8 + np.array([0, 0.45])), r"$\mathbf{d}_i$", fontsize=11, ha="center")
ax.text(*(Bst + dj * 2.4 + np.array([0, 0.45])), r"$\mathbf{d}_j$", fontsize=11, ha="center")
ax.text(*((Aend + Bst) / 2 + np.array([-0.3, 0.3])), r"$\mathbf{g}=\mathbf{b}-\mathbf{a}$", fontsize=10.5, color=RED, ha="right")
ax.text(Aend[0] + 1.62, Aend[1] + 0.22, r"$\theta_i$", fontsize=10, color=AMB, va="center")
ax.text(along[0] + 0.15, (along[1] + Bst[1]) / 2 - 0.1, r"$\ell=|\mathbf{g}\cdot\mathbf{n}|$", fontsize=10, color=CVG, va="center")
ax.text(*(ci + np.array([0, -0.75])), r"$\mathbf{c}_i$", fontsize=10.5, ha="center")
ax.text(*(cj + np.array([0, -0.75])), r"$\mathbf{c}_j$", fontsize=10.5, ha="center")
ax.text(0.0, 0.6, "piece i", fontsize=8, color=YOLO); ax.text(10.0, 2.45, "piece j", fontsize=8, color=YOLO)
ax.text(12.4, 2.3, "Four tests, all must pass:\n"
        r"1. gap $\|\mathbf{g}\|\leq G=120$ px" "\n"
        r"2. angles $\theta_i,\ \theta_j\leq 25^\circ$" "\n"
        r"3. lateral $\ell\leq 40$ px" "\n"
        r"4. centroid offset $|(\mathbf{c}_j-\mathbf{c}_i)\cdot\mathbf{n}|\leq 60$ px",
        fontsize=8.2, va="top", linespacing=1.6)
ax.set_xlim(-0.6, 19.2); ax.set_ylim(-1.0, 2.8)
save(fig, "f06_join_schematic.png")

# helpers for real pair figures ------------------------------------------------------
BIN = cv2.threshold(cv2.GaussianBlur(GRAY, (CV["ksize"],) * 2, 0), int(S["cv_binary_threshold"]), 255, cv2.THRESH_BINARY_INV)[1]
_pieces_all = [c for c in cv2.findContours(BIN, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0] if cv2.contourArea(c) >= float(S["cv_min_piece_area"])]
BRIDGED = BIN.copy()
cv2.drawContours(BRIDGED, _pieces_all, -1, 255, cv2.FILLED)
cm._bridge_pieces_on_mask(BRIDGED, _pieces_all, S["cv_max_gap_px"], S["cv_max_join_angle_deg"], S["cv_join_width_px"], min_length=float(S["mask_width_threshold_px"]))
FILLED = np.zeros_like(BIN); cv2.drawContours(FILLED, _pieces_all, -1, 255, cv2.FILLED)


def draw_pair(ax, P, base, show_checks=True, title=None):
    pi, pj = CV["pieces"][P["i"]], CV["pieces"][P["j"]]
    show(ax, base, cmap="gray")
    for p, c, dy in ((pi, PAL[2], 26), (pj, PAL[3], -16)):
        polyline(ax, p["poly"], c, lw=1.2)
        tag(ax, p["centroid"][0], p["centroid"][1] + dy, f"piece {p['id']}  L = {p['length']:.0f}", color="white", fs=6.8, ha="center", bg=(*c, 0.85))
        ax.scatter([e[0] for e in p["ends"]], [e[1] for e in p["ends"]], s=22, color=AMB, edgecolor="black", lw=0.5, zorder=4)
    if show_checks:
        for ci_, c in enumerate(P.get("checks", [])):
            ok = c.get("fail") is None
            col = CVG if ok else RED
            ax.plot([c["pa"][0], c["pb"][0]], [c["pa"][1], c["pb"][1]], color=col, lw=1.4, ls="-" if ok else (0, (4, 2)))
            fr = [0.5, 0.72, 0.28, 0.5][ci_ % 4] if c.get("fail") == "gap" else 0.5
            mx, my = c["pa"][0] + fr * (c["pb"][0] - c["pa"][0]), c["pa"][1] + fr * (c["pb"][1] - c["pa"][1])
            if c.get("fail") == "gap":
                s = f"{c['dist']:.0f} > 120"
            elif c.get("fail") == "angle":
                s = f"{c['dist']:.0f} px, θ = {c['ang_i']:.0f}° / {c['ang_j']:.0f}° > 25°"
            elif c.get("fail") == "lateral":
                s = f"$\ell$ = {c['lateral']:.0f} > 40"
            else:
                s = f"{c['dist']:.1f} px, θ = {c['ang_i']:.1f}° / {c['ang_j']:.1f}°, $\ell$ = {c['lateral']:.1f}"
            tag(ax, mx, my + 8, s, color="white", fs=6.4, ha="center", va="top", bg=(*col, 0.88))
    box_ = pts_box(pi["poly"] + pj["poly"], 30)
    zoom(ax, box_)
    if title:
        ax.set_title(title)


# F07 real join (top cable) ----------------------------------------------------------
join_top = next(p for p in CV["pairs"] if p["result"] == "join" and p["i"] in TOP_PIECES and p["j"] in TOP_PIECES)
fig, ax = plt.subplots(2, 1, figsize=(7.2, 4.0))
draw_pair(ax[0], join_top, FILLED, title=f"Pieces {join_top['i']} and {join_top['j']} (top cable): the four end pairs")
draw_pair(ax[1], join_top, BRIDGED, show_checks=False, title=f"After cv2.line(..., thickness = {S['cv_join_width_px']}): one blob")
a, b = join_top["pair"]
ax[1].plot([a[0], b[0]], [a[1], b[1]], color=AMB, lw=1.0, ls="--")
panel(ax[0], "a"); panel(ax[1], "b")
fig.tight_layout(h_pad=0.6)
save(fig, "f07_join_real.png")
NUM["join_top"] = join_top

# F08 rejected pair (angle) ------------------------------------------------------------
rej = sorted([p for p in CV["pairs"] if p["result"] == "ends" and p.get("checks") and any(c.get("fail") == "angle" for c in p["checks"])], key=lambda p: p["min_dist"])[0]
fig, ax = plt.subplots(figsize=(7.2, 2.6))
draw_pair(ax, rej, FILLED, title=f"Pieces {rej['i']} and {rej['j']}: ends are close, but the gap points across the cables")
save(fig, "f08_reject_angle.png")
NUM["reject_angle"] = rej
rejg = sorted([p for p in CV["pairs"] if p["result"] == "ends" and p.get("checks") and all(c.get("fail") == "gap" for c in p["checks"])], key=lambda p: p["min_dist"])[0]
NUM["reject_gap"] = rejg

# F09 grouping graph + overlap schematic ----------------------------------------------
fig = plt.figure(figsize=(7.2, 2.75))
gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1], wspace=0.05)
a0, a1 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
show(a0, (FILLED * 0.35).astype(np.uint8), cmap="gray", vmin=0, vmax=255)
cents = {p["id"]: p["centroid"] for p in CV["pieces"]}
for p in CV["pairs"]:
    A_, B_ = cents[p["i"]], cents[p["j"]]
    if p["result"] == "join":
        a0.plot([A_[0], B_[0]], [A_[1], B_[1]], color=CVG, lw=2.2, zorder=3)
    elif p["result"] == "ends" and p.get("min_dist", 999) < 150:
        a0.plot([A_[0], B_[0]], [A_[1], B_[1]], color=RED, lw=1.0, ls=(0, (3, 2)), zorder=2)
    elif p["result"] == "both_long":
        a0.plot([A_[0], B_[0]], [A_[1], B_[1]], color="0.55", lw=0.4, ls=":", zorder=1)
for pid, c in cents.items():
    a0.scatter(*c, s=70, color="white", edgecolor="black", lw=0.6, zorder=4)
    a0.text(c[0], c[1], str(pid), fontsize=6.5, ha="center", va="center", zorder=5)
a0.legend(handles=[Line2D([], [], color=CVG, lw=2.2, label="join (edge in the graph)"), Line2D([], [], color=RED, lw=1, ls="--", label="rejected, ends < 150 px apart"),
                   Line2D([], [], color="0.55", lw=0.6, ls=":", label="both long, skipped")], loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=3, framealpha=0.9, fontsize=6.6)
a0.set_title("Nodes = pieces, edges = accepted joins; groups = connected components", fontsize=8)
a1.set_aspect("equal"); a1.axis("off")
a1.add_patch(Ellipse((0.36, 0.5), 0.56, 0.36, fc=(*YOLO, 0.25), ec=YOLO, lw=1.2))
a1.add_patch(Ellipse((0.64, 0.5), 0.52, 0.30, fc=(*AMB, 0.25), ec=AMB, lw=1.2))
a1.text(0.18, 0.5, r"$M_a$", fontsize=11, ha="center", va="center"); a1.text(0.84, 0.5, r"$M_b$", fontsize=11, ha="center", va="center")
a1.text(0.5, 0.5, r"$M_a\cap M_b$", fontsize=8, ha="center", va="center")
a1.text(0.5, 0.2, r"$r=\frac{|M_a\cap M_b|}{\min(|M_a|,\ |M_b|)}$", fontsize=10, ha="center")
a1.text(0.5, 0.04, r"$r\geq 0.1\ \Rightarrow$ drop lower score", fontsize=7.5, ha="center")
a1.set_xlim(0, 1); a1.set_ylim(0, 0.9); a1.set_title("Overlap suppression")
panel(a0, "a"); panel(a1, "b")
save(fig, "f09_groups_overlap.png")
NUM["cv_pairs"] = {r: sum(1 for p in CV["pairs"] if p["result"] == r) for r in set(p["result"] for p in CV["pairs"])}
NUM["yolo_pairs"] = {r: sum(1 for p in J["pairs"] if p["result"] == r) for r in set(p["result"] for p in J["pairs"])}
NUM["suppress"] = T["yolo"]["suppress"]["steps"]

# F10 grayscale / blur / threshold ----------------------------------------------------
BLUR = cv2.GaussianBlur(GRAY, (CV["ksize"],) * 2, 0)
crop = [430, 40, 590, 120]
fig, ax = plt.subplots(2, 3, figsize=(7.2, 3.35), gridspec_kw=dict(height_ratios=[1, 0.62]))
for j, (im, ttl) in enumerate([(GRAY, "grayscale Y"), (BLUR, f"Gaussian blur {CV['ksize']}×{CV['ksize']}"), (BIN, f"inverse threshold at {S['cv_binary_threshold']}")]):
    show(ax[0, j], im, cmap="gray", vmin=0, vmax=255); ax[0, j].set_title(ttl)
    ax[0, j].add_patch(Rectangle((crop[0], crop[1]), crop[2] - crop[0], crop[3] - crop[1], fill=False, ec=AMB, lw=1.0))
    show(ax[1, j], im, cmap="gray", vmin=0, vmax=255); zoom(ax[1, j], crop)
for j, l in enumerate("abc"):
    panel(ax[0, j], l)
fig.tight_layout(h_pad=0.3, w_pad=0.4)
save(fig, "f10_preprocess.png")
k1 = cv2.getGaussianKernel(CV["ksize"], 0).ravel()
NUM["gauss"] = {"k1d": k1.tolist(), "sigma": 0.3 * ((CV["ksize"] - 1) * 0.5 - 1) + 0.8, "k2d_center": float(k1[2] * k1[2]), "k2d_corner": float(k1[0] * k1[0])}
px = [int(round(K["point"][0])), int(round(K["point"][1]))]
lab = [470, 78]
NUM["gray_examples"] = {"wire_px": px, "wire_bgr": IMG[px[1], px[0]].tolist(), "wire_Y": int(GRAY[px[1], px[0]]), "wire_blur": int(BLUR[px[1], px[0]]),
                        "label_px": lab, "label_bgr": IMG[lab[1], lab[0]].tolist(), "label_Y": int(GRAY[lab[1], lab[0]]), "label_blur": int(BLUR[lab[1], lab[0]])}

# F11 histogram -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 1.9))
hist = np.bincount(BLUR.ravel(), minlength=256)
ax.bar(np.arange(256), hist, width=1.0, color=["0.15" if v <= S["cv_binary_threshold"] else "0.65" for v in range(256)])
ax.axvline(S["cv_binary_threshold"], color=RED, lw=1.2)
ax.set_yscale("log"); ax.set_xlim(0, 255); ax.set_xlabel("blurred gray value B(x, y)"); ax.set_ylabel("pixels (log)")
frac = float((BLUR <= S["cv_binary_threshold"]).mean())
ax.text(S["cv_binary_threshold"] + 3, hist.max() * 0.5, f"τ = {S['cv_binary_threshold']}:  {frac * 100:.1f} % of pixels ≤ τ → white (255)", color=RED, fontsize=7.5, va="top")
ax.spines[["top", "right"]].set_visible(False)
save(fig, "f11_histogram.png")
NUM["threshold"] = {"frac_white": frac, "white_px": int((BIN > 0).sum())}

# F12 contours / pieces ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 4.1))
show(ax, (BIN * 0.45).astype(np.uint8), cmap="gray", vmin=0, vmax=255)
for p in CV["small_polys"]:
    polyline(ax, p, RED, lw=0.8, ls=(0, (2, 1.5)))
for i, p in enumerate(CV["pieces"]):
    c = PAL[i % len(PAL)]
    ax.add_patch(Polygon(p["poly"], closed=True, fc=(*c, 0.35), ec=c, lw=1.0))
    tag(ax, p["centroid"][0], p["centroid"][1], f"{p['id']}", color="black", fs=7, ha="center", va="center", bg=(1, 1, 1, 0.85))
ax.set_title(f"{CV['n_contours']} contours: {len(CV['pieces'])} pieces ≥ {S['cv_min_piece_area']} px² (coloured), {CV['n_small']} smaller (red dashes, dropped)")
save(fig, "f12_pieces.png")
NUM["cv_pieces"] = [{k: p[k] for k in ["id", "length", "contour_area", "centroid"]} for p in CV["pieces"]]
NUM["cv_counts"] = {"n_contours": CV["n_contours"], "n_small": CV["n_small"], "n_pieces": len(CV["pieces"])}

# F13 bridged + size filter -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 4.1))
show(ax, (BRIDGED * 0.55).astype(np.uint8), cmap="gray", vmin=0, vmax=255)
for b in CV["bridges"]:
    ax.plot([b["a"][0], b["b"][0]], [b["a"][1], b["b"][1]], color=AMB, lw=3.2, solid_capstyle="round")
for r in CV["cables"]:
    c = CVG if r["result"] == "pass" else RED
    polyline(ax, r["poly"], c, lw=1.1)
    pp = np.array(r["poly"])
    tag(ax, pp[:, 0].mean(), pp[:, 1].max() + 3, f"L = {r['length']:.0f} px,  A = {r['area']:.0f} px²  pass", color="white", fs=6.8, ha="center", va="top", bg=(*c, 0.85))
ax.set_title(f"{len(CV['bridges'])} bridges (amber) turn {len(CV['pieces'])} pieces into {len(CV['cables'])} blobs; all pass A ≥ {S['cv_min_cable_area']} and L ≥ {S['mask_width_threshold_px']}")
save(fig, "f13_bridged.png")
NUM["cv_cables"] = [{k: r[k] for k in ["area", "length", "result"]} for r in CV["cables"]]

# F14 YOLO union vs candidates ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 4.1))
show(ax, rgb(IMG)); ax.imshow(np.zeros_like(IMG), alpha=0.35)
tint(ax, rle(CV["union"]), YOLO, 0.5)
for k, c in enumerate(CV["candidates"]):
    mk = rle(c["mask"])
    ctr = max(cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea).reshape(-1, 2)
    kept = c["result"] == "kept"
    if kept:
        tint(ax, mk, CVG, 0.6)
    polyline(ax, ctr, CVG if kept else RED, lw=1.3)
    bb = c["mask"]["bbox"]
    tag(ax, bb[2] - 4, (bb[1] + bb[3]) / 2, f"candidate {k}: |C ∩ U| = {c['overlap_px']:,} px → {'keep' if kept else 'reject'}",
        color="white", fs=6.8, ha="right", va="center", bg=(*(CVG if kept else RED), 0.88))
ax.set_title("Blue: union U of all YOLO masks.  Outlines: OpenCV candidates C.  Keep C only if C ∩ U is empty.")
save(fig, "f14_union_test.png")
NUM["candidates"] = [{"overlap_px": c["overlap_px"], "result": c["result"], "bbox": c["mask"]["bbox"]} for c in CV["candidates"]]
NUM["union_area"] = CV["union"]["area"]

# F15 alpha blending example ------------------------------------------------------------
ycab = next(c for c in T["cables"] if c["source"] == "yolo")
ym0 = rle(ycab["mask"])
ys_, xs_ = np.where(ym0 > 0)
pick_ix = len(xs_) // 2
bx, by = int(xs_[pick_ix]), int(ys_[pick_ix])
over = IMG.copy(); over[ym0 > 0] = (255, 0, 0)
blend = cv2.addWeighted(over, 0.45, IMG, 0.55, 0)
crop2 = [bx - 60, by - 30, bx + 60, by + 30]
fig, ax = plt.subplots(1, 3, figsize=(7.2, 1.55))
for j, (im, ttl) in enumerate([(IMG, "photo I"), (over, "overlay O (mask painted blue)"), (blend, "0.45·O + 0.55·I")]):
    show(ax[j], rgb(im)); zoom(ax[j], crop2); ax[j].set_title(ttl, fontsize=8)
    ax[j].scatter([bx], [by], s=30, facecolor="none", edgecolor=YEL, lw=1.2)
fig.tight_layout(w_pad=0.3)
save(fig, "f15_blend.png")
NUM["blend"] = {"px": [bx, by], "I": IMG[by, bx].tolist(), "O": over[by, bx].tolist(), "out": blend[by, bx].tolist()}

# F16 PCA ----------------------------------------------------------------------------------
cov = np.cov(PTS.T.astype(np.float64), bias=True)
lam, vec = np.linalg.eigh(cov)
order = np.argsort(lam)[::-1]; lam, vec = lam[order], vec[:, order]
fig, ax = plt.subplots(figsize=(7.2, 2.35))
show(ax, rgb(IMG)); ax.imshow(np.zeros_like(IMG), alpha=0.45)
sub = PTS[::9]
ax.scatter(sub[:, 0], sub[:, 1], s=0.6, color="white", alpha=0.7)
for kk in (1, 2):
    ax.add_patch(Ellipse(MEAN, 2 * kk * np.sqrt(lam[0]), 2 * kk * np.sqrt(lam[1]), angle=np.degrees(np.arctan2(vec[1, 0], vec[0, 0])), fill=False, ec=AMB, lw=0.9, ls="--" if kk == 2 else "-"))
for kk, col, nm in [(0, AMB, r"$\mathbf{v}_1$"), (1, (0.3, 0.85, 1.0), r"$\mathbf{v}_2$")]:
    v = vec[:, kk] * (np.sign(vec[0, kk]) if kk == 0 else 1)
    L = 1.2 * np.sqrt(lam[kk]) if kk == 0 else 2.5 * np.sqrt(lam[kk])
    v = v if kk == 0 else (v if v[1] < 0 else -v)
    ax.add_patch(FancyArrowPatch(MEAN, MEAN + v * L, arrowstyle="-|>", mutation_scale=10, lw=1.8, color=col, zorder=6))
    tip = MEAN + v * L
    tag(ax, tip[0] + (6 if kk == 0 else 8), tip[1] - (10 if kk == 0 else 0), (r"$\mathbf{v}_1$ (long axis)" if kk == 0 else r"$\mathbf{v}_2$"), color=col, fs=8, va="center")
ax.scatter(*MEAN, s=40, color="white", edgecolor="black", zorder=5)
tag(ax, MEAN[0] + 6, MEAN[1] + 22, r"mean $\bar{\mathbf{p}}$", fs=7.5, va="top")
zoom(ax, [CB[0] - 15, max(0, CB[1] - 15), CB[2] + 15, CB[3] + 20])
ax.set_title(f"{len(PTS):,} mask pixels as points; ellipses at 1σ and 2σ;  λ₁ = {lam[0]:,.0f},  λ₂ = {lam[1]:,.1f}")
save(fig, "f16_pca.png")
NUM["pca"] = {"mean": MEAN.tolist(), "cov": cov.tolist(), "lambda": lam.tolist(), "v1": vec[:, 0].tolist(), "v2": vec[:, 1].tolist(), "axis": AXIS.tolist(),
              "std1": float(np.sqrt(lam[0])), "std2": float(np.sqrt(lam[1])), "t0": float(T0), "t1": float(T1), "n": int(len(PTS))}

# F17 projection --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 2.2))
show(ax, rgb(IMG)); ax.imshow(np.zeros_like(IMG), alpha=0.5)
im = ax.imshow(TMAP, cmap="viridis", interpolation="nearest")
for tv, nm in [(T0, r"$t_0$"), (0.0, r"$t=0$"), (T1, r"$t_1$")]:
    p = at_t(tv); ax.scatter(*p, s=30, color="white", edgecolor="black", zorder=5)
    ax.text(p[0], p[1] - 14, f"{nm} = {tv:.1f}" if tv else r"$\bar{\mathbf{p}}$  ($t=0$)", color="white", fontsize=8, ha="center")
a_, b_ = at_t(T0 - 25), at_t(T1 + 25)
ax.add_patch(FancyArrowPatch(a_, b_, arrowstyle="-|>", mutation_scale=10, lw=1.0, color=AMB, ls="--"))
zoom(ax, [CB[0] - 30, max(0, CB[1] - 15), CB[2] + 30, CB[3] + 22])
cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01); cbar.set_label("t (px along the axis)", fontsize=7.5)
save(fig, "f17_projection.png")

# F18 distance transform + cross-section --------------------------------------------
tsec = float(K["pick_t"])
ds = {"radius": float(RAD[K["pick_i"]])}
us = np.linspace(-12, 36, 193)
sec_pts = np.array([at_t(tsec, u) for u in us])
dvals = DIST[np.clip(np.rint(sec_pts[:, 1]).astype(int), 0, H - 1), np.clip(np.rint(sec_pts[:, 0]).astype(int), 0, W - 1)]
fig = plt.figure(figsize=(7.2, 4.0))
gs = fig.add_gridspec(2, 1, height_ratios=[1.05, 1], hspace=0.35)
a0, a1 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
dshow = np.ma.masked_where(M == 0, DIST)
show(a0, np.zeros((H, W)), cmap="gray", vmin=0, vmax=1)
imd = a0.imshow(dshow, cmap="magma", interpolation="nearest")
a0.plot(sec_pts[:, 0], sec_pts[:, 1], color="cyan", lw=1.0)
zoom(a0, [CB[0] - 10, max(0, CB[1] - 10), CB[2] + 10, CB[3] + 12])
cbar = fig.colorbar(imd, ax=a0, fraction=0.025, pad=0.01); cbar.set_label("D (px to nearest background)", fontsize=7.5)
a0.set_title(f"Distance transform D of the top cable (max {DIST.max():.1f} px)")
a1.plot(us, dvals, color="0.15", lw=1.2)
a1.axhline(ds["radius"], color=AMB, lw=0.9, ls="--"); a1.text(-11.5, ds["radius"] + 0.4, f"peak = radius ≈ {ds['radius']:.1f} px", fontsize=7.5, color=AMB)
a1.axvline(0, color="cyan", lw=0.9, ls=":"); a1.text(-0.6, 5.0, "u = 0: the PCA axis\n(not the wire centre)", fontsize=7, color="0.3", ha="right")
uc = float(us[np.argmax(dvals)]); a1.axvline(uc, color=YEL, lw=0.9, ls=":"); a1.text(uc + 0.6, 2.2, f"wire centre, u ≈ {uc:.1f}", fontsize=7, color=(0.6, 0.5, 0))
a1.fill_between(us, 0, dvals, where=dvals > 0, color=(0.6, 0.2, 0.5), alpha=0.15)
a1.set_xlabel("u (px across the cable, along the normal n)"); a1.set_ylabel("D(u)"); a1.set_title(f"Section across the wire at t = {tsec:.1f} (cyan line in (a)): a tent whose height is the radius", fontsize=8)
a1.spines[["top", "right"]].set_visible(False)
panel(a0, "a"); panel(a1, "b")
save(fig, "f18_distance.png")
NUM["distance"] = {"max": float(DIST.max()), "section_t": tsec, "section_peak": float(dvals.max()), "width_px": float((dvals > 0).sum() * (us[1] - us[0]))}

# F19 slices -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 2.25))
show(ax, np.zeros((H, W)), cmap="gray", vmin=0, vmax=1)
ax.imshow(dshow, cmap="magma", interpolation="nearest", alpha=0.85)
for d in K["demo"]:
    selm = (np.abs(TMAP - d["t"]) <= K["band_px"])
    tint(ax, selm.astype(np.uint8), (0.3, 1.0, 1.0), 0.9)
    ax.scatter(*d["best"], s=26, color=YEL, edgecolor="black", lw=0.5, zorder=5)
    tag(ax, d["best"][0], CB[3] + 6, f"k = {d['i']}: |S| = {d['count']},  r = {d['radius']:.2f},  g = {d['gray']:.1f}", fs=6.6, ha="center", va="top", bg=(0, 0.35, 0.4, 0.85))
zoom(ax, [CB[0] - 10, max(0, CB[1] - 10), CB[2] + 10, CB[3] + 26])
ax.set_title(f"Three of the {len(SAMP)} slices S_k (cyan, |t − s_k| ≤ {K['band_px']:.0f}); yellow = pixel with the largest D in the slice")
save(fig, "f19_slices.png")

# F20 profiles + straightened strip -----------------------------------------------------
ci = K["clip"]["clip_i"]
tc = SAMP[ci]
fig, ax = plt.subplots(3, 1, figsize=(7.2, 4.6), sharex=True, gridspec_kw=dict(height_ratios=[0.62, 1, 0.8], hspace=0.08))
ax[0].imshow(STRIP, extent=[STS[0], STS[-1], SUS[-1], SUS[0]], aspect="auto", interpolation="bilinear")
ax[0].set_yticks([]); ax[0].set_ylabel("u", rotation=0, labelpad=6)
ax[0].set_title("Photo resampled along the PCA axis (\"straightened\" cable), same t axis as the curves below")
for a_ in ax:
    a_.axvspan(K["clip"]["mid"] - K["clip"]["half"], K["clip"]["mid"] + K["clip"]["half"], color=AMB, alpha=0.10, lw=0)
    a_.axvspan(T0, T0 + K["margin"], color=RED, alpha=0.07, lw=0); a_.axvspan(T1 - K["margin"], T1, color=RED, alpha=0.07, lw=0)
    a_.axvline(tc, color=AMB, lw=1.0)
ax[1].plot(SAMP, RAD, color=(0.1, 0.6, 0.6), lw=1.2, label=r"radius $r_k$")
ax[1].axhline(K["wire_ref"], color=CVG, lw=0.9, ls="--"); ax[1].text(T0 + 4, K["wire_ref"] - 2.3, f"w = P40 = {K['wire_ref']:.2f}", color=CVG, fontsize=7)
ax[1].axhline(K["clip"]["thresh"], color=AMB, lw=0.9, ls="--"); ax[1].text(T0 + 4, K["clip"]["thresh"] + 0.6, f"τ = max(1.35w, w+3) = {K['clip']['thresh']:.2f}", color=AMB, fontsize=7)
ax[1].scatter(SAMP[K["clip"]["peaks"]], RAD[K["clip"]["peaks"]], s=14, color=AMB, zorder=4)
ax[1].annotate(f"clip  k* = {ci}", xy=(tc, RAD[ci]), xytext=(tc + 40, RAD[ci] + 6), fontsize=7.5, arrowprops=dict(arrowstyle="-", color="0.3", lw=0.7))
ax[1].set_ylabel("r (px)"); ax[1].spines[["top", "right"]].set_visible(False)
ax[2].plot(SAMP, GPROF, color="0.35", lw=1.0)
ax[2].axhline(150, color=RED, lw=0.8, ls="--"); ax[2].text(T1 - 4, 156, "150", color=RED, fontsize=7, ha="right")
ax[2].set_ylabel("g (gray)"); ax[2].set_xlabel("t (px along the PCA axis)"); ax[2].set_ylim(0, 255)
ax[2].spines[["top", "right"]].set_visible(False)
ax[1].set_ylim(0, RAD.max() * 1.18)
ax[1].text(K["clip"]["mid"], RAD.max() * 1.15, "search window m ± h", color=AMB, fontsize=7, ha="center", va="top")
for t_, s_ in [(T0 + K["margin"] / 2, "margin"), (T1 - K["margin"] / 2, "margin")]:
    ax[2].text(t_, 235, s_, color=RED, fontsize=6.8, ha="center", va="top")
save(fig, "f20_profiles.png")
NUM["profile"] = {"r_max": float(RAD.max()), "r_max_t": float(SAMP[np.argmax(RAD)]), "g_max": float(GPROF.max()), "g_min": float(GPROF.min())}

# F21 side scores --------------------------------------------------------------------------
w = K["wire_ref"]
ev = (GPROF > 150) & (RAD < 1.5 * w)
fig, ax = plt.subplots(2, 1, figsize=(7.2, 3.0), sharex=True, gridspec_kw=dict(height_ratios=[0.6, 1], hspace=0.08))
ax[0].imshow(STRIP, extent=[STS[0], STS[-1], SUS[-1], SUS[0]], aspect="auto")
ax[0].set_yticks([])
sn, sp = K["side"]["neg"], K["side"]["pos"]
for a_ in ax:
    a_.axvspan(SAMP[sn["lo"]], SAMP[sn["hi"]], color=PAL[2], alpha=0.12, lw=0)
    a_.axvspan(SAMP[sp["lo"]], SAMP[sp["hi"]], color=PAL[3], alpha=0.12, lw=0)
    a_.axvline(tc, color=AMB, lw=1.0)
ax[1].plot(SAMP, GPROF, color="0.35", lw=1.0, label="g (gray)")
ax[1].scatter(SAMP[ev], GPROF[ev], s=16, color=RED, zorder=4, label="bright and thin: g > 150 and r < 1.5w")
ax[1].axhline(150, color=RED, lw=0.7, ls="--")
ax[1].text(SAMP[sn["lo"]] + 5, 238, f"side −1:  σ₋ = {sn['n_both']} + 0.05 × {sn['g_max']:.0f} = {sn['score']:.2f}", color=PAL[2], fontsize=7.5, va="top")
ax[1].text(SAMP[sp["hi"]] - 5, 238, f"side +1:  σ₊ = {sp['n_both']} + 0.05 × {sp['g_max']:.0f} = {sp['score']:.2f}", color=PAL[3], fontsize=7.5, va="top", ha="right")
ax[1].set_ylim(0, 255); ax[1].set_ylabel("g"); ax[1].set_xlabel("t (px)"); ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 0.86), framealpha=0.9, ncol=2)
if not ev.any():
    ax[1].text(SAMP[int(np.argmax(GPROF))], GPROF.max() + 8, f"label peak g = {GPROF.max():.0f} < 150:\nno sample counts, only 0.05·max(g) decides", fontsize=7, ha="center", va="bottom", color="0.2")
ax[1].spines[["top", "right"]].set_visible(False)
ax[0].set_title(f"label_side_score: σ₋ ≥ σ₊, so δ = {K['side']['dir_pref']} (toward the sticker)")
save(fig, "f21_side_score.png")

# F22 walk, offset, checks (zoom around the clip) -------------------------------------
tr = K["tries"][-1]
lo_i, hi_i = max(0, tr["pick_i"] - 18), min(len(SAMP) - 1, ci + 22)
fig, ax = plt.subplots(2, 1, figsize=(7.2, 3.6), sharex=True, gridspec_kw=dict(height_ratios=[0.85, 1], hspace=0.08))
ax[0].imshow(STRIP, extent=[STS[0], STS[-1], SUS[-1], SUS[0]], aspect="auto")
ax[0].set_yticks([])
ax[1].plot(SAMP, RAD, color=(0.1, 0.6, 0.6), lw=1.2, marker="o", ms=2)
ax[1].axhline(tr["limit"], color=RED, lw=0.8, ls="--"); ax[1].text(SAMP[hi_i] - 1, tr["limit"] + 0.3, f"λ = max(1.2w, w+2) = {tr['limit']:.2f}", color=RED, fontsize=7, ha="right")
ax[1].axhline(1.35 * w, color=(0.8, 0.5, 0.2), lw=0.7, ls=":"); ax[1].text(SAMP[lo_i] + 1, 1.35 * w + 0.3, f"1.35w = {1.35 * w:.2f}", color=(0.8, 0.5, 0.2), fontsize=7)
ax[1].scatter(SAMP[tr["walk"]], RAD[tr["walk"]], s=22, color=RED, zorder=4)
te, tp0, tp = SAMP[tr["edge_i"]], SAMP[tr["pick_i0"]], SAMP[tr["pick_i"]]
for a_ in ax:
    a_.axvline(tc, color=AMB, lw=1.2); a_.axvline(te, color=RED, lw=1.0); a_.axvline(tp, color=(0.85, 0.75, 0), lw=1.8)
ymax = max(RAD[lo_i:hi_i + 1]) * 1.25
ax[1].set_ylim(0, ymax)
ax[1].annotate("", xy=(tp0, ymax * 0.82), xytext=(te, ymax * 0.82), arrowprops=dict(arrowstyle="-|>", color=AMB, lw=1.4))
ax[1].text((te + tp0) / 2, ymax * 0.85, f"Δ = round({S['pickup_offset_px']} / {K['step']:.3f}) = {K['extra']} samples", ha="center", fontsize=7.5, color=AMB)
ax[1].annotate("", xy=(te, ymax * 0.66), xytext=(tc, ymax * 0.66), arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.2))
ax[1].text((te + tc) / 2, ymax * 0.9, f"walk: {len(tr['walk']) - 1} steps", ha="center", fontsize=7, color=RED)
for t_, s_, c_, yy in [(tc, f"clip k* = {ci}", AMB, SUS[0] + 2), (te, f"exit k_e = {tr['edge_i']}", RED, SUS[0] + 22), (tp, f"pick k_p = {tr['pick_i']}", (0.7, 0.6, 0), SUS[0] + 2)]:
    ax[0].text(t_, yy, s_, color="white", fontsize=7, ha="center", va="top", bbox=dict(boxstyle="round,pad=0.15", fc=(*c_, 0.9), ec="none"))
ax[1].set_xlim(SAMP[lo_i], SAMP[hi_i]); ax[1].set_xlabel("t (px)"); ax[1].set_ylabel("r (px)")
ax[1].spines[["top", "right"]].set_visible(False)
bt = tr["backtrack"]
ax[1].text(tp + 1, ymax * 0.12, f"check at k_p: g = {bt[-1][1]:.1f} < 150,  r = {bt[-1][2]:.1f} ≤ 1.35w  (pass)", fontsize=7, color="0.2")
ax[0].set_title("Walk off the bulge, then move Δ samples further along the axis")
save(fig, "f22_walk_offset.png")

# F23 final pixel ---------------------------------------------------------------------------
pt = K["point"]
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.5))
show(ax[0], np.zeros((H, W)), cmap="gray", vmin=0, vmax=1)
ax[0].imshow(dshow, cmap="magma", interpolation="nearest")
selm = (np.abs(TMAP - K["pick_t"]) <= K["final"]["band"])
tint(ax[0], selm.astype(np.uint8), (0.3, 1.0, 1.0), 0.55)
ax[0].scatter(*pt, s=40, color=YEL, edgecolor="black", lw=0.6, zorder=5)
zb = [pt[0] - 40, pt[1] - 30, pt[0] + 40, pt[1] + 30]
zoom(ax[0], zb)
ax[0].set_title(f"Slice at t_p = {K['pick_t']:.1f}: {K['final']['count']} candidates; argmax D = {K['final']['best_dist']:.1f}", fontsize=8)
show(ax[1], rgb(ANN)); zoom(ax[1], [pt[0] - 70, max(0, pt[1] - 45), pt[0] + 70, pt[1] + 45])
ax[1].set_title(f"draw_pickup_points: ({round(pt[0])}, {round(pt[1])}), r = {S['pickup_marker_radius']}", fontsize=8)
panel(ax[0], "a"); panel(ax[1], "b")
save(fig, "f23_final_point.png")

# F24 both photos ----------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.15))
show(ax[0], rgb(ANN)); ax[0].set_title("image84: 4 YOLO + 1 OpenCV")
show(ax[1], rgb(T85["_ann"])); ax[1].set_title("image85: 5 YOLO, OpenCV adds nothing")
for c in T["cables"]:
    ax[0].text(c["pickup"]["point"][0] + 10, c["pickup"]["point"][1] - 8, f"({c['pickup']['point'][0]:.0f}, {c['pickup']['point'][1]:.0f})", color="white", fontsize=5.6)
for c in T85["cables"]:
    ax[1].text(c["pickup"]["point"][0] + 10, c["pickup"]["point"][1] - 8, f"({c['pickup']['point'][0]:.0f}, {c['pickup']['point'][1]:.0f})", color="white", fontsize=5.6)
panel(ax[0], "a"); panel(ax[1], "b")
fig.tight_layout(w_pad=0.6)
save(fig, "f24_both_results.png")

# summary tables for the report --------------------------------------------------------------
NUM["all_cables"] = {name: [{"source": c["source"], "score": c["score"], "bbox": c["mask"]["bbox"], "point": c["pickup"]["point"], "clip_i": c["pickup"]["clip"]["clip_i"],
                             "wire_ref": c["pickup"]["wire_ref"], "dir": c["pickup"]["side"]["dir_pref"], "pick_i": c["pickup"]["pick_i"]} for c in TT["cables"]]
                     for name, TT in [("image84", T), ("image85", T85)]}
NUM["checks"] = {"image84": T["checks"], "image85": T85["checks"]}
with open(os.path.join(OUT, "numbers.json"), "w") as f:
    json.dump(NUM, f, indent=1, default=float)
print("done")

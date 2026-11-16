# Cable Segmentation and Robot Pickup-Point Selection: A Step-by-Step Account of the `kyunshin_pickup` Pipeline

*Technical report · project `kyunshin_pickup`, branch `pick_point` · data: `Sheet_Data_1_folder/image84.png`, `image85.png`*

---

## Abstract

This report explains, step by step and with the underlying mathematics, how the `kyunshin_pickup` pipeline turns one camera photo of a tray of cable harnesses into one robot pickup point per cable. The pipeline has three stages. **Stage 1** runs a YOLO instance-segmentation model and cleans its masks: it resizes them, joins fragments split by labels, filters by size and suppresses duplicates. **Stage 2** recovers cables that YOLO missed, using classical image processing: grayscale conversion, Gaussian blur, inverse thresholding, contour extraction, geometric bridging of fragments and an overlap test against the YOLO masks. **Stage 3** computes a pickup point on every cable mask: principal component analysis (PCA) gives an axis, a distance transform gives a thickness profile along that axis, the thickest point near the middle is taken as the mounting clip, and the point is placed a fixed distance beyond the clip bulge on the sticker side, at the centre of the wire.

Every figure and number below comes from an **exact trace** of the real code (`trace_export.py`) run on `image84.png` with the settings in `config.yaml`. The trace was checked against the project's own functions: `join_instance_masks`, `suppress_overlapping_masks`, `detect_cv_masks` and `pickup_point` all return identical results, and the final `infer()` image differs in **0 pixels**. The worked example is the top cable of `image84`, which YOLO missed entirely and OpenCV recovered. Its pickup point is **(597, 79)**.

---

## 1. Introduction

A robot that picks cable harnesses from a tray needs, for each cable, one image point where a gripper can close safely: on plain wire, away from the connectors, the mounting clip and the paper label. `MODEL.process()` produces exactly that. Figure 1 shows the input and the output for `image84`.

![Figure 1](figures/f01_overview.png)

**Figure 1.** (a) The input photo, 720 × 1280 pixels, five cables. (b) The output of `infer()`. Blue overlays are YOLO masks (with confidence), green is a mask found by the OpenCV fallback, and the yellow dots are the pickup points.

Figure 2 shows how the stages connect. Stages 1 and 2 decide *where the cables are*. Stage 3 decides *where to grab each one*. The rest of the report follows the order in which `infer()` calls its helpers.

![Figure 2](figures/f02_pipeline.png)

**Figure 2.** Data flow through `MODEL.infer()`. Every mask that survives Stage 1 or Stage 2 goes through Stage 3 independently.

### 1.1 Notation

| Symbol | Meaning |
|---|---|
| $I(x,y)$ | photo, with channels $(B,G,R)$, $x$ = column $\in[0,1279]$, $y$ = row $\in[0,719]$ |
| $M(x,y)\in\{0,1\}$ | a binary mask: 1 = cable pixel |
| $\vert M\vert$ | area of a mask = number of 1-pixels |
| $\mathbf{p}=(x,y)$ | a pixel position, used as a 2-D vector |
| $\mathbf{u}\cdot\mathbf{v}$, $\Vert\mathbf{u}\Vert$ | dot product and Euclidean length |
| $[\,\cdot\,]$ | Iverson bracket: 1 if the condition is true, else 0 |

Settings from `config.yaml` are written in `code font` and listed in Appendix A.

---

## 2. Stage 1: YOLO segmentation and mask cleanup

### 2.1 Instance segmentation and confidence (`model.predict`)

A YOLO segmentation model predicts, for every object it detects, a bounding box, a class, a **confidence score** $s\in[0,1]$ and a **mask**. Only detections with

$$ s \geq \tau_{\text{conf}} = 0.25 \tag{1} $$

are returned (`model_confidence`). On `image84` the model returned four masks, with scores 0.973, 0.971, 0.969 and 0.714 (Figure 3). **None of them covers the top cable.** The rest of Stage 1 cannot fix a cable the model never saw. That is the job of Stage 2.

![Figure 3](figures/f03_yolo_raw.png)

**Figure 3.** The four raw YOLO masks and their scores. The dashed green box marks the top cable, which has no YOLO mask at all.

### 2.2 From model grid to photo grid (`_resize_mask`)

The masks come back as `result.masks.data` with shape $(4, 544, 960)$: one probability map per detection on the model's input grid, not on the photo grid. `_resize_mask` binarises each map, then stretches it to the photo size with **nearest-neighbour** interpolation:

$$ M_0(x,y) = \left[\,P(x,y) > 0.5\,\right] \tag{2} $$

$$ M(x',y') = M_0\left(\left\lfloor x'\cdot\frac{960}{1280}\right\rfloor,\ \left\lfloor y'\cdot\frac{544}{720}\right\rfloor\right) \tag{3} $$

Nearest neighbour *copies* the closest source pixel instead of averaging neighbours, so the result stays strictly 0/1 (Figure 4c). Averaging would create grey edge values that are neither cable nor background.

![Figure 4](figures/f04_resize.png)

**Figure 4.** (a) One raw mask on the 544 × 960 model grid. (b) The same mask after `_resize_mask`, on the 720 × 1280 photo. (c) Nearest-neighbour up-scaling by 2: every source pixel becomes a 2 × 2 block, and no new values appear.

**A small geometric detail.** At width 960, a 16:9 photo only needs $960\cdot 720/1280 = 540$ rows. The model grid has 544, so 2 rows of letterbox padding sit above and below the photo. Equation (3) maps the full 544 rows onto 720, which stretches that padding into the image. Photo row $y$ is read from model row $0.7556\,y$, which really belongs to photo row $(0.7556\,y-2)/0.75 = 1.0074\,y - 2.67$. The mask is therefore displaced by

$$ e(y) = y - (1.0074\,y - 2.67) \approx 2.67 - 0.0074\,y\ \ \mathrm{px}, \tag{4} $$

that is 2.7 px downward at the top of the photo, 0 near row 360 and 2.7 px upward at the bottom: the masks are squeezed slightly toward the middle. The trace compared every resized mask with Ultralytics' own correctly-scaled outlines (`masks.xy`) and measured mean shifts between −1.1 px (bottom cable) and +2.0 px (upper cable), with the signs Equation (4) predicts. This is too small to matter here, but it explains why the overlays can sit a pixel or two off at the image edges.

### 2.3 Pieces, directions and ends (`_piece_ends`)

Each resized mask with $\vert M\vert \geq$ `cv_min_piece_area` = 300 px is reduced to its largest external contour, called a **piece**. Three measurements are taken from it.

**Direction.** `cv2.fitLine(…, DIST_L2, …)` finds the straight line that minimises the sum of squared perpendicular distances from the contour points $\{\mathbf{q}_i\}$:

$$ (\mathbf{q}_0,\mathbf{d}) = \arg\min_{\mathbf{q}_0,\ \Vert\mathbf{d}\Vert=1}\ \sum_i \left\Vert(\mathbf{q}_i-\mathbf{q}_0) - \left((\mathbf{q}_i-\mathbf{q}_0)\cdot\mathbf{d}\right)\mathbf{d}\right\Vert^2 \tag{5} $$

The solution passes through the mean point $\mathbf{q}_0$, and $\mathbf{d}$ is the eigenvector of the points' scatter matrix with the largest eigenvalue. This is the same idea as the PCA used in Stage 3.

**Ends.** Each contour point is projected onto the line, $t_i=(\mathbf{q}_i-\mathbf{q}_0)\cdot\mathbf{d}$. The two ends are the averages of the points within 5 px of the extremes:

$$ \mathbf{e}_{\min} = \operatorname{mean}\{\mathbf{q}_i : t_i < t_{\min}+5\},\qquad \mathbf{e}_{\max} = \operatorname{mean}\{\mathbf{q}_i : t_i > t_{\max}-5\} \tag{6} $$

**Length and centre.** `_contour_length` uses the rotated minimum-area rectangle, $L=\max(w_{\text{rect}},h_{\text{rect}})$. `_contour_centroid` uses the image moments:

$$ m_{pq}=\sum_{(x,y)\in\text{shape}} x^p\,y^q, \qquad \mathbf{c} = \left(\frac{m_{10}}{m_{00}},\ \frac{m_{01}}{m_{00}}\right). \tag{7} $$

![Figure 5](figures/f05_piece_geometry.png)

**Figure 5.** Geometry of one YOLO piece (the cable second from the top). White: contour. Cyan dashes: minimum-area rectangle, 825.8 × 79.7 px. Amber dashes: fitted line, $\mathbf{d}=(0.9921,-0.1257)$. Red: the 37 and 18 contour points within 5 px of each extreme. Amber dots: their averages, the ends. White cross: centroid. (b) Zoom on the left end.

### 2.4 Should two pieces be joined? (`_ends_should_join`)

A white label can break one cable into two fragments. `join_instance_masks` checks every pair of pieces $(i,j)$. First, pieces that are both long are **never** joined, because they are two whole cables:

$$ L_i \geq \lambda_{\mathrm{long}}\ \wedge\ L_j \geq \lambda_{\mathrm{long}}\ \Rightarrow\ \mathrm{skip},\qquad \lambda_{\mathrm{long}} = \max(0.75\,L_{\min},\ 400) = 400\ \mathrm{px}, \tag{8} $$

where $L_{\min}$ = `mask_width_threshold_px` = 500.

Otherwise all four end combinations $(\mathbf{a},\mathbf{b})$, with $\mathbf{a}$ an end of piece $i$ and $\mathbf{b}$ an end of piece $j$, are tried in order. The first combination that passes every test below is accepted (Figure 6).

**Gap.** With $\mathbf{g}=\mathbf{b}-\mathbf{a}$:

$$ 10^{-6} \leq \Vert\mathbf{g}\Vert \leq G = 120\ \mathrm{px} \tag{9} $$

(`cv_max_gap_px`).

**Angle.** The gap must run along both pieces. `_angle_deg` uses the *absolute* cosine, so the arrow's sign does not matter:

$$ \theta(\mathbf{u},\mathbf{v}) = \arccos\frac{\vert\mathbf{u}\cdot\mathbf{v}\vert}{\Vert\mathbf{u}\Vert\,\Vert\mathbf{v}\Vert}, \qquad \theta(\mathbf{g},\mathbf{d}_i)\leq 25^\circ,\quad \theta(\mathbf{g},\mathbf{d}_j)\leq 25^\circ. \tag{10} $$

**Lateral offset.** Build a shared axis and its normal:

$$ \mathbf{a}_{ij} = \frac{\mathbf{d}_i + \mathrm{sgn}(\mathbf{d}_i\cdot\mathbf{d}_j)\,\mathbf{d}_j}{\Vert\mathbf{d}_i + \mathrm{sgn}(\mathbf{d}_i\cdot\mathbf{d}_j)\,\mathbf{d}_j\Vert}, \qquad \mathbf{n}=(-a_{ij,y},\ a_{ij,x}). \tag{11} $$

The sideways part of the gap must be small:

$$ \ell = \vert\mathbf{g}\cdot\mathbf{n}\vert \leq 40\ \mathrm{px}. \tag{12} $$

**Common centreline.** If an end pair passes, the caller makes one more check, on the centroids:

$$ \vert(\mathbf{c}_j-\mathbf{c}_i)\cdot\mathbf{n}\vert \leq \max(1.5\cdot 40,\ 50) = 60\ \mathrm{px}. \tag{13} $$

![Figure 6](figures/f06_join_schematic.png)

**Figure 6.** The quantities in Equations (9)–(13). Two pieces of one cable face each other across a short gap $\mathbf{g}$ that points along both directions $\mathbf{d}_i$ and $\mathbf{d}_j$. Two neighbouring cables can have close ends too, but their gap points sideways.

**Result on `image84`.** All four YOLO masks are about 810–830 px long, so all $4\cdot 3/2=6$ pairs are skipped by Equation (8) and no YOLO masks are joined. Section 3.4 shows the same tests at work on the OpenCV pieces, where they matter.

### 2.5 Grouping (union-find), bridges and the size filter

Accepted pairs are edges of a graph whose nodes are pieces. **Union-find** merges pieces into groups: if $A$ joins $B$ and $B$ joins $C$, then $\{A,B,C\}$ is one group, even if $A$ and $C$ were never tested against each other. The groups are the graph's *connected components*. For each group, the member masks are OR-ed onto one canvas, and `_bridge_pieces_on_mask` draws a filled line of thickness `cv_join_width_px` = 15 px between each accepted end pair (Section 3.4). The largest contour of the canvas is then kept only if

$$ A \geq A_{\min} = 8000\ \mathrm{px}^2 \quad\wedge\quad L \geq L_{\min} = 500\ \mathrm{px} \tag{14} $$

(`cv_min_cable_area`, `mask_width_threshold_px`).

On `image84` each group has one piece, and all four pass, with areas of 22 862–25 635 px².

### 2.6 Duplicate suppression (`suppress_overlapping_masks`)

YOLO sometimes outputs two masks for one object. Masks are visited in order of decreasing score (ties: larger area first). Each one is compared with every mask already kept, using the overlap ratio

$$ r(M_a,M_b) = \frac{\vert M_a\cap M_b\vert}{\min(\vert M_a\vert,\ \vert M_b\vert)}, \tag{15} $$

and dropped if $r \geq$ `mask_overlap_thresh` = 0.1 against any kept mask. Dividing by the *smaller* area means a small mask lying inside a big one gives $r=1$ and is removed, whereas plain intersection-over-union would barely notice it. On `image84` every intersection is 0 px, so all four masks are kept (Figure 12b shows the idea).

---

## 3. Stage 2: OpenCV recovery of missed cables

`detect_cv_masks` looks for long dark objects without any learning, and keeps only those that no YOLO mask touches.

### 3.1 Grayscale

OpenCV converts BGR to luma with the ITU-R BT.601 weights:

$$ Y = 0.299\,R + 0.587\,G + 0.114\,B. \tag{16} $$

The black sleeve on the top cable at (597, 79) has $(B,G,R)=(21,21,21)$, so $Y=21$. The white label at (470, 78) has $(181,169,164)$, so $Y = 0.299\cdot164 + 0.587\cdot169 + 0.114\cdot181 \approx 169$.

### 3.2 Gaussian blur

The image is convolved with a normalised Gaussian kernel:

$$ B(x,y) = \sum_{u=-2}^{2}\sum_{v=-2}^{2} k(u)\,k(v)\;Y(x+u,\ y+v). \tag{17} $$

For `cv_gaussian_ksize` = 5 and $\sigma=0$, OpenCV's rule $\sigma = 0.3\,((k-1)/2-1)+0.8$ gives $\sigma = 1.1$. For kernel sizes up to 7 it uses a fixed table matching that width, the binomial kernel

$$ k = \frac{1}{16}\,(1,\ 4,\ 6,\ 4,\ 1) = (0.0625,\ 0.25,\ 0.375,\ 0.25,\ 0.0625), \tag{18} $$

so the centre pixel weighs $0.375^2 = 0.141$ and each corner $0.0625^2 = 0.0039$. The blur removes sensor noise and the woven texture of the sleeve (Figure 7b), so the threshold that follows gives clean edges.

### 3.3 Inverse binary threshold

With $\tau$ = `cv_binary_threshold` = 70:

$$ T(x,y) = 255\cdot\left[\,B(x,y)\leq\tau\,\right] \tag{19} $$

*Inverse* means dark becomes white, so the cables become foreground. The grey-level histogram (Figure 8) shows two populations: the dark cables and the bright tray. The threshold sits between them, and 12.4 % of the photo (114 020 px) becomes white. The labels and the bright blue tape are above 70, so **every cable breaks at its label** (Figure 7c).

![Figure 7](figures/f10_preprocess.png)

**Figure 7.** (a) Grayscale, (b) 5 × 5 Gaussian blur, (c) inverse threshold at 70. The bottom row zooms on the label of the top cable (amber box). The label is brighter than 70, so the threshold cuts the cable into two pieces there.

![Figure 8](figures/f11_histogram.png)

**Figure 8.** Histogram of the blurred image (log scale). The pixels at or left of $\tau=70$ (dark bars) become the white foreground.

### 3.4 Contours, joining and bridges

External contours of $T$ are extracted: 23 on `image84`. The 13 with area below 300 px² are noise (holes in the tray, specks) and are dropped. The other **10 pieces** are two per cable: a short ~220 px piece between the left connector and the label, and a long ~530 px piece from the label to the right connector (Figure 9).

![Figure 9](figures/f12_pieces.png)

**Figure 9.** The ten pieces kept after the area filter (coloured, numbered) and the dropped small contours (red dashes). Pieces 8 and 9 are the top cable.

The same tests as in Section 2.4 then run on all $10\cdot 9/2=45$ pairs, with one difference: here $\lambda_{\text{long}} = \max(0.75\cdot 500, 400) = 400$ px separates the short pieces (~220 px) from the long ones (~530 px). The outcome was:

* **10 pairs skipped** by Equation (8): both pieces long.
* **30 pairs rejected** by Equations (9)–(12).
* **5 pairs accepted** and bridged: exactly one per cable.

**The accepted bridge of the top cable (pieces 8 and 9).** Figure 10a shows the end pairs in the order the loop tries them:

* $\mathbf{e}^{(8)}_{\min}\to\mathbf{e}^{(9)}_{\min}$: $\Vert\mathbf{g}\Vert = 286.2 > 120$, rejected by Equation (9).
* $\mathbf{e}^{(8)}_{\min}\to\mathbf{e}^{(9)}_{\max}$: $813.3 > 120$, rejected.
* $\mathbf{e}^{(8)}_{\max}=(458.5, 96.2)\to\mathbf{e}^{(9)}_{\min}=(526.0, 84.5)$: $\Vert\mathbf{g}\Vert = 68.5 \leq 120$, angles $23.4^\circ$ and $2.9^\circ \leq 25^\circ$, $\ell = 15.6 \leq 40$. **Accepted.**
* The centroid test, Equation (13), gives $28.9 \leq 60$. The pair is bridged with a 15-px line (Figure 10b).

The first angle, 23.4°, is close to the 25° limit. The short piece 8 includes part of the left connector, which tilts its fitted direction. A slightly stricter `cv_max_join_angle_deg` would have left this cable split in two.

![Figure 10](figures/f07_join_real.png)

**Figure 10.** (a) The end pairs checked for pieces 8 and 9. Red dashes failed the gap test; the green pair passed all tests. (b) After `cv2.line(…, thickness=15)`, the two pieces are one blob.

**A rejected pair.** Pieces 3 and 4 belong to two different cables. Their nearest ends are only 90.3 px apart, which passes Equation (9). But the gap is $73^\circ$ off piece 3's direction and $71^\circ$ off piece 4's, so Equation (10) rejects it (Figure 11). This is exactly the neighbouring-cable case the angle test exists for. Pieces 1 and 2 show the plainer failure: their closest ends are 134.0 px apart, which is more than 120.

![Figure 11](figures/f08_reject_angle.png)

**Figure 11.** Pieces 3 and 4: the ends are close, but every short gap runs *across* the cables, at about 70–73°, far above the 25° limit.

![Figure 12](figures/f09_groups_overlap.png)

**Figure 12.** (a) The decisions as a graph: pieces are nodes, accepted joins (green) are edges, and each connected component is one cable. (b) The overlap ratio of Equation (15), used for duplicate suppression in Stage 1.

### 3.5 Size filter and the YOLO-overlap test

Contours are extracted again on the bridged mask. All five blobs pass Equation (14), with lengths of 806–829 px and areas of 21 490–25 278 px² (Figure 13). They are the **candidates** $C_k$. All YOLO masks are OR-ed into one union mask (`_yolo_union_mask`):

$$ U(x,y) = \max_{m}\ M^{\mathrm{YOLO}}_m(x,y), \qquad \vert U\vert = 101\,214\ \mathrm{px}. \tag{20} $$

A candidate is kept only if it touches **no** YOLO pixel at all:

$$ \mathrm{keep}\ C_k \ \Leftrightarrow\ \vert C_k \cap U\vert = 0. \tag{21} $$

The rule uses `np.logical_and(mask, yolo_union).any()`, so a single shared pixel is enough to reject a candidate. On `image84` the four lower candidates share 21 768–23 937 px with $U$ and are rejected. The top cable shares **0 px** and is kept (Figure 14). **This is how the top cable was recovered.** On `image85`, YOLO found all five cables, so all five OpenCV candidates are rejected and Stage 2 adds nothing.

![Figure 13](figures/f13_bridged.png)

**Figure 13.** Five bridges (amber) turn the ten pieces into five blobs. Each passes the size filter of Equation (14).

![Figure 14](figures/f14_union_test.png)

**Figure 14.** The YOLO union $U$ (blue) and the OpenCV candidates. Red: candidates that overlap $U$, rejected. Green: the one candidate with no overlap, kept.

### 3.6 Drawing the overlays (alpha blending)

`overlay_masks` paints every mask pixel with a solid colour into a copy $O$ of the photo, then blends:

$$ \mathrm{out} = \mathrm{round}\left(\alpha\,O + (1-\alpha)\,I\right),\qquad \alpha = 0.45. \tag{22} $$

Outside the masks $O=I$, so the photo is unchanged there. For example, the YOLO pixel at (243, 237) has $I=(29,34,31)$ and $O=(255,0,0)$, which gives $(131,19,17)$ (Figure 15). YOLO masks are blue, with their score printed at the mask's mean pixel; OpenCV masks are green, without a score.

![Figure 15](figures/f15_blend.png)

**Figure 15.** Alpha blending, Equation (22), at one pixel (yellow circle).

---

## 4. Stage 3: The pickup point

`pickup_points_for_masks` calls `pickup_point` on every mask, in the order YOLO masks first, then OpenCV masks, with `fraction` = 0.5 and `offset_px` = 45. Two more parameters keep their defaults: `band_px` = 4 and `clip_window` = 0.25. The worked example is the top (OpenCV) cable of `image84`.

### 4.1 The cable's main axis: PCA (`_mask_pca_axis`)

Take every mask pixel as a point $\mathbf{p}_i=(x_i,y_i)$, $i=1\ldots N$; here $N = 26\,246$. Their mean and covariance are

$$ \bar{\mathbf{p}} = \frac{1}{N}\sum_i \mathbf{p}_i = (646.46,\ 66.28), \qquad \Sigma = \frac{1}{N}\sum_i (\mathbf{p}_i-\bar{\mathbf{p}})(\mathbf{p}_i-\bar{\mathbf{p}})^{T}, \tag{23} $$

which here has entries $\Sigma_{xx} = 66\,045.5$, $\Sigma_{xy} = -2\,178.5$ and $\Sigma_{yy} = 597.2$ (px²).

PCA solves the eigenproblem

$$ \Sigma\,\mathbf{v}_k = \lambda_k\,\mathbf{v}_k,\qquad \lambda_1 = 66\,118,\quad \lambda_2 = 524.8. \tag{24} $$

$\lambda_k$ is the variance of the points along $\mathbf{v}_k$. The first eigenvector is the direction of greatest spread, which for a long object is its length direction. Here $\mathbf{v}_1 = (0.99945, -0.03323)$, an axis tilted by $-1.9^\circ$. The two standard deviations are $\sqrt{\lambda_1}=257.1$ px along the cable and $\sqrt{\lambda_2}=22.9$ px across it, a ratio of 11 to 1 (Figure 16). An eigenvector is only defined up to its sign, so the code fixes the sign to make the axis point right:

$$ \mathbf{a} = -\mathbf{v}_1\ \ \mathrm{if}\ \ v_{1,x}<0\ \ \mathrm{or}\ \ (\vert v_{1,x}\vert<10^{-6}\ \wedge\ v_{1,y}<0),\qquad \mathrm{otherwise}\ \ \mathbf{a}=\mathbf{v}_1. \tag{25} $$

This makes "direction −1" always mean "toward the left end". No flip was needed here.

*Sanity check.* A uniform rod of length $L$ has variance $L^2/12$ along its axis, which would give $L=\sqrt{12\lambda_1}\approx 891$ px. The measured extent below is 818 px. The rod model overestimates because the heavy connectors at both ends put extra pixels far from the centre.

![Figure 16](figures/f16_pca.png)

**Figure 16.** The mask pixels of the top cable as a point cloud (every 9th point shown). The ellipses are the 1σ and 2σ contours of $\Sigma$, and the arrows are the eigenvectors.

### 4.2 Projection: one coordinate along the cable

Every pixel gets a position along the axis:

$$ t_i = (\mathbf{p}_i-\bar{\mathbf{p}})\cdot\mathbf{a},\qquad t_0 = \min_i t_i = -406.29,\qquad t_1=\max_i t_i = 411.28. \tag{26} $$

From here on, any position along the cable is a single number $t$, and the span is $t_1-t_0 = 817.58$ px (Figure 17). The inverse map, $\mathbf{p}(t,u)=\bar{\mathbf{p}}+t\,\mathbf{a}+u\,\mathbf{n}$ with $\mathbf{n}=(-a_y,a_x)$, is used to draw the "straightened" cable in Figures 20–22.

![Figure 17](figures/f17_projection.png)

**Figure 17.** Each mask pixel coloured by its projection $t$.

### 4.3 Thickness: the distance transform

For every mask pixel, the distance transform gives the distance to the nearest background pixel:

$$ D(\mathbf{p}) = \min_{\mathbf{q}:\ M(\mathbf{q})=0}\ \Vert\mathbf{p}-\mathbf{q}\Vert. \tag{27} $$

OpenCV's `DIST_L2` with `maskSize = 5` computes a close approximation: a two-pass 5 × 5 chamfer scheme with step costs 1 (horizontal or vertical), 1.4 (diagonal) and 2.1969 (knight's move). On a straight band of width $w$, $D$ rises linearly from each edge and peaks in the middle at $w/2$. **The maximum of $D$ across the wire is the wire's radius.** On the top cable $D$ peaks at 31.2 px inside the connectors. At the eventual pickup position the cross-section is a tent of height 11.0 px, so the wire is about 22 px wide (Figure 18b).

![Figure 18](figures/f18_distance.png)

**Figure 18.** (a) The distance transform of the top cable. (b) A section across the wire at $t=-46.5$ (the cyan line in (a)). Its height is the radius. Note that the tent's peak lies about 11 px from $u=0$: the PCA axis is a straight line through the whole cable, not through the wire centre at this point. Section 4.9 shows how the code corrects for this.

### 4.4 Sampling the axis: radius and gray profiles

`_radius_profile` places evenly spaced sample positions along the axis:

$$ n = \max\left(8,\ \left\lceil\frac{t_1-t_0}{2}\right\rceil + 1\right) = 410, \qquad s_k = t_0 + k\,\frac{t_1-t_0}{n-1},\ \ k=0,\ldots,n-1, \tag{28} $$

so the spacing is $\bar{s} = 817.58/409 = 1.999$ px. Each sample owns a thin **slice** of the mask, perpendicular to the axis:

$$ S_k = \{\,i :\ \vert t_i - s_k\vert \leq b\,\},\qquad b = 4\ \mathrm{px} \tag{29} $$

(`band_px`). The **radius profile** is the largest distance value in each slice, and the **gray profile** (`_gray_along`) is the mean brightness of the photo, Equation (16), over the same slice:

$$ r_k = \max_{i\in S_k} D(\mathbf{p}_i), \qquad g_k = \frac{1}{\vert S_k\vert}\sum_{i\in S_k} Y(\mathbf{p}_i). \tag{30} $$

Figure 19 shows three of the 410 slices. Figure 20 shows both profiles over the whole cable, under a "straightened" image of the cable on the same $t$ axis. Reading Figure 20 from left to right:

* the left connector, with $r$ up to 29 px;
* plain wire, $r\approx 12$;
* a stretch where $r\approx 8.6$: this is the label region, where the mask is the **15-px bridge**, not the real cable, so $r\approx 15/2$;
* the clip, $r\approx 20.6$, near $t=12$;
* plain wire again;
* the blue tape, where the mask comes from OpenCV and is slightly thinner and brighter;
* the right connector, $r$ up to 31 px.

The gray profile is low (20–40) on the wire. It rises to 120–148 on the label, and 40–60 on the blue tape and the right connector.

![Figure 19](figures/f19_slices.png)

**Figure 19.** Three slices $S_k$ (cyan) on the distance transform. The yellow dot in each is the pixel with the largest $D$, whose value is $r_k$.

![Figure 20](figures/f20_profiles.png)

**Figure 20.** Top: the photo resampled along $\mathbf{p}(t,u)$, a "straightened" cable. Middle: the radius profile $r_k$, with $w$, the peak threshold $\tau$, the clip search window (amber band) and the end margins (red bands). Bottom: the gray profile $g_k$.

### 4.5 The typical wire radius

Most of the cable is plain wire, so a low percentile of the radius profile measures the wire and ignores the connectors and the clip:

$$ w = P_{40}\left(\{\,r_k :\ r_k > 0\,\}\right) = 11.98\ \mathrm{px}. \tag{31} $$

$P_{40}$ is the value that 40 % of the samples fall below. All 410 samples are positive here.

### 4.6 Finding the mounting clip (`_find_clip_index`)

The clip is searched for only near mid-length, so that the much thicker connectors at both ends never win:

$$ m = t_0 + f\,(t_1-t_0) = 2.50,\qquad h = c\,(t_1-t_0) = 204.39,\qquad \mathcal{W}=\{\,k :\ \vert s_k-m\vert\leq h\,\} = \{103,\ldots,306\}, \tag{32} $$

with $f$ = `fraction` = 0.5 and $c$ = `clip_window` = 0.25. A sample $k\in\mathcal{W}$ is a **local peak** if

$$ r_k \geq r_{k-1},\qquad r_k\geq r_{k+1},\qquad r_k \geq \tau = \max(1.35\,w,\ w+3) = 16.18. \tag{33} $$

The $\geq$ signs mean a flat top counts several times: samples 209–212 all qualify. The clip index is the peak closest to $m$, with the taller peak winning a tie:

$$ k^\ast = \arg\min_{k\in\mathcal{P}}\ \left(\vert s_k - m\vert,\ -r_k\right) = 209, \qquad s_{209} = 11.5,\quad r_{209} = 20.6, \tag{34} $$

where $\mathcal{P}$ is the set of peaks and the pair is compared lexicographically (distance to $m$ first).

If there were no peak, the thickest sample in $\mathcal{W}$ would be used instead.

### 4.7 Which side of the clip? (`label_side_score`, nested in `pickup_point`)

For each direction $\delta\in\{-1,+1\}$ the code scores the samples between the clip and that end:

$$ \sigma_\delta = \sum_{k\in K_\delta}\left[\,g_k > 150\,\right]\left[\,r_k < 1.5\,w\,\right] \;+\; 0.05\,\max_{k\in K_\delta} g_k, \tag{35} $$

with $K_{-1} = \{0,\ldots,k^\ast\}$ and $K_{+1}=\{k^\ast,\ldots,n-1\}$. The first term counts samples that are **bright and wire-thin**, which is what a label looks like. The second term is a small bonus for the brightest value. The preferred direction is

$$ \delta^\ast = -1\ \ \mathrm{if}\ \ \sigma_{-1}\geq\sigma_{+1},\qquad \mathrm{otherwise}\ \ \delta^\ast = +1. \tag{36} $$

On the top cable, **no sample has $g_k > 150$**. The label averages only 120–148 in its slices, so both counting terms are 0 and only the tie-breaker decides:

$$ \sigma_{-1} = 0 + 0.05\cdot147.9 = 7.40,\qquad \sigma_{+1} = 0 + 0.05\cdot 58.3 = 2.92 \ \ \Rightarrow\ \ \delta^\ast=-1\ \text{(left, toward the label)}. $$

The answer is still the label side, because the label is the brightest thing on the cable. But this shows that the bright-and-thin count is not doing the work on these photos. Across all ten cables in `image84` and `image85`, only one sample set (the second cable in `image84`, with 14 samples) ever exceeds 150. The other nine decisions come from the $0.05\max g$ term alone (Figure 21).

The docstring of `pickup_point` says the code prefers "the side that looks like dark wire". The code actually prefers the **label** side here (as its inline comment says), and keeps the gripper off the label later, in Section 4.9. This report follows the code.

![Figure 21](figures/f21_side_score.png)

**Figure 21.** The side scores of Equation (35). Blue band: the samples counted for $\delta=-1$. Purple band: those for $\delta=+1$. The label peaks at $g=148$, just below the 150 limit, so no sample is counted and $0.05\max g$ decides.

### 4.8 Leaving the bulge and moving the offset (`build_pick`, `_walk_off_bulge`)

Starting at the clip, the code steps one sample at a time in direction $\delta^\ast$ until the radius is back to wire size:

$$ \lambda = \max(1.2\,w,\ w+2) = 14.38,\qquad j^\ast = \min\{\,j\geq 1 :\ r_{k^\ast+\delta^\ast j}\leq\lambda\,\},\qquad k_e = k^\ast + \delta^\ast j^\ast. \tag{37} $$

Here $j^\ast = 6$, so the bulge exit is $k_e = 203$. The requested offset in pixels is then converted to samples:

$$ \Delta = \max\left(1,\ \mathrm{round}\frac{o}{\bar{s}}\right) = \mathrm{round}\frac{45}{1.999} = 23, \qquad k_p = \mathrm{clip}(k_e + \delta^\ast\Delta,\ 0,\ n-1) = 180, \tag{38} $$

with $o$ = `offset_px` = 45.

The offset is measured **along the PCA axis, starting at the bulge exit**, not from the clip centre and not along the curve of the wire. Where the wire makes an angle $\varphi$ with the axis, the distance travelled along the wire is about $45/\cos\varphi$. Here the cable is within a few degrees of the axis, so the difference is under 1 %.

The position is then kept away from both ends by a margin of 12 % of the span:

$$ t_p = \mathrm{clip}\left(s_{k_p},\ t_0 + 0.12\,(t_1-t_0),\ t_1 - 0.12\,(t_1-t_0)\right) = \mathrm{clip}(-46.48,\ -308.19,\ 313.18) = -46.48, \tag{39} $$

and $k_p = \arg\min_k\vert s_k - t_p\vert$, which is still 180 here (Figure 22).

![Figure 22](figures/f22_walk_offset.png)

**Figure 22.** Zoom around the clip. The red dots are the walk of Equation (37), from the clip (amber) to the bulge exit (red line). The amber arrow is the 23-sample offset of Equation (38), which ends at the pick (yellow line).

### 4.9 Safety checks and the final pixel

**Backtracking.** If the chosen sample is bright or thick, the code steps back toward the clip, at most $\Delta+8$ times, until

$$ g_{k_p} < 150 \quad\wedge\quad r_{k_p}\leq 1.35\,w. \tag{40} $$

At $k_p=180$: $g=22.8<150$ and $r=11.0\leq16.18$, so the first check passes and nothing moves.

**Bad-side switch.** If the result still satisfies $g_{k_p}\geq150$ or $r_{k_p}>1.5\,w$, the whole of Section 4.8 is repeated with $-\delta^\ast$, and the second result is used *without comparing it to the first*. Here $22.8 < 150$ and $11.0\leq17.98$, so there is no switch. In fact no cable in either photo needed backtracking or a switch.

**From $t_p$ back to a pixel (`_point_at_projection`).** The pixels of the slice at $t_p$ are the candidates, and the one deepest inside the mask wins:

$$ \mathbf{p}^\ast = \arg\max_{i:\ \vert t_i - t_p\vert\leq b}\ D(\mathbf{p}_i) = (597,\ 79),\qquad D(\mathbf{p}^\ast) = 11.0. \tag{41} $$

There are 176 candidates. Taking the maximum of $D$ moves the point from the straight PCA axis onto the wire's centreline: Figure 18b shows that these two are about 11 px apart at this slice. Several pixels share the maximum value 11.0 along the slice's 8-px width. `np.argmax` returns the first one in row-major order (smallest row, then smallest column), so the dot sits at one edge of that tie (Figure 23a). Finally `draw_pickup_points` rounds the point to integers and draws a filled yellow disc of radius `pickup_marker_radius` = 6 with a 1-px black ring of radius 8 (Figure 23b).

![Figure 23](figures/f23_final_point.png)

**Figure 23.** (a) The candidate slice at $t_p = -46.5$ (cyan) on the distance transform, and the selected pixel. (b) The marker drawn by `draw_pickup_points` on the output image.

### 4.10 The whole decision for the top cable, in one paragraph

PCA lays a straight axis along the cable, from $t_0=-406$ to $t_1=411$. Plain wire has radius $w\approx12$ px. The thickest local peak within ±204 px of the middle is the clip, at $t=11.5$ ($k^\ast=209$). The left side scores higher (7.40 against 2.92) only because the label is the brightest thing on it, so the code heads left. The clip bulge ends 6 samples later ($k_e=203$). 23 more samples, which is 45 px along the axis, reach $t=-46.5$ ($k_p=180$). That position is inside the end margins, dark ($g=22.8$) and wire-thin ($r=11$), so it is accepted. The deepest mask pixel in that slice, (597, 79), is the pickup point.

---

## 5. Results and verification

Figure 24 shows both photos. Table 1 lists every cable.

![Figure 24](figures/f24_both_results.png)

**Figure 24.** (a) `image84`: four YOLO cables and one OpenCV cable. (b) `image85`: five YOLO cables; Stage 2 adds nothing. Pickup coordinates are printed next to each marker.

**Table 1.** Every cable, top to bottom. $k^\ast$ is the clip sample, $j^\ast$ the number of walk steps, $\sigma_{\pm}$ the side scores and $k_p$ the pick sample. $\Delta=23$ for every cable.

| Photo | Source | Score | $w$ (px) | $k^\ast$ | $\sigma_{-1}$ / $\sigma_{+1}$ | $j^\ast$ | $k_p$ | Pickup point |
|---|---|---|---|---|---|---|---|---|
| image84 | OpenCV | – | 11.98 | 209 | 7.40 / 2.92 | 6 | 180 | (597, 79) |
| image84 | YOLO | 0.973 | 11.79 | 212 | 22.48 / 4.57 | 12 | 177 | (595, 243) |
| image84 | YOLO | 0.971 | 12.00 | 215 | 7.29 / 3.65 | 5 | 187 | (617, 394) |
| image84 | YOLO | 0.969 | 12.59 | 216 | 7.04 / 3.94 | 11 | 182 | (558, 511) |
| image84 | YOLO | 0.714 | 12.99 | 205 | 6.90 / 3.68 | 8 | 174 | (608, 653) |
| image85 | YOLO | 0.978 | 13.00 | 212 | 7.00 / 3.40 | 5 | 184 | (611, 79) |
| image85 | YOLO | 0.970 | 11.98 | 213 | 7.18 / 4.33 | 5 | 185 | (596, 197) |
| image85 | YOLO | 0.974 | 12.00 | 215 | 7.23 / 3.79 | 7 | 185 | (606, 381) |
| image85 | YOLO | 0.976 | 12.39 | 215 | 6.96 / 4.35 | 11 | 181 | (557, 509) |
| image85 | YOLO | 0.967 | 13.00 | 203 | 6.92 / 3.59 | 5 | 175 | (608, 652) |

**Table 2.** Verification of the trace against the project's own functions.

| Check | image84 | image85 |
|---|---|---|
| `join_instance_masks` output identical | yes | yes |
| `suppress_overlapping_masks` output identical | yes | yes |
| `detect_cv_masks` output identical | yes | yes |
| `pickup_point` identical for every cable | yes | yes |
| Pixels that differ from `MODEL.infer()` | **0** | **0** |

---

## 6. Discussion

**What worked as designed.** The bridging logic did exactly its job on the OpenCV path: one bridge per cable across the label gap, and no bridge between neighbouring cables (30 rejections). The any-pixel overlap rule cleanly separated the one missing cable from the four YOLO had already found. In Stage 3, the windowed clip search, the walk off the bulge and the distance-transform maximum produced points on plain wire for all ten cables, without needing backtracking or a side switch.

**Observations worth knowing.**

1. **The sticker-side choice relies on the tie-breaker** (Section 4.7). In 9 of 10 cables no slice is brighter than 150, so $\delta^\ast$ comes from $0.05\max g$ alone. This is fine as long as the label is the brightest feature on one side of the clip. A bright reflection or the blue tape on the other side could flip it.
2. **Join angle margin.** The top cable was joined with a 23.4° angle against a 25° limit (Section 3.4). The short piece's fitted direction is tilted by its connector.
3. **The axis is not the centreline.** At the pick position the PCA axis is about 11 px from the wire centre (Figure 18). The final $\arg\max D$ step is what puts the point on the wire's centre.
4. **Ties in the final step.** Several pixels can share the maximum $D$ in a slice. `np.argmax` takes the first in scan order, which can shift the point by a few pixels *along* the wire inside the 8-px slice.
5. **Letterbox in `_resize_mask`.** Up to ±2.7 px of vertical misalignment near the image edges (Equation 4); measured at most 2 px.
6. **Bridge-shaped masks.** On OpenCV masks the label region is the 15-px bridge, so its radius is about 7.5–8.6 px rather than the wire's ~12 (Figure 20). It still counts as "wire-thin" in Equation (35).
7. **Docstring versus code.** `pickup_point`'s docstring describes preferring the dark-wire side. The code heads toward the label side and then avoids landing on white pixels.

---

## 7. Reproducing this report

All commands run from the project folder in WSL:

```bash
python3 trace_export.py --image Sheet_Data_1_folder/image84.png --image Sheet_Data_1_folder/image85.png
```

```bash
python3 docs/pipeline_report/make_figures.py
```

```bash
python3 docs/pipeline_report/build_pdf.py
```

The first command records the traces (`traces/*.json`). The second draws every figure and writes `figures/numbers.json`, which holds all the derived numbers quoted above. The third turns this Markdown into `pipeline_report.pdf`. The interactive version of the same walk-through is `pipeline_explorer.html`.

---

## Appendix A. Settings used (`config.yaml`)

| Key | Value | Used in |
|---|---|---|
| `model_confidence` | 0.25 | Eq. (1) |
| `mask_width_threshold_px` | 500 | Eqs. (8), (14) |
| `cv_min_piece_area` | 300 | §2.3, §3.4 |
| `cv_max_gap_px` | 120 | Eq. (9) |
| `cv_max_join_angle_deg` | 25 | Eq. (10) |
| `cv_join_width_px` | 15 | §2.5, §3.4 |
| `cv_min_cable_area` | 8000 | Eq. (14) |
| `mask_overlap_thresh` | 0.1 | Eq. (15) |
| `cv_gaussian_ksize` | 5 | Eqs. (17)–(18) |
| `cv_binary_threshold` | 70 | Eq. (19) |
| `pickup_fraction` | 0.5 | Eq. (32) |
| `pickup_offset_px` | 45 | Eq. (38) |
| `pickup_marker_radius` | 6 | §4.9 |
| `band_px` (code default) | 4 | Eqs. (29), (41) |
| `clip_window` (code default) | 0.25 | Eq. (32) |
| lateral limit (code default) | 40 | Eqs. (12), (13) |

## Appendix B. Where each function appears

| Function | File | Section |
|---|---|---|
| `MODEL.process`, `MODEL.infer` | model.py | 1, 2.1 |
| `_resize_mask` | cv_masks.py | 2.2 |
| `_piece_ends`, `_contour_length`, `_contour_centroid` | cv_masks.py | 2.3 |
| `_ends_should_join`, `_angle_deg` | cv_masks.py | 2.4, 3.4 |
| `join_instance_masks`, `_bridge_pieces_on_mask`, `_contour_passes_size` | cv_masks.py | 2.5, 3.4, 3.5 |
| `suppress_overlapping_masks` | cv_masks.py | 2.6 |
| `detect_cv_masks`, `_find_cable_contours`, `_odd_ksize`, `_yolo_union_mask` | cv_masks.py | 3.1–3.5 |
| `overlay_masks`, `overlay_cv_masks` | cv_masks.py | 3.6 |
| `pickup_points_for_masks`, `pickup_point` | utils.py | 4 |
| `_mask_pca_axis` | utils.py | 4.1, 4.2 |
| `_radius_profile`, `_gray_along` | utils.py | 4.4 |
| `_find_clip_index` | utils.py | 4.6 |
| `_walk_off_bulge` | utils.py | 4.8 |
| `_point_at_projection`, `draw_pickup_points` | utils.py | 4.9 |
| `filter_result_by_box_size` | utils.py | not called by `infer()` |

import argparse
import os
import time

import cv2
import numpy as np
import torch
import yaml
from loguru import logger
from ultralytics import YOLO

from cv_masks import (
    detect_cv_masks,
    join_instance_masks,
    overlay_cv_masks,
    overlay_masks,
    suppress_overlapping_masks,
)
from utils import draw_pickup_points, pickup_points_for_masks

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class MODEL:
    """
    Simple wrapper around one YOLO segmentation model.

    - Loads the model once (and warms it up).
    - Reads parameters from a YAML file into self.params (a dict).
    - reload_params_if_changed() re-reads the YAML only when the file changed.
    """

    def __init__(self, yaml_path="config.yaml", log_path="logs/model.log", warmup_runs=1):
        # ---------------- logger ----------------
        self.logger = logger
        try:
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
            self.logger.add(log_path, rotation="100 MB", retention="7 days")
        except Exception as e:
            # Logging to file failed; console logging still works.
            self.logger.error(f"Could not create log file '{log_path}': {e}")

        self.logger.info("Initializing model...")
        start = time.time()

        # ---------------- parameters ----------------
        self.yaml_path = yaml_path
        self.params = {}          # all values from the YAML file live here
        self._yaml_mtime = None   # last modified time of the YAML file
        if not self.load_params():
            raise FileNotFoundError(f"Could not load parameters from: {yaml_path}")

        model_path = self.get_param("model_path")
        if not model_path:
            raise ValueError("YAML must include 'model_path'")
        self.model_confidence = float(self.get_param("model_confidence", 0.5))
        self.warmup_runs = int(self.get_param("warmup_runs", 1))
        self.mask_width_threshold_px = int(self.get_param("mask_width_threshold_px", 600))
        self.cv_gaussian_ksize = int(self.get_param("cv_gaussian_ksize", 5))
        if self.cv_gaussian_ksize % 2 == 0:
            self.cv_gaussian_ksize += 1
        self.cv_binary_threshold = int(self.get_param("cv_binary_threshold", 70))
        self.cv_min_piece_area = int(self.get_param("cv_min_piece_area", 300))
        self.cv_max_gap_px = int(self.get_param("cv_max_gap_px", 120))
        self.cv_max_join_angle_deg = float(self.get_param("cv_max_join_angle_deg", 25))
        self.cv_join_width_px = int(self.get_param("cv_join_width_px", 15))
        self.cv_min_cable_area = int(self.get_param("cv_min_cable_area", 8000))
        self.mask_overlap_thresh = float(self.get_param("mask_overlap_thresh", 0.1))
        self.enable_pickup = bool(self.get_param("enable_pickup", True))
        self.pickup_fraction = float(self.get_param("pickup_fraction", 0.5))
        self.pickup_offset_px = float(self.get_param("pickup_offset_px", 45))
        self.pickup_marker_radius = int(self.get_param("pickup_marker_radius", 6))
        # ---------------- model ----------------
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.logger.info(
            f"Loaded model '{model_path}' on {self.device} "
            f"(confidence={self.model_confidence}, warmup_runs={self.warmup_runs}, "
            f"mask_w>={self.mask_width_threshold_px}, "
            f"cv_ksize={self.cv_gaussian_ksize}, cv_thresh={self.cv_binary_threshold}, "
            f"cv_gap={self.cv_max_gap_px}, cv_join_w={self.cv_join_width_px}, "
            f"mask_ovlp>={self.mask_overlap_thresh}, pickup={self.enable_pickup})"
        )

        if getattr(self.model, "task", None) != "segment":
            self.logger.warning(f"Model task is '{self.model.task}', expected 'segment'")

        # Warm-up: first prediction is slow, so do it now instead of on the first real image
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(self.warmup_runs):
            self.model.predict(dummy, conf=self.model_confidence, verbose=False)

        self.logger.info(f"Model ready in {time.time() - start:.2f} s")

    # ------------------------------------------------------------------ #
    def load_params(self):
        """
        Read the YAML file into self.params.
        If the file is missing or broken, keep the previous parameters.
        Returns True if parameters were loaded, False otherwise.
        """
        if not os.path.exists(self.yaml_path):
            self.logger.warning(f"YAML file not found: {self.yaml_path} (keeping old parameters)")
            return False

        try:
            with open(self.yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            self.logger.error(f"Could not read YAML '{self.yaml_path}': {e} (keeping old parameters)")
            return False

        if not isinstance(data, dict):
            self.logger.error("YAML top level must be key: value pairs (keeping old parameters)")
            return False

        self.params = data
        self._yaml_mtime = os.path.getmtime(self.yaml_path)
        self.logger.info(f"Loaded parameters from {self.yaml_path}: {self.params}")
        return True

    def reload_params_if_changed(self):
        """
        Call this before each inference.
        Re-reads the YAML only if the file was modified since the last load.
        Returns True if parameters were reloaded.
        """
        try:
            mtime = os.path.getmtime(self.yaml_path)
        except OSError:
            return False  # file missing right now -> keep current parameters

        if mtime == self._yaml_mtime:
            return False  # nothing changed

        self.logger.info("YAML file changed, reloading parameters...")
        ok = self.load_params()
        if ok:
            self.model_confidence = float(self.get_param("model_confidence", self.model_confidence))
            self.warmup_runs = int(self.get_param("warmup_runs", self.warmup_runs))
            self.mask_width_threshold_px = int(self.get_param("mask_width_threshold_px", self.mask_width_threshold_px))
            self.cv_gaussian_ksize = int(self.get_param("cv_gaussian_ksize", self.cv_gaussian_ksize))
            if self.cv_gaussian_ksize % 2 == 0:
                self.cv_gaussian_ksize += 1
            self.cv_binary_threshold = int(self.get_param("cv_binary_threshold", self.cv_binary_threshold))
            self.cv_min_piece_area = int(self.get_param("cv_min_piece_area", self.cv_min_piece_area))
            self.cv_max_gap_px = int(self.get_param("cv_max_gap_px", self.cv_max_gap_px))
            self.cv_max_join_angle_deg = float(
                self.get_param("cv_max_join_angle_deg", self.cv_max_join_angle_deg)
            )
            self.cv_join_width_px = int(self.get_param("cv_join_width_px", self.cv_join_width_px))
            self.cv_min_cable_area = int(self.get_param("cv_min_cable_area", self.cv_min_cable_area))
            self.mask_overlap_thresh = float(
                self.get_param("mask_overlap_thresh", self.mask_overlap_thresh)
            )
            self.enable_pickup = bool(self.get_param("enable_pickup", self.enable_pickup))
            self.pickup_fraction = float(self.get_param("pickup_fraction", self.pickup_fraction))
            self.pickup_offset_px = float(self.get_param("pickup_offset_px", self.pickup_offset_px))
            self.pickup_marker_radius = int(
                self.get_param("pickup_marker_radius", self.pickup_marker_radius)
            )
        return ok

    def get_param(self, key, default=None):
        """Safe way to read one parameter: returns `default` if the key is missing."""
        return self.params.get(key, default)

    def infer(
        self,
        image,
        confidence_threshold=None,
        model=None,
        mask_width_threshold_px=None,
        cv_gaussian_ksize=None,
        cv_binary_threshold=None,
        cv_min_piece_area=None,
        cv_max_gap_px=None,
        cv_max_join_angle_deg=None,
        cv_join_width_px=None,
        cv_min_cable_area=None,
        mask_overlap_thresh=None,
        enable_pickup=None,
        pickup_fraction=None,
        pickup_offset_px=None,
        pickup_marker_radius=None,
    ):
        """
        Run segmentation on one image and return a mask-annotated copy.

        On any failure, logs the error and returns a copy of the input image
        (or None if the image itself is invalid).
        """
        start = time.time()

        if confidence_threshold is None:
            confidence_threshold = self.model_confidence
        if model is None:
            model = self.model
        if mask_width_threshold_px is None:
            mask_width_threshold_px = self.mask_width_threshold_px
        if cv_gaussian_ksize is None:
            cv_gaussian_ksize = self.cv_gaussian_ksize
        if cv_binary_threshold is None:
            cv_binary_threshold = self.cv_binary_threshold
        if cv_min_piece_area is None:
            cv_min_piece_area = self.cv_min_piece_area
        if cv_max_gap_px is None:
            cv_max_gap_px = self.cv_max_gap_px
        if cv_max_join_angle_deg is None:
            cv_max_join_angle_deg = self.cv_max_join_angle_deg
        if cv_join_width_px is None:
            cv_join_width_px = self.cv_join_width_px
        if cv_min_cable_area is None:
            cv_min_cable_area = self.cv_min_cable_area
        if mask_overlap_thresh is None:
            mask_overlap_thresh = self.mask_overlap_thresh
        if enable_pickup is None:
            enable_pickup = self.enable_pickup
        if pickup_fraction is None:
            pickup_fraction = self.pickup_fraction
        if pickup_offset_px is None:
            pickup_offset_px = self.pickup_offset_px
        if pickup_marker_radius is None:
            pickup_marker_radius = self.pickup_marker_radius

        # ---- validate inputs ----
        if image is None:
            self.logger.error("infer: image is None")
            return None
        try:
            if not isinstance(image, np.ndarray) or image.size == 0:
                self.logger.error("infer: image must be a non-empty numpy array")
                return None
        except Exception as e:
            self.logger.error(f"infer: bad image: {e}")
            return None

        if model is None:
            self.logger.error("infer: model is None")
            return None

        try:
            confidence_threshold = float(confidence_threshold)
            if not (0.0 <= confidence_threshold <= 1.0):
                self.logger.warning(f"infer: confidence {confidence_threshold} out of [0, 1], clamping")
                confidence_threshold = max(0.0, min(1.0, confidence_threshold))
        except (TypeError, ValueError) as e:
            self.logger.error(f"infer: bad confidence_threshold={confidence_threshold!r}: {e}")
            return image.copy()

        # ---- predict + annotate ----
        try:
            results = model.predict(image, conf=confidence_threshold, verbose=False)
            if not results:
                self.logger.warning("infer: no results from model")
                return image.copy()

            result = results[0]
            h, w = image.shape[:2]

            raw_masks, raw_scores = [], []
            if result.masks is not None:
                raw_masks = list(result.masks.data.detach().cpu().numpy())
                if result.boxes is not None and len(result.boxes):
                    raw_scores = [
                        float(c) for c in result.boxes.conf.detach().cpu().numpy()
                    ]
                else:
                    raw_scores = [1.0] * len(raw_masks)

            n_raw = len(raw_masks)
            yolo_masks, yolo_scores = join_instance_masks(
                raw_masks,
                (h, w),
                max_gap_px=cv_max_gap_px,
                max_join_angle_deg=cv_max_join_angle_deg,
                join_width_px=cv_join_width_px,
                min_piece_area=cv_min_piece_area,
                min_cable_area=cv_min_cable_area,
                min_length=mask_width_threshold_px,
                scores=raw_scores if raw_scores else None,
            )
            n_joined = len(yolo_masks)
            yolo_masks, yolo_scores = suppress_overlapping_masks(
                yolo_masks, yolo_scores, overlap_thresh=mask_overlap_thresh
            )
            n_yolo = len(yolo_masks)
            if n_raw and self.logger is not None:
                self.logger.info(
                    f"infer: yolo masks raw={n_raw} -> joined={n_joined} "
                    f"-> after_overlap={n_yolo}"
                )

            cv_only = detect_cv_masks(
                image,
                yolo_masks,
                mask_width_threshold_px,
                cv_gaussian_ksize,
                cv_binary_threshold,
                cv_min_piece_area,
                cv_max_gap_px,
                cv_max_join_angle_deg,
                cv_join_width_px,
                cv_min_cable_area,
                self.logger,
            )

            # Blue = processed YOLO, green = CV-only (plot() skipped: masks were remapped).
            annotated = overlay_masks(
                image, yolo_masks, color=(255, 0, 0), alpha=0.45, scores=yolo_scores
            )
            annotated = overlay_cv_masks(annotated, cv_only)

            n_pick = 0
            if enable_pickup:
                all_masks = list(yolo_masks) + list(cv_only)
                picks = pickup_points_for_masks(
                    all_masks,
                    fraction=pickup_fraction,
                    offset_px=pickup_offset_px,
                    image=image,
                )
                n_pick = sum(1 for p in picks if p is not None)
                annotated = draw_pickup_points(
                    annotated, picks, radius=pickup_marker_radius, color=(0, 255, 255)
                )

            self.logger.info(
                f"infer: yolo={n_yolo}, cv={len(cv_only)}, pick={n_pick}, "
                f"conf={confidence_threshold}, took {time.time() - start:.3f} s"
            )
            return annotated
        except Exception as e:
            self.logger.error(f"infer failed: {e}")
            try:
                return image.copy()
            except Exception:
                return None

    def process(self, image):
        """Run inference on one image and return the annotated result."""
        self.reload_params_if_changed()
        return self.infer(image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO seg inference on a folder of images.")
    parser.add_argument("--input", required=True, help="Folder of input images")
    parser.add_argument("--output", required=True, help="Folder for annotated images")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    m = MODEL(yaml_path=args.config)

    if not os.path.isdir(args.input):
        m.logger.error(f"Input folder not found: {args.input}")
        raise SystemExit(1)

    os.makedirs(args.output, exist_ok=True)

    names = sorted(
        name for name in os.listdir(args.input)
        if os.path.splitext(name)[1].lower() in IMAGE_EXTS
    )
    if not names:
        m.logger.warning(f"No images found in {args.input}")
        raise SystemExit(0)

    m.logger.info(f"{len(names)} image(s): {args.input} -> {args.output}")
    ok, fail = 0, 0

    for name in names:
        in_path = os.path.join(args.input, name)
        out_path = os.path.join(args.output, name)
        try:
            image = cv2.imread(in_path)
            if image is None:
                m.logger.error(f"Could not read image: {in_path}")
                fail += 1
                continue

            annotated = m.process(image)
            if annotated is None:
                m.logger.error(f"process returned None for {name}")
                fail += 1
                continue

            if not cv2.imwrite(out_path, annotated):
                m.logger.error(f"Could not save: {out_path}")
                fail += 1
                continue

            m.logger.info(f"Saved {out_path}")
            ok += 1
        except Exception as e:
            m.logger.error(f"Failed on {name}: {e}")
            fail += 1

    m.logger.info(f"Done: {ok} saved, {fail} failed")

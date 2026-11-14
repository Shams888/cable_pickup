import argparse
import os
import time

import cv2
import numpy as np
import torch
import yaml
from loguru import logger
from ultralytics import YOLO

from cv_masks import detect_cv_masks, overlay_cv_masks
from utils import filter_result_by_box_size

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
        self.cv_gaussian_ksize = int(self.get_param("cv_gaussian_ksize", 3))
        if self.cv_gaussian_ksize % 2 == 0:
            self.cv_gaussian_ksize += 1
        self.cv_binary_threshold = int(self.get_param("cv_binary_threshold", 70))
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
            f"cv_ksize={self.cv_gaussian_ksize}, cv_thresh={self.cv_binary_threshold})"
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
            result = filter_result_by_box_size(
                result, mask_width_threshold_px, self.logger
            )

            n_yolo = 0 if result.masks is None else len(result.masks)
            yolo_masks = []
            if result.masks is not None:
                yolo_masks = list(result.masks.data.detach().cpu().numpy())

            cv_only = detect_cv_masks(
                image,
                yolo_masks,
                mask_width_threshold_px,
                cv_gaussian_ksize,
                cv_binary_threshold,
                self.logger,
            )

            annotated = result.plot() if n_yolo > 0 else image.copy()
            annotated = overlay_cv_masks(annotated, cv_only)  # green = CV-only
            self.logger.info(
                f"infer: yolo={n_yolo}, cv={len(cv_only)}, conf={confidence_threshold}, "
                f"took {time.time() - start:.3f} s"
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

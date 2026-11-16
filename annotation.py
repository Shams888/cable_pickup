"""
annotation.py

Draw on an OpenCV image and build an SVG at the same time.

Example:
    ann = Annotator(image)
    ann.box(10, 20, 100, 120, color=(0, 255, 0), label="H #0")
    ann.circle((55, 70), 5, color=(0, 0, 255))
    ann.polygon([[10, 10], [50, 10], [30, 40]], color=(255, 0, 0))
    ann.text("Hello", (10, 30))

    annotated_image = ann.get_image()
    svg_string = ann.get_svg()
"""

from xml.sax.saxutils import escape

import cv2
import numpy as np


def bgr_to_svg(color):
    """OpenCV uses BGR, SVG uses RGB."""
    b, g, r = color
    return f"rgb({int(r)},{int(g)},{int(b)})"


class Annotator:
    def __init__(self, image, draw_on_image=True):
        """
        image: the image to draw on (a copy is made, the original is not changed)
        draw_on_image: set False to only build the SVG (faster)
        """
        if image is None or image.size == 0:
            raise ValueError("Annotator needs a valid image")

        self.image = image.copy()
        self.draw_on_image = draw_on_image
        self.height, self.width = image.shape[:2]
        self.svg_parts = []

    # ------------------------------------------------------------------ #
    def box(self, x1, y1, x2, y2, color=(0, 255, 0), thickness=2, label=None):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        if self.draw_on_image:
            cv2.rectangle(self.image, (x1, y1), (x2, y2), color, thickness)

        self.svg_parts.append(
            f'<rect x="{x1}" y="{y1}" width="{x2 - x1}" height="{y2 - y1}" '
            f'fill="none" stroke="{bgr_to_svg(color)}" stroke-width="{thickness}"/>'
        )

        if label:
            self.text(label, (x1, max(y1 - 6, 12)), color=color)

    def circle(self, center, radius=5, color=(0, 0, 255), filled=True, thickness=2):
        cx, cy = int(center[0]), int(center[1])
        radius = int(radius)

        if self.draw_on_image:
            cv2.circle(self.image, (cx, cy), radius, color, -1 if filled else thickness)

        svg_color = bgr_to_svg(color)
        fill = svg_color if filled else "none"
        self.svg_parts.append(
            f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="{fill}" '
            f'stroke="{svg_color}" stroke-width="{thickness}"/>'
        )

    def polygon(self, points, color=(255, 0, 0), thickness=2, fill_opacity=0.3):
        """points: list of [x, y] (e.g. a YOLO segmentation mask outline)."""
        if points is None or len(points) < 3:
            return  # not a valid polygon, skip

        pts = np.array(points, dtype=np.int32)

        if self.draw_on_image:
            if fill_opacity > 0:
                overlay = self.image.copy()
                cv2.fillPoly(overlay, [pts], color)
                self.image = cv2.addWeighted(overlay, fill_opacity, self.image, 1 - fill_opacity, 0)
            cv2.polylines(self.image, [pts], True, color, thickness)

        svg_color = bgr_to_svg(color)
        pts_str = " ".join(f"{x},{y}" for x, y in pts.tolist())
        self.svg_parts.append(
            f'<polygon points="{pts_str}" fill="{svg_color}" fill-opacity="{fill_opacity}" '
            f'stroke="{svg_color}" stroke-width="{thickness}"/>'
        )

    def text(self, text, position, color=(255, 255, 255), font_scale=0.5, thickness=1):
        x, y = int(position[0]), int(position[1])
        text = str(text)

        if self.draw_on_image:
            cv2.putText(self.image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, color, thickness, cv2.LINE_AA)

        font_size = int(font_scale * 30)  # rough match to OpenCV text size
        self.svg_parts.append(
            f'<text x="{x}" y="{y}" fill="{bgr_to_svg(color)}" '
            f'font-size="{font_size}" font-family="sans-serif">{escape(text)}</text>'
        )

    # ------------------------------------------------------------------ #
    def get_image(self):
        return self.image

    def get_svg(self):
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" '
            f'height="{self.height}" viewBox="0 0 {self.width} {self.height}">'
            + "".join(self.svg_parts)
            + "</svg>"
        )

    def save_svg(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.get_svg())

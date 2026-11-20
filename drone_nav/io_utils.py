"""
Stage 2 — Data loading and preprocessing.

* Reads the region map and the sequence of drone frames from user-given paths.
* Sorts frames in *natural* order (crop_2 < crop_10), because the route order
  is encoded only in file names.
* Converts to grayscale, optionally equalises contrast (CLAHE) and downscales.

Lecture link: "Experiment with image downscaling for better feature matching"
(task notes) and step 1 "Preprocess: ... scale to map pixels" of the
frame-to-map pipeline (lecture, slide 25).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")


def natural_key(path: str):
    """Sort key that orders 'crop_2' before 'crop_10'."""
    name = os.path.basename(path)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def imread_any(path: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """cv2.imread that also works with non-ASCII (e.g. Cyrillic) Windows paths."""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    if img is None:
        raise IOError(f"Cannot read image: {path}")
    return img


def imwrite_any(path: str, img: np.ndarray) -> None:
    """cv2.imwrite counterpart of imread_any (unicode-safe)."""
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise IOError(f"Cannot encode image: {path}")
    buf.tofile(path)


def list_frames(frames_dir: str) -> List[str]:
    """Return all image files from a folder, in natural (flight) order."""
    files = [os.path.join(frames_dir, f) for f in os.listdir(frames_dir)
             if f.lower().endswith(IMAGE_EXT)]
    if not files:
        raise FileNotFoundError(f"No images found in {frames_dir}")
    return sorted(files, key=natural_key)


@dataclass
class Prepared:
    """A grayscale working copy of an image plus the factor used to get it.

    `scale` maps ORIGINAL pixel coords to WORKING coords: p_work = p_orig * scale.
    All final results are reported in original map pixels.
    """
    gray: np.ndarray
    scale: float


def to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def prepare(img: np.ndarray, scale: float = 1.0, max_side: int | None = None,
            clahe: bool = True) -> Prepared:
    """Grayscale -> (optional) downscale -> (optional) CLAHE.

    Downscaling uses INTER_AREA (proper anti-aliasing) so that small-scale
    noise/moire disappears instead of becoming fake keypoints.
    CLAHE equalises local contrast, which reduces the effect of different
    lighting between map and drone photo (lecture slide 6: "Light and shadows").
    """
    g = to_gray(img)
    h, w = g.shape[:2]
    if max_side is not None and max(h, w) * scale > max_side:
        scale = max_side / max(h, w)
    if abs(scale - 1.0) > 1e-3:
        g = cv2.resize(g, (max(1, round(w * scale)), max(1, round(h * scale))),
                       interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    if clahe:
        g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    return Prepared(g, scale)

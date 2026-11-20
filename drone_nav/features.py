"""
Stage 3 — Feature extraction and matching (steps 1-4 of the lecture's
five-step pipeline: Detect -> Describe -> Match -> Filter; slide 7).

Detectors (lecture slides 9-11):
  * SIFT  (default) — DoG blobs, 128-float descriptor, invariant to scale and
            rotation; the lecture's recommended classic baseline for drones,
            because altitude changes scale and heading changes rotation.
  * AKAZE — binary, faster, also scale/rotation invariant.
  * ORB   — fastest, binary; weakest across scale changes.

Matching (slide 12):
  * FLANN kd-tree for float descriptors, brute force Hamming for binary ones.
  * Lowe ratio test d1/d2 < ratio (0.7–0.8 in the lecture, default 0.75).
  * Optional mutual cross-check.

Own addition: `MapIndex` — map features are computed ONCE, and a query can be
restricted to a spatial window (the lecture's "Select region: map crop around
the predicted position", slide 25) without recomputing anything.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


def create_detector(name: str = "sift", n_features: int = 0):
    name = name.lower()
    if name == "sift":
        # contrastThreshold a bit lower than default -> more points on
        # low-texture fields; nfeatures=0 means "no cap".
        return cv2.SIFT_create(nfeatures=n_features, contrastThreshold=0.02)
    if name == "akaze":
        if not hasattr(cv2, "AKAZE_create"):   # moved out of main module in OpenCV 5
            raise RuntimeError("AKAZE is not available in this OpenCV build (use opencv 4.x)")
        return cv2.AKAZE_create(threshold=5e-4)
    if name == "orb":
        return cv2.ORB_create(nfeatures=n_features or 20000, scaleFactor=1.2,
                              nlevels=12, fastThreshold=10)
    raise ValueError(f"Unknown detector: {name}")


def is_binary(name: str) -> bool:
    return name.lower() in ("orb", "akaze")


def detect(detector, gray: np.ndarray, tile: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Detect + describe. Returns (N,2) float32 keypoint coords and descriptors.

    `tile` > 0 splits a big image into overlapping tiles. Own addition:
    without it SIFT on a huge panorama spends its features on the most
    contrasted areas and leaves fields almost empty.
    """
    if tile <= 0 or max(gray.shape) <= tile:
        kps, des = detector.detectAndCompute(gray, None)
        pts = np.float32([k.pt for k in kps]).reshape(-1, 2)
        return pts, des
    h, w = gray.shape
    pad = 32
    all_pts, all_des = [], []
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            ya, xa = max(0, y0 - pad), max(0, x0 - pad)
            yb, xb = min(h, y0 + tile + pad), min(w, x0 + tile + pad)
            kps, des = detector.detectAndCompute(gray[ya:yb, xa:xb], None)
            if des is None:
                continue
            pts = np.float32([k.pt for k in kps]) + (xa, ya)
            # keep a point only in its own tile (no duplicates from the padding)
            keep = ((pts[:, 0] >= x0) & (pts[:, 0] < x0 + tile) &
                    (pts[:, 1] >= y0) & (pts[:, 1] < y0 + tile))
            all_pts.append(pts[keep]); all_des.append(des[keep])
    return np.concatenate(all_pts), np.concatenate(all_des)


def _matcher(binary: bool):
    if binary:
        return cv2.BFMatcher(cv2.NORM_HAMMING)
    return cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))


def match(des_q: np.ndarray, des_t: np.ndarray, binary: bool,
          ratio: float = 0.75, cross_check: bool = False) -> np.ndarray:
    """Return (M,2) int array of index pairs (query_idx, train_idx)."""
    if des_q is None or des_t is None or len(des_q) < 2 or len(des_t) < 2:
        return np.empty((0, 2), int)
    m = _matcher(binary)
    knn = m.knnMatch(des_q, des_t, k=2)
    pairs = [(a.queryIdx, a.trainIdx) for a, *rest in
             (p for p in knn if len(p) == 2) if a.distance < ratio * rest[0].distance]
    pairs = np.array(pairs, int).reshape(-1, 2)
    if cross_check and len(pairs):
        back = m.match(des_t, des_q)
        rev = {b.queryIdx: b.trainIdx for b in back}
        pairs = np.array([p for p in pairs if rev.get(p[1]) == p[0]], int).reshape(-1, 2)
    return pairs


@dataclass
class MapIndex:
    """All map keypoints/descriptors, computed once, with window queries."""
    pts: np.ndarray        # (N,2) in WORKING map coords
    des: np.ndarray
    binary: bool

    def window(self, cx: float, cy: float, radius: float) -> np.ndarray:
        """Indices of map keypoints inside a square window around (cx, cy)."""
        sel = ((np.abs(self.pts[:, 0] - cx) < radius) &
               (np.abs(self.pts[:, 1] - cy) < radius))
        return np.nonzero(sel)[0]

    def match(self, des_q: np.ndarray, subset: Optional[np.ndarray] = None,
              ratio: float = 0.75, cross_check: bool = False):
        """Match query descriptors to all map features or to a subset."""
        if subset is None:
            return match(des_q, self.des, self.binary, ratio, cross_check)
        pairs = match(des_q, self.des[subset], self.binary, ratio, cross_check)
        if len(pairs):
            pairs[:, 1] = subset[pairs[:, 1]]
        return pairs

"""
Stage 4 — Frame-to-map localization (lecture Part 3, slide 25):

  1 Preprocess   -> io_utils.prepare (+ adaptive "scale to map pixels", below)
  2 Select region-> MapIndex.window around the predicted position
  3 Match        -> features.match (ratio test)
  4 Verify       -> RANSAC / MAGSAC++ homography + geometric sanity checks
  5 Georeference -> frame centre -> map pixel; heading; footprint polygon

Homography (slide 13-14): the camera looks down on (almost) flat ground, so a
3x3 H with 8 DoF maps frame pixels to map pixels; 4 matches define it, RANSAC
keeps the matches that agree on one H.

Own additions (not in the lecture):
  * Sanity checks on H (convex, non-mirrored footprint, plausible scale and
    perspective, consistent with previous scale) — rejects "lucky" RANSAC fits
    on repeated texture that still have >= min_inliers.
  * Adaptive frame scale: after the first good fixes we know how many map
    pixels one frame pixel covers and resize next frames so that frame and
    map have ~equal pixel size. This automates the task hint "experiment with
    downscaling" and the lecture's "scale to map pixels".
  * Automatic fallback: local window -> whole map, when the local search fails
    (prior was wrong or the drone made a jump).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from .features import MapIndex, create_detector, detect, is_binary
from .io_utils import prepare


@dataclass
class FrameFix:
    """Localization result for one frame. All coordinates in ORIGINAL map px."""
    index: int
    name: str
    ok: bool = False
    center: Optional[np.ndarray] = None      # (2,)
    corners: Optional[np.ndarray] = None     # (4,2) TL, TR, BR, BL
    heading_deg: float = float("nan")        # 0 = map-up, clockwise positive
    gsd: float = float("nan")                # map px per original frame px
    n_kp: int = 0
    n_matches: int = 0
    n_inliers: int = 0
    rmse: float = float("nan")
    mode: str = ""                           # "local" / "global"
    frame_scale: float = 1.0
    H: Optional[np.ndarray] = None           # frame(orig) -> map(orig)
    notes: List[str] = field(default_factory=list)


def _scale_mat(s: float) -> np.ndarray:
    return np.diag([s, s, 1.0])


def _find_h(src: np.ndarray, dst: np.ndarray, thr: float):
    method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)   # MAGSAC++ (lecture slide 7)
    H, mask = cv2.findHomography(src, dst, method, thr, maxIters=10000, confidence=0.999)
    if H is None:
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, thr, maxIters=5000)
    return H, mask


class Localizer:
    def __init__(self, map_img: np.ndarray, detector: str = "sift",
                 map_max_side: int = 6000, map_scale: float = 1.0,
                 frame_scale: str | float = "auto", ratio: float = 0.75,
                 ransac_thr: float = 5.0, min_inliers: int = 15,
                 search_radius_factor: float = 2.5, tile: int = 2048,
                 clahe: bool = False, verbose: bool = True):
        self.det_name = detector
        self.det = create_detector(detector)
        self.ratio, self.thr, self.min_inl = ratio, ransac_thr, min_inliers
        self.search_k = search_radius_factor
        self.clahe, self.verbose = clahe, verbose
        self.map_shape = map_img.shape[:2]

        mp = prepare(map_img, scale=map_scale, max_side=map_max_side, clahe=clahe)
        self.s_m = mp.scale
        pts, des = detect(self.det, mp.gray, tile=tile)
        self.map = MapIndex(pts, des, is_binary(detector))
        self._log(f"[map] {self.map_shape[1]}x{self.map_shape[0]} px, working scale "
                  f"{self.s_m:.3f}, {len(pts)} keypoints")

        self.auto_scale = frame_scale == "auto"
        self.s_f = 0.5 if self.auto_scale else float(frame_scale)
        self._gsd_hist: List[float] = []
        self.prev: Optional[FrameFix] = None
        self._update = True
        self.velocity = np.zeros(2)

    # ------------------------------------------------------------------ utils
    def _log(self, msg: str):
        if self.verbose:
            print(msg, flush=True)

    def _predict(self) -> Optional[np.ndarray]:
        """Constant-velocity prediction of the next centre (map orig px)."""
        if self.prev is None or not self.prev.ok:
            return None
        return self.prev.center + self.velocity

    def _validate(self, H: np.ndarray, w: int, h: int) -> tuple[bool, str, np.ndarray, float]:
        """Geometric sanity checks of frame(orig) -> map(orig) homography."""
        c = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        poly = cv2.perspectiveTransform(c, H).reshape(-1, 2)
        if not np.all(np.isfinite(poly)):
            return False, "non-finite", poly, np.nan
        if not cv2.isContourConvex(poly.astype(np.float32)):
            return False, "non-convex footprint", poly, np.nan
        # mirrored footprint => orientation of polygon flips sign
        area_signed = 0.5 * np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) -
                                   np.roll(poly[:, 0], -1) * poly[:, 1])
        if area_signed <= 0:
            return False, "mirrored", poly, np.nan
        gsd = np.sqrt(area_signed / (w * h))
        sides = np.linalg.norm(poly - np.roll(poly, -1, axis=0), axis=1)
        if sides[0] / sides[2] > 3 or sides[2] / sides[0] > 3 or \
           sides[1] / sides[3] > 3 or sides[3] / sides[1] > 3:
            return False, "extreme perspective", poly, gsd
        if self._gsd_hist:
            ref = float(np.median(self._gsd_hist))
            if not (ref / 2.0 < gsd < ref * 2.0):
                return False, f"scale jump ({gsd:.3f} vs {ref:.3f})", poly, gsd
        return True, "", poly, gsd

    # ------------------------------------------------------------- main step
    def localize(self, idx: int, name: str, frame: np.ndarray,
                 prior: Optional[np.ndarray] = None, update_state: bool = True) -> FrameFix:
        """Localize one frame. `prior` (map orig px) overrides the motion
        prediction; `update_state=False` is used for the retry pass."""
        fix = FrameFix(idx, name, frame_scale=self.s_f)
        self._update = update_state
        h0, w0 = frame.shape[:2]
        fp = prepare(frame, scale=self.s_f, clahe=self.clahe)
        pts_f, des_f = detect(self.det, fp.gray)
        fix.n_kp = len(pts_f)
        if des_f is None or len(pts_f) < 8:
            fix.notes.append("too few keypoints")
            return self._finish(fix)

        pred = prior if prior is not None else self._predict()
        attempts = []
        if pred is not None and self._gsd_hist:
            # search window: a few frame footprints around the predicted centre
            foot = np.median(self._gsd_hist) * max(w0, h0)
            radius = self.search_k * foot * self.s_m
            attempts.append(("local", self.map.window(*(pred * self.s_m), radius)))
        attempts.append(("global", None))

        # frame(working) -> frame(orig) and map(working) -> map(orig)
        F = _scale_mat(fp.scale)
        Minv = _scale_mat(1.0 / self.s_m)
        for mode, subset in attempts:
            if subset is not None and len(subset) < self.min_inl:
                continue
            pairs = self.map.match(des_f, subset, self.ratio)
            fix.n_matches = len(pairs)
            if len(pairs) < self.min_inl:
                fix.notes.append(f"{mode}: {len(pairs)} matches")
                continue
            src = pts_f[pairs[:, 0]]
            dst = self.map.pts[pairs[:, 1]]
            Hw, mask = _find_h(src, dst, self.thr)
            if Hw is None:
                fix.notes.append(f"{mode}: no homography")
                continue
            inl = mask.ravel().astype(bool)
            if inl.sum() < self.min_inl:
                fix.notes.append(f"{mode}: {inl.sum()} inliers")
                continue
            H = Minv @ Hw @ F                       # frame orig -> map orig
            ok, why, poly, gsd = self._validate(H, w0, h0)
            if not ok:
                fix.notes.append(f"{mode}: {why}")
                continue
            proj = cv2.perspectiveTransform(src[inl].reshape(-1, 1, 2), Hw).reshape(-1, 2)
            fix.rmse = float(np.sqrt(np.mean(np.sum((proj - dst[inl]) ** 2, 1)))) / self.s_m
            ctr = cv2.perspectiveTransform(np.float32([[[w0 / 2, h0 / 2]]]), H).ravel()
            top = cv2.perspectiveTransform(np.float32([[[w0 / 2, 0]]]), H).ravel()
            v = top - ctr
            fix.ok, fix.mode, fix.H = True, mode, H
            fix.center, fix.corners, fix.gsd = ctr, poly, gsd
            fix.heading_deg = float(np.degrees(np.arctan2(v[0], -v[1])) % 360)
            fix.n_inliers = int(inl.sum())
            break
        return self._finish(fix)

    def _finish(self, fix: FrameFix) -> FrameFix:
        if fix.ok and self._update:
            if self.prev is not None and self.prev.ok:
                self.velocity = 0.5 * self.velocity + 0.5 * (fix.center - self.prev.center)
            self._gsd_hist = (self._gsd_hist + [fix.gsd])[-10:]
            if self.auto_scale:
                # want 1 working frame px == 1 working map px:
                #   s_f * (orig frame px) == s_m * gsd * (orig frame px)
                target = float(np.clip(self.s_m * np.median(self._gsd_hist), 0.1, 2.0))
                self.s_f = 0.5 * self.s_f + 0.5 * target
            self.prev = fix
        self._log(f"[{fix.index:3d}] {fix.name:<20} "
                  + (f"OK {fix.mode:<6} inl={fix.n_inliers:4d}/{fix.n_matches:<5d} "
                     f"xy=({fix.center[0]:7.1f},{fix.center[1]:7.1f}) "
                     f"hdg={fix.heading_deg:5.1f} rmse={fix.rmse:4.2f}"
                     if fix.ok else "FAIL " + "; ".join(fix.notes))
                  + f"  [fs={fix.frame_scale:.2f}]")
        return fix

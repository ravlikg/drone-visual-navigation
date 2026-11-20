"""
Stage 5 — Trajectory post-processing.

Task notes: "Use trajectory interpolation and smoothing (if needed)".
Lecture slide 25, step 6 "Fuse: Kalman filter with IMU and odometry".
We have no IMU, so the Kalman filter here fuses the visual fixes with a
constant-velocity motion model only.

Pipeline:
  1. outlier rejection  — a fix far from the median of its neighbours is
                          treated as a wrong match (own heuristic, robust to
                          a single bad RANSAC fit that passed the checks);
  2. Kalman filter + RTS smoother (constant velocity) — fills gaps of
     failed frames (interpolation) and smooths noise; measurement noise is
     scaled by 1/inliers, so confident fixes pull the track harder (own idea);
  3. heading — unwrap angles and smooth them with a small moving average.
"""
from __future__ import annotations

from typing import List

import numpy as np

from .localize import FrameFix


def raw_xy(fixes: List[FrameFix]) -> np.ndarray:
    """(N,2) centres, NaN where the frame was not localized."""
    out = np.full((len(fixes), 2), np.nan)
    for i, f in enumerate(fixes):
        if f.ok:
            out[i] = f.center
    return out


def reject_outliers(xy: np.ndarray, win: int = 3, k: float = 0.5) -> np.ndarray:
    """Mark as NaN fixes that do not fit the local motion of their neighbours.

    For every fix, a quadratic x(t), y(t) is fitted to up to `win` valid
    neighbours on each side (the fix itself excluded) and evaluated at t.
    A fix is rejected if it is further than k * (median step) from that
    prediction. A drone cannot jump half a step sideways between two frames,
    while a wrong homography usually lands hundreds of pixels away.
    Only the single worst fix is removed per pass, so one outlier cannot
    get its good neighbours rejected.
    """
    xy = xy.copy()
    for _ in range(len(xy)):
        valid = np.where(~np.isnan(xy[:, 0]))[0]
        if len(valid) < 6:
            break
        step = max(float(np.median(np.linalg.norm(np.diff(xy[valid], axis=0), axis=1)
                                    / np.diff(valid)[:, None].ravel())), 1.0)
        res = np.zeros(len(valid))
        for j, i in enumerate(valid):
            nb = np.r_[valid[max(0, j - win): j], valid[j + 1: j + 1 + win]]
            deg = 2 if len(nb) >= 4 else 1
            pred = [np.polyval(np.polyfit(nb - i, xy[nb, c], deg), 0) for c in (0, 1)]
            res[j] = np.linalg.norm(xy[i] - pred)
        worst = int(np.argmax(res))
        # end points are extrapolated, so they get a 2x looser bound
        limit = k * step * (2 if worst in (0, len(valid) - 1) else 1)
        if res[worst] <= limit:
            break
        xy[valid[worst]] = np.nan
    return xy


def kalman_rts(xy: np.ndarray, weights: np.ndarray | None = None,
               q: float = 1.0, r: float = 25.0) -> np.ndarray:
    """Constant-velocity Kalman filter + Rauch–Tung–Striebel smoother.

    State [x, y, vx, vy]; time step = 1 frame. NaN measurements are skipped
    (pure prediction), which is how missing frames get interpolated.
    q — process noise (px^2), r — base measurement noise (px^2).
    """
    n = len(xy)
    F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], float)
    Hm = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
    G = np.array([[0.5, 0], [0, 0.5], [1, 0], [0, 1]])
    Q = q * G @ G.T
    valid = np.where(~np.isnan(xy[:, 0]))[0]
    if len(valid) == 0:
        return xy.copy()
    x = np.zeros(4); x[:2] = xy[valid[0]]
    if len(valid) > 1:
        x[2:] = (xy[valid[1]] - xy[valid[0]]) / (valid[1] - valid[0])
    P = np.diag([r, r, 100.0, 100.0])
    xs_f, Ps_f, xs_p, Ps_p = [], [], [], []
    for t in range(n):
        if t > 0:
            x = F @ x; P = F @ P @ F.T + Q
        xs_p.append(x.copy()); Ps_p.append(P.copy())
        if not np.isnan(xy[t, 0]):
            w = 1.0 if weights is None else weights[t]
            R = np.eye(2) * r / max(w, 1e-3)
            S = Hm @ P @ Hm.T + R
            K = P @ Hm.T @ np.linalg.inv(S)
            x = x + K @ (xy[t] - Hm @ x)
            P = (np.eye(4) - K @ Hm) @ P
        xs_f.append(x.copy()); Ps_f.append(P.copy())
    xs = [None] * n
    xs[-1] = xs_f[-1]; Ps = Ps_f[-1]
    for t in range(n - 2, -1, -1):
        C = Ps_f[t] @ F.T @ np.linalg.inv(Ps_p[t + 1])
        xs[t] = xs_f[t] + C @ (xs[t + 1] - xs_p[t + 1])
        Ps = Ps_f[t] + C @ (Ps - Ps_p[t + 1]) @ C.T
    return np.array(xs)[:, :2]


def smooth_heading(fixes: List[FrameFix], win: int = 3) -> np.ndarray:
    """Unwrap headings, linearly fill gaps and smooth with moving average."""
    h = np.array([f.heading_deg if f.ok else np.nan for f in fixes])
    idx = np.arange(len(h)); ok = ~np.isnan(h)
    if ok.sum() == 0:
        return h
    un = np.degrees(np.unwrap(np.radians(h[ok])))
    full = np.interp(idx, idx[ok], un)
    if win > 1 and len(full) >= win:
        pad = np.pad(full, (win // 2, win // 2), mode="edge")
        full = np.convolve(pad, np.ones(win) / win, mode="valid")
    return full % 360


def estimate_noise(clean: np.ndarray, fixes: List[FrameFix]) -> tuple[float, float]:
    """Data-driven Kalman parameters (own addition).

    q — process noise: mean squared acceleration (2nd difference) of the
        track, so turns of a fast drone are not "cut" by the filter;
    r — measurement noise: from the homography reprojection RMSE scaled by
        the footprint size (a wrong-but-consistent fix is still ~1-3 px off).
    """
    v = clean[~np.isnan(clean[:, 0])]
    acc = np.diff(v, 2, axis=0) if len(v) > 3 else np.zeros((1, 2))
    q = float(max(1.0, np.mean(np.sum(acc ** 2, axis=1))))
    rm = [f.rmse for f in fixes if f.ok and np.isfinite(f.rmse)]
    r = float(max(4.0, 9 * np.median(rm) ** 2)) if rm else 25.0
    return q, r


def build_trajectory(fixes: List[FrameFix], q: float | None = None, r: float | None = None):
    """Returns dict with raw, cleaned and smoothed tracks + heading."""
    raw = raw_xy(fixes)
    clean = reject_outliers(raw)
    inl = np.array([f.n_inliers for f in fixes], float)
    w = inl / np.nanmedian(inl[inl > 0]) if np.any(inl > 0) else None
    q0, r0 = estimate_noise(clean, fixes)
    smooth = kalman_rts(clean, w, q=q if q is not None else q0, r=r if r is not None else r0)
    return dict(raw=raw, clean=clean, smooth=smooth, heading=smooth_heading(fixes),
                rejected=np.where(~np.isnan(raw[:, 0]) & np.isnan(clean[:, 0]))[0],
                failed=np.where(np.isnan(raw[:, 0]))[0])


def densify(track: np.ndarray, factor: int = 8) -> np.ndarray:
    """Catmull–Rom spline through track points (for smooth drawing / video)."""
    if len(track) < 3 or factor <= 1:
        return track
    p = np.vstack([track[0], track, track[-1]])
    out = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        for t in np.linspace(0, 1, factor, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t +
                              (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
                              (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(track[-1])
    return np.array(out)

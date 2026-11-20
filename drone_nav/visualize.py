"""
Stage 6 — Visualization.

Deliverable from the task: "An image with a map and visualized route with
information about start/finish and order of route points".
Optional (task notes): a video with zoomed-in map region and per-frame
trajectory, and drone-photo bounding boxes "with preservation of their size
and rotation" — the footprints come directly from the homography.

Everything here is own implementation with plain OpenCV drawing.
"""
from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from .localize import FrameFix
from .trajectory import densify

GREEN, RED, YELLOW, CYAN, WHITE, BLACK = (60, 200, 60), (40, 40, 230), (0, 220, 255), \
    (255, 220, 0), (255, 255, 255), (0, 0, 0)


def _p(pt, s=1.0):
    return int(round(pt[0] * s)), int(round(pt[1] * s))


def _label(img, text, org, scale=0.5, color=WHITE, thick=1):
    """Text with a dark outline so it is readable on any background."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _colormap(n: int) -> np.ndarray:
    """Per-point colours from start (blue-ish) to finish (yellow) — encodes order."""
    lut = cv2.applyColorMap(np.linspace(0, 255, max(n, 2)).astype(np.uint8)[:, None],
                            cv2.COLORMAP_PLASMA)[:, 0]
    return lut[:n]


def draw_route(map_img: np.ndarray, fixes: List[FrameFix], traj: dict,
               out_scale: float = 1.0, draw_footprints: bool = True,
               title: str = "UAV route reconstruction") -> np.ndarray:
    s = out_scale
    canvas = cv2.resize(map_img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s != 1 \
        else map_img.copy()
    # darken the map slightly so the overlay stands out
    canvas = cv2.addWeighted(canvas, 0.75, np.zeros_like(canvas), 0.25, 0)
    smooth, raw = traj["smooth"], traj["raw"]
    n = len(smooth)
    cols = _colormap(n)
    lw = max(2, int(3 * s))

    if draw_footprints:
        overlay = canvas.copy()
        for i, f in enumerate(fixes):
            if f.ok and f.corners is not None and i not in traj["rejected"]:
                poly = (f.corners * s).astype(np.int32)
                cv2.polylines(overlay, [poly], True, tuple(int(c) for c in cols[i]), 1, cv2.LINE_AA)
                # mark the "top" edge of the photo => shows drone heading
                cv2.line(overlay, tuple(poly[0]), tuple(poly[1]), WHITE, 1, cv2.LINE_AA)
        canvas = cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0)

    # smooth curve through the filtered positions
    curve = densify(smooth, 10)
    for k in range(len(curve) - 1):
        c = cols[min(int(k / 10), n - 1)]
        cv2.line(canvas, _p(curve[k], s), _p(curve[k + 1], s), tuple(int(v) for v in c), lw, cv2.LINE_AA)

    # raw measurements (small grey dots) and rejected outliers (red crosses)
    for i in range(n):
        if not np.isnan(raw[i, 0]):
            cv2.circle(canvas, _p(raw[i], s), 2, (200, 200, 200), -1, cv2.LINE_AA)
    for i in traj["rejected"]:
        cv2.drawMarker(canvas, _p(raw[i], s), RED, cv2.MARKER_TILTED_CROSS, 12, 2)

    # numbered route points (order of frames); interpolated points are hollow
    for i in range(n):
        pt = _p(smooth[i], s)
        col = tuple(int(v) for v in cols[i])
        interp = i in traj["failed"] or i in traj["rejected"]
        cv2.circle(canvas, pt, 6, col, 1 if interp else -1, cv2.LINE_AA)
        cv2.circle(canvas, pt, 7, BLACK, 1, cv2.LINE_AA)
        _label(canvas, str(i), (pt[0] + 7, pt[1] - 6), 0.45)

    # start / finish
    st, fin = _p(smooth[0], s), _p(smooth[-1], s)
    cv2.circle(canvas, st, 14, GREEN, 3, cv2.LINE_AA)
    _label(canvas, "START", (st[0] - 25, st[1] - 20), 0.7, GREEN, 2)
    cv2.drawMarker(canvas, fin, RED, cv2.MARKER_SQUARE, 26, 3)
    _label(canvas, "FINISH", (fin[0] - 30, fin[1] + 32), 0.7, RED, 2)

    # legend
    ok = sum(f.ok for f in fixes)
    lines = [title,
             f"frames: {n}   localized: {ok}   rejected: {len(traj['rejected'])}   "
             f"interpolated: {len(traj['failed']) + len(traj['rejected'])}",
             "line colour: frame order (dark = start, yellow = finish)",
             "filled dot = measured, hollow = interpolated, grey = raw fix",
             "thin polygons = photo footprints (white edge = photo top)"]
    x0, y0 = 12, 12
    box_h = 26 * len(lines) + 10
    cv2.rectangle(canvas, (x0 - 6, y0 - 6), (x0 + 620, y0 + box_h), (30, 30, 30), -1)
    for k, t in enumerate(lines):
        _label(canvas, t, (x0, y0 + 22 + 26 * k), 0.6 if k == 0 else 0.5, YELLOW if k == 0 else WHITE)
    return canvas


def render_video(path: str, map_img: np.ndarray, frames: List[np.ndarray],
                 fixes: List[FrameFix], traj: dict, sub: int = 8, fps: int = 24,
                 view: int = 900, zoom_px: Optional[float] = None) -> None:
    """Video: zoomed map window that follows the drone + inset of the current
    drone photo + whole-map minimap. `sub` intermediate steps between frames
    give the "real-time moving position" requested by the task.
    """
    import imageio.v2 as imageio
    smooth = traj["smooth"]; n = len(smooth)
    curve = densify(smooth, sub)
    head = np.degrees(np.unwrap(np.radians(traj["heading"])))
    head_d = np.interp(np.linspace(0, n - 1, len(curve)), np.arange(n), head)
    if zoom_px is None:   # map window = ~4 footprints wide
        foots = [np.linalg.norm(f.corners[1] - f.corners[0]) for f in fixes if f.ok]
        zoom_px = 4 * float(np.median(foots)) if foots else 800
    zoom_px = int(min(zoom_px, min(map_img.shape[:2]) - 2))
    H, W = map_img.shape[:2]
    mini_w = 300; ms = mini_w / W
    mini_base = cv2.resize(map_img, (mini_w, int(H * ms)), interpolation=cv2.INTER_AREA)
    cols = _colormap(n)
    full_route = map_img.copy()
    for k in range(len(curve) - 1):   # pre-draw faint full route
        cv2.line(full_route, _p(curve[k]), _p(curve[k + 1]), (120, 120, 120), 2, cv2.LINE_AA)

    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=7,
                                ffmpeg_params=["-preset", "veryfast"], macro_block_size=1)
    try:
        for k in range(len(curve)):
            i = min(int(round(k / sub)), n - 1)
            img = full_route.copy()
            for j in range(k):
                c = cols[min(j // sub, n - 1)]
                cv2.line(img, _p(curve[j]), _p(curve[j + 1]), tuple(int(v) for v in c), 3, cv2.LINE_AA)
            for j in range(0, i + 1):
                cv2.circle(img, _p(smooth[j]), 5, tuple(int(v) for v in cols[j]), -1, cv2.LINE_AA)
                _label(img, str(j), (_p(smooth[j])[0] + 6, _p(smooth[j])[1] - 6), 0.45)
            f = fixes[i]
            if f.ok and f.corners is not None:   # footprint of the nearest photo
                cv2.polylines(img, [f.corners.astype(np.int32)], True, CYAN, 2, cv2.LINE_AA)
            pos = curve[k]
            # drone marker: triangle pointing to heading
            a = np.radians(head_d[k]); d = np.array([np.sin(a), -np.cos(a)]); nrm = np.array([-d[1], d[0]])
            tri = np.array([pos + 16 * d, pos - 10 * d + 9 * nrm, pos - 10 * d - 9 * nrm], np.int32)
            cv2.fillPoly(img, [tri], RED, cv2.LINE_AA); cv2.polylines(img, [tri], True, WHITE, 1, cv2.LINE_AA)
            # crop zoom window around the drone
            half = zoom_px // 2
            cx = int(np.clip(pos[0], half, W - half)); cy = int(np.clip(pos[1], half, H - half))
            win = img[cy - half: cy + half, cx - half: cx + half]
            win = cv2.resize(win, (view, view), interpolation=cv2.INTER_LINEAR)
            # right panel: current drone photo + minimap
            panel = np.full((view, 340, 3), 25, np.uint8)
            ph = cv2.resize(frames[i], (320, int(320 * frames[i].shape[0] / frames[i].shape[1])))
            panel[30:30 + ph.shape[0], 10:330] = ph[: view - 30]
            _label(panel, f"frame {i}/{n - 1}", (10, 22), 0.6, YELLOW, 2)
            mini = mini_base.copy()
            cv2.polylines(mini, [(curve[: k + 1] * ms).astype(np.int32)], False, YELLOW, 1, cv2.LINE_AA)
            cv2.rectangle(mini, _p((cx - half, cy - half), ms), _p((cx + half, cy + half), ms), CYAN, 1)
            y = 30 + ph.shape[0] + 20
            if y + mini.shape[0] < view:
                panel[y: y + mini.shape[0], 20: 20 + mini.shape[1]] = mini
                ytxt = y + mini.shape[0] + 30
                _label(panel, f"x={pos[0]:.0f} y={pos[1]:.0f}", (10, ytxt), 0.55)
                _label(panel, f"heading={head_d[k] % 360:.0f} deg", (10, ytxt + 24), 0.55)
                _label(panel, f"inliers={f.n_inliers}" if f.ok else "interpolated", (10, ytxt + 48), 0.55)
            frame_out = np.hstack([win, panel])
            writer.append_data(cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB))
    finally:
        writer.close()

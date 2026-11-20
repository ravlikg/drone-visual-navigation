"""
UAV visual navigation: reconstruct a drone route from its photos and a
satellite map (It-Jim CV test task 2026).

Examples
--------
Standard level:
    python main.py --map data/global_map.png --frames data/crops --out outputs

With the optional fly-along video and debug overlays:
    python main.py --map data/global_map.png --frames data/crops --out outputs --video-out --debug

Outputs (in --out):
    route.png         map with the route, numbered points, START/FINISH, photo footprints
    trajectory.csv    per-frame position, heading, quality
    route.mp4         (with --video-out) zoomed fly-along video
    matches/*.jpg     (with --debug) frame <-> map match visualizations
"""
from __future__ import annotations

import argparse
import csv
import os
import time

import cv2
import numpy as np

from drone_nav.io_utils import imread_any, imwrite_any, list_frames
from drone_nav.localize import Localizer
from drone_nav.trajectory import build_trajectory
from drone_nav.visualize import draw_route, render_video


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", required=True, help="path to the satellite map image")
    ap.add_argument("--frames", required=True, help="folder with drone photos (natural-sorted by name)")
    ap.add_argument("--out", default="outputs", help="output folder")
    ap.add_argument("--detector", default="sift", choices=["sift", "akaze", "orb"])
    ap.add_argument("--map-max-side", type=int, default=6000, help="downscale map if larger")
    ap.add_argument("--frame-scale", default="auto",
                    help="'auto' (match map pixel size) or a fixed factor, e.g. 0.4")
    ap.add_argument("--ratio", type=float, default=0.75, help="Lowe ratio test threshold")
    ap.add_argument("--ransac-thr", type=float, default=5.0, help="reprojection threshold, px")
    ap.add_argument("--min-inliers", type=int, default=15)
    ap.add_argument("--video-out", action="store_true", help="also render route.mp4")
    ap.add_argument("--clahe", action="store_true",
                    help="local contrast equalisation (off by default: no gain on the given data)")
    ap.add_argument("--debug", action="store_true", help="save per-frame match visualizations")
    return ap.parse_args()


def save_csv(path, fixes, traj):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "frame", "status", "x_raw", "y_raw", "x", "y", "heading_deg",
                    "inliers", "matches", "rmse_px", "gsd_map_px_per_frame_px", "search"])
        for i, f in enumerate(fixes):
            status = ("rejected" if i in traj["rejected"] else
                      "interpolated" if i in traj["failed"] else "measured")
            w.writerow([i, f.name, status,
                        *(np.round(traj["raw"][i], 2) if f.ok else ["", ""]),
                        *np.round(traj["smooth"][i], 2), round(float(traj["heading"][i]), 1),
                        f.n_inliers, f.n_matches, round(f.rmse, 2) if f.ok else "",
                        round(f.gsd, 4) if f.ok else "", f.mode])


def debug_match(path, map_img, frame, fix):
    """Frame warped onto the map next to the map crop: quick visual check."""
    if not fix.ok:
        return
    x0, y0 = np.maximum(fix.corners.min(0).astype(int) - 20, 0)
    x1, y1 = fix.corners.max(0).astype(int) + 20
    crop = map_img[y0:y1, x0:x1]
    T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], float)
    warped = cv2.warpPerspective(frame, T @ fix.H, (crop.shape[1], crop.shape[0]))
    blend = cv2.addWeighted(crop, 0.5, warped, 0.5, 0)
    imwrite_any(path, np.hstack([crop, warped, blend]))


def main():
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    map_img = imread_any(a.map)

    paths = list_frames(a.frames)
    frames = [imread_any(p) for p in paths]
    names = [os.path.basename(p) for p in paths]
    print(f"[data] map {map_img.shape[1]}x{map_img.shape[0]}, {len(frames)} frames")

    loc = Localizer(map_img, detector=a.detector, map_max_side=a.map_max_side,
                    frame_scale=a.frame_scale if a.frame_scale == "auto" else float(a.frame_scale),
                    ratio=a.ratio, ransac_thr=a.ransac_thr, min_inliers=a.min_inliers,
                    clahe=a.clahe)
    fixes = [loc.localize(i, n, f) for i, (n, f) in enumerate(zip(names, frames))]

    # second pass for failed frames: now the scale and neighbours are known,
    # so a local search between two good neighbours often succeeds
    for i, f in enumerate(fixes):
        if f.ok:
            continue
        nb = [fixes[j].center for j in (i - 1, i + 1) if 0 <= j < len(fixes) and fixes[j].ok]
        if nb:
            print(f"[retry] frame {i} around its neighbours")
            retry = loc.localize(i, names[i], frames[i], prior=np.mean(nb, axis=0),
                                 update_state=False)
            if retry.ok:
                fixes[i] = retry

    traj = build_trajectory(fixes)
    save_csv(os.path.join(a.out, "trajectory.csv"), fixes, traj)
    route = draw_route(map_img, fixes, traj)
    imwrite_any(os.path.join(a.out, "route.png"), route)
    print(f"[out] {os.path.join(a.out, 'route.png')}")

    if a.debug:
        os.makedirs(os.path.join(a.out, "matches"), exist_ok=True)
        for f, fr in zip(fixes, frames):
            debug_match(os.path.join(a.out, "matches", f"{f.index:03d}.jpg"), map_img, fr, f)
    if a.video_out:
        render_video(os.path.join(a.out, "route.mp4"), map_img, frames, fixes, traj)
        print(f"[out] {os.path.join(a.out, 'route.mp4')}")

    ok = sum(f.ok for f in fixes)
    print(f"[done] localized {ok}/{len(fixes)} frames, rejected {len(traj['rejected'])}, "
          f"time {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

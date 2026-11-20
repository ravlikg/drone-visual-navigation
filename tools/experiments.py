"""
Ablation experiments used to justify design choices (see docs/).

Compares detectors (SIFT / AKAZE / ORB — lecture slides 10-11, 23),
frame downscaling (task note "experiment with image downscaling"),
local search window vs global search, and the ratio-test threshold
(lecture slide 12: 0.7–0.8).

Reference = SIFT, auto scale, local search. Position error of other runs is
measured against it (no ground truth is given in the task).

    python tools/experiments.py --map data/global_map.png --frames data/crops
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drone_nav.io_utils import imread_any, list_frames  # noqa: E402
from drone_nav.localize import Localizer  # noqa: E402

CONFIGS = [
    # name,                       kwargs
    ("SIFT auto-scale local (ref)", dict(detector="sift")),
    ("SIFT auto-scale GLOBAL only", dict(detector="sift", search_radius_factor=0)),
    ("SIFT scale 1.0 (no downscale)", dict(detector="sift", frame_scale=1.0)),
    ("SIFT scale 0.5", dict(detector="sift", frame_scale=0.5)),
    ("SIFT scale 0.25", dict(detector="sift", frame_scale=0.25)),
    ("SIFT scale 0.15", dict(detector="sift", frame_scale=0.15)),
    ("SIFT ratio 0.7", dict(detector="sift", ratio=0.7)),
    ("SIFT ratio 0.8", dict(detector="sift", ratio=0.8)),
    ("SIFT + CLAHE", dict(detector="sift", clahe=True)),
    ("AKAZE auto-scale", dict(detector="akaze")),
    ("ORB auto-scale", dict(detector="orb")),
    ("ORB scale 1.0", dict(detector="orb", frame_scale=1.0)),
]


def run(map_img, frames, names, kw):
    t = time.time()
    loc = Localizer(map_img, verbose=False, **kw)
    t_map = time.time() - t
    t = time.time()
    fixes = [loc.localize(i, n, f) for i, (n, f) in enumerate(zip(names, frames))]
    return fixes, t_map, (time.time() - t) / len(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", default="outputs/experiments.md")
    a = ap.parse_args()
    map_img = imread_any(a.map)
    paths = list_frames(a.frames)
    frames = [imread_any(p) for p in paths]
    names = [os.path.basename(p) for p in paths]

    rows, ref = [], None
    for name, kw in CONFIGS:
        try:
            fixes, t_map, t_fr = run(map_img, frames, names, kw)
        except RuntimeError as e:
            print(f"{name:<32} skipped: {e}")
            continue
        xy = np.array([f.center if f.ok else [np.nan, np.nan] for f in fixes])
        if ref is None:
            ref = xy
        err = np.linalg.norm(xy - ref, axis=1)
        good = err < 10  # within 10 map px of the reference = "correct"
        ok = sum(f.ok for f in fixes)
        inl = np.mean([f.n_inliers for f in fixes if f.ok]) if ok else 0
        rmse = np.mean([f.rmse for f in fixes if f.ok]) if ok else np.nan
        rows.append((name, ok, int(good.sum()), np.nanmedian(err) if ok else np.nan,
                     inl, rmse, t_map, t_fr * 1000))
        print(f"{name:<32} ok={ok:2d} correct={good.sum():2d} inl={inl:6.1f} "
              f"rmse={rmse:4.2f} map={t_map:5.1f}s frame={t_fr * 1000:6.0f}ms", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write("| Config | Localized /50 | Correct (<10 px from ref) | Median dev, px | "
                 "Mean inliers | Mean RMSE, px | Map features, s | Per frame, ms |\n")
        fh.write("|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            fh.write(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]:.2f} | {r[4]:.0f} | {r[5]:.2f} | "
                     f"{r[6]:.1f} | {r[7]:.0f} |\n")
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()

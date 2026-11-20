"""
Robustness test of Stage 5: pretend that some frames failed to localize and
check how well the Kalman/RTS interpolation recovers them, plus how far the
smoothed track deviates from measured positions.

    python tools/gap_test.py --map data/global_map.png --frames data/crops
"""
import argparse
import copy
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drone_nav.io_utils import imread_any, list_frames  # noqa: E402
from drone_nav.localize import Localizer  # noqa: E402
from drone_nav.trajectory import build_trajectory  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--drop", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    paths = list_frames(a.frames)
    loc = Localizer(imread_any(a.map), verbose=False)
    fixes = [loc.localize(i, os.path.basename(p), imread_any(p)) for i, p in enumerate(paths)]
    ref = np.array([f.center for f in fixes])

    full = build_trajectory(fixes)
    dev = np.linalg.norm(full["smooth"] - ref, axis=1)
    print(f"smoothed vs measured (no gaps): median {np.median(dev):.2f}, max {dev.max():.2f} px")

    rng = np.random.default_rng(a.seed)
    drop = np.sort(rng.choice(np.arange(1, len(fixes) - 1), a.drop, replace=False))
    broken = copy.deepcopy(fixes)
    for i in drop:
        broken[i].ok = False
    # also inject one gross outlier (a wrong match far away)
    bad = int(rng.choice([i for i in range(2, len(fixes) - 2) if i not in drop]))
    broken[bad].center = broken[bad].center + np.array([400.0, -300.0])
    tr = build_trajectory(broken)
    err = np.linalg.norm(tr["smooth"][drop] - ref[drop], axis=1)
    print(f"dropped frames {drop.tolist()}")
    print("per dropped frame, px:", dict(zip(drop.tolist(), np.round(err, 1).tolist())))
    print(f"interpolation error: median {np.median(err):.2f}, max {err.max():.2f} px")
    print(f"injected outlier at {bad}: rejected={bad in tr['rejected']}, "
          f"error after recovery {np.linalg.norm(tr['smooth'][bad] - ref[bad]):.2f} px")
    step = np.median(np.linalg.norm(np.diff(ref, axis=0), axis=1))
    print(f"(for scale: median step between frames = {step:.1f} px)")


if __name__ == "__main__":
    main()

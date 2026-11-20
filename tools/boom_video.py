"""
Bonus video: the normal fly-along route video, and at the FINISH point the
drone "hits the target" with a cartoon explosion (procedural VFX).

Reuses the route video produced by main.py (--video-out) and appends an
explosion sequence rendered over its last frame.

    python tools/boom_video.py --video outputs/route.mp4 --out outputs/route_boom.mp4
"""
import argparse

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def find_drone(frame_bgr, view=900):
    """Locate the red drone triangle in the map window of the last frame."""
    win = frame_bgr[:, :view]
    hsv = cv2.cvtColor(win, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (0, 170, 170), (8, 255, 255)) | cv2.inRange(hsv, (172, 170, 170), (180, 255, 255))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return np.array([view / 2, view / 2])
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return cen[k]


def noise_field(h, w, rng, scale=24):
    """Smooth random field in [0,1] (for ragged fireball edges and smoke)."""
    small = rng.random((h // scale + 2, w // scale + 2)).astype(np.float32)
    f = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    return (f - f.min()) / (np.ptp(f) + 1e-6)


def text(img, s, xy, size, color, stroke=4):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    f = ImageFont.truetype(FONT, size)
    w = d.textlength(s, font=f)
    d.text((xy[0] - w / 2, xy[1]), s, font=f, fill=color[::-1], stroke_width=stroke, stroke_fill=(0, 0, 0))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def explosion_frames(base, center, n=130, seed=3, view=900):
    rng = np.random.default_rng(seed)
    H, W = base.shape[:2]
    # remove the drone marker from the scene (it is gone after impact)
    hsv = cv2.cvtColor(base, cv2.COLOR_BGR2HSV)
    m = (cv2.inRange(hsv, (0, 170, 170), (8, 255, 255)) | cv2.inRange(hsv, (172, 170, 170), (180, 255, 255)))
    m[:, view:] = 0
    m = cv2.dilate(m, np.ones((9, 9), np.uint8))
    after = cv2.inpaint(base, m, 7, cv2.INPAINT_TELEA)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dist = np.hypot(xx - center[0], yy - center[1])
    ang = np.arctan2(yy - center[1], xx - center[0])
    edge = noise_field(H, W, rng, 14)
    edge2 = noise_field(H, W, rng, 45)
    smoke_n = noise_field(H, W, rng, 30)
    # debris particles
    P = 160
    pv = rng.normal(0, 1, (P, 2)); pv /= np.linalg.norm(pv, axis=1, keepdims=True)
    pv *= rng.uniform(5, 24, (P, 1))
    psize = rng.integers(2, 6, P)
    # embers that keep glowing in the crater
    E = 90
    er = 170 * np.sqrt(rng.random(E)); ea = rng.uniform(0, 2 * np.pi, E)
    epos = center + np.c_[er * np.cos(ea), er * np.sin(ea)]
    ephase = rng.uniform(0, 2 * np.pi, E)
    out = []
    for t in range(n):
        img = (base if t < 2 else after).astype(np.float32)
        # scorched ground + crater
        r_sc = 280 * min(1, t / 20)
        scorch = np.clip(1.15 - dist / (r_sc + 1) * (0.75 + 0.5 * edge2), 0, 1)
        crater = np.clip(1.3 - dist / (70 * min(1, t / 10) + 1), 0, 1)
        img *= (1 - 0.8 * scorch[..., None])
        img *= (1 - 0.9 * crater[..., None])
        # glowing embers
        if t > 14:
            for k in range(E):
                b = 0.5 + 0.5 * np.sin(0.4 * t + ephase[k])
                cv2.circle(img, (int(epos[k, 0]), int(epos[k, 1])), 3,
                           (20, 90 + 100 * b, 230 + 25 * b), -1, cv2.LINE_AA)
        # fireball: fast growth, then fades into the smoke
        rf = 330 * (1 - (1 - min(1, t / 16)) ** 3)
        fade = 1 - np.clip((t - 20) / 45, 0, 1)
        if fade > 0:
            rr = rf * (0.75 + 0.45 * edge + 0.06 * np.sin(9 * ang + t * 0.4))
            core = np.clip(1 - dist / (rr + 1), 0, 1)
            h = core ** 0.8
            fire = np.stack([10 + 230 * h ** 4, 40 + 215 * h ** 1.6, np.full_like(h, 255)], -1)  # BGR
            a = np.clip(core * 3.0, 0, 1)[..., None] * fade
            img = img * (1 - a) + fire * a
        # thick black smoke: grows, rises, stays till the end
        if t > 6:
            k = t - 6
            rs = 140 + 300 * min(1, k / 55)
            ds = np.hypot(xx - center[0], (yy - center[1] + 1.2 * k) * 1.1)
            sm = np.clip(1.1 - ds / rs * (0.7 + 0.6 * smoke_n), 0, 1)
            sm *= min(1, k / 12) * (1 - 0.45 * max(0, (t - 80) / 50))
            sm *= 0.92
            img = img * (1 - sm[..., None]) + 25 * sm[..., None]
        # shockwave ring
        rs_w = 40 + 34 * t
        if rs_w < 1200:
            ring = np.exp(-((dist - rs_w) / 12) ** 2) * (1 - rs_w / 1200)
            img += 170 * ring[..., None]
        # debris
        for p in range(P):
            pos = center + pv[p] * t + np.array([0, 0.2 * t * t])
            if t < 70:
                cv2.circle(img, (int(pos[0]), int(pos[1])), int(psize[p]),
                           (15, 20, 25) if p % 3 else (20, 140, 255), -1)
        # white flash
        if t < 7:
            img = img * (1 - (1 - t / 7) * 0.95) + 255 * (1 - t / 7) * 0.95
        img = np.clip(img, 0, 255).astype(np.uint8)
        img[:, view:] = base[:, view:]          # keep the side panel untouched
        # camera shake
        amp = 26 * max(0, 1 - t / 32)
        if amp > 0:
            M = np.float32([[1, 0, rng.normal(0, amp)], [0, 1, rng.normal(0, amp)]])
            img[:, :view] = cv2.warpAffine(img[:, :view], M, (view, H), borderMode=cv2.BORDER_REFLECT)
        if t > 26:
            alpha = min(1, (t - 26) / 8)
            img = text(img, "БАБАХ!", (view / 2, 50), int(130 * (0.5 + 0.5 * alpha)), (40, 200, 255), 7)
        if t > 60:
            img = text(img, "Ціль уражено. Місію виконано.", (view / 2, H - 110), 44, (255, 255, 255), 4)
        out.append(img)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="outputs/route.mp4")
    ap.add_argument("--out", default="outputs/route_boom.mp4")
    a = ap.parse_args()
    r = imageio.get_reader(a.video)
    fps = r.get_meta_data()["fps"]
    w = imageio.get_writer(a.out, fps=fps, codec="libx264", quality=7,
                           ffmpeg_params=["-preset", "veryfast"], macro_block_size=1)
    last = None
    for fr in r:
        w.append_data(fr); last = fr
    base = cv2.cvtColor(last, cv2.COLOR_RGB2BGR)
    c = find_drone(base)
    for _ in range(int(fps * 0.6)):          # short pause: "target acquired"
        w.append_data(last)
    for f in explosion_frames(base, c):
        w.append_data(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    for _ in range(int(fps * 1.5)):          # hold the final frame
        w.append_data(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    w.close()
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()

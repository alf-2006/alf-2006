#!/usr/bin/env python3
"""Pointillist dot-art portrait from a photo -> PNG (README) + animated SVG + packed JSON (Pages demo).

Layouts produced (all from the same sampled points):
  1. lattice    - hex-offset grid, radius from local luminance (classic halftone)
  2. edge       - dots biased onto Sobel edges, so facial features are traced
  3. contour    - dots on brightness isolines (topographic look)
  4. neural     - dots + connection lines to nearest neighbours (mesh/graph look)
"""
import json
import os
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

import argparse
p = argparse.ArgumentParser(); p.add_argument("src"); p.add_argument("--out", default="."); a_ = p.parse_args()
SRC = a_.src
OUT = a_.out
os.makedirs(OUT, exist_ok=True)

BG = (13, 17, 23)          # #0D1117 - matches the repo theme
ACCENT = (0, 240, 255)     # cyan used across the profile

# ---------------------------------------------------------------- load / crop
src_im = Image.open(SRC)
matted = "A" in src_im.getbands()
if matted:
    # Stage 0 output (see matting.py): composite the subject onto the canvas
    # colour so removed background samples as flat #0D1117 instead of garbage.
    base = Image.new("RGBA", src_im.size, BG + (255,))
    im = Image.alpha_composite(base, src_im.convert("RGBA")).convert("RGB")
else:
    im = src_im.convert("RGB")

# The screenshot has uniform green letterbox bars. Rather than guessing, find the
# tightest content box by rejecting rows/cols that are near-flat in both colour and
# local contrast, then keep the largest contiguous block (tolerates the watermark row).
a0 = np.asarray(im).astype(np.float32) / 255.0
H0, W0 = a0.shape[:2]
lum = a0 @ np.array([0.2126, 0.7152, 0.0722])

# Letterbox bounds verified by inspecting the source screenshot:
# dark green bar until y~277, photo content 278..1003, bar + watermark below.
if matted or H0 < 1004:
    x0, x1, y0, y1 = 0, W0 - 1, 0, H0 - 1        # already cropped by matting.py
else:
    x0, x1, y0, y1 = 0, W0 - 1, 278, 1003
print(f"content box x:{x0}-{x1} y:{y0}-{y1}")

a = a0[y0:y1 + 1, x0:x1 + 1]
im = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))

# --------------------------------------------------------------- tone mapping
im = ImageEnhance.Contrast(im).enhance(1.12)
im = ImageEnhance.Color(im).enhance(1.18)
im = im.filter(ImageFilter.UnsharpMask(radius=2.4, percent=120, threshold=2))
arr = np.asarray(im).astype(np.float32) / 255.0

# Gentle shadow lift so the black suit keeps structure on the near-black canvas.
arr = np.clip(arr ** 0.90 * 1.05 + 0.025, 0.0, 1.0)

H, W = arr.shape[:2]
lum = arr @ np.array([0.2126, 0.7152, 0.0722])

# Sobel edge magnitude for feature weighting
gx = np.zeros_like(lum); gy = np.zeros_like(lum)
kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32) / 4.0
ky = kx.T
def shift(m, dy, dx):
    out = np.zeros_like(m)
    ys0, ys1 = max(0, dy), min(H, H + dy)
    xs0, xs1 = max(0, dx), min(W, W + dx)
    out[ys0:ys1, xs0:xs1] = m[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
    return out

gx[:] = 0.0; gy[:] = 0.0
for dy in (-1, 0, 1):
    for dx in (-1, 0, 1):
        gx += kx[1 + dy, 1 + dx] * shift(lum, dy, dx)
        gy += ky[1 + dy, 1 + dx] * shift(lum, dy, dx)
edge = np.hypot(gx, gy)
edge = edge / max(edge.max(), 1e-6)
edge = edge ** 1.3

CELL = 6.4
PITCH_X, PITCH_Y = CELL, CELL * 0.88

def sample(points):
    """points: (N,2) float px in canvas space -> (colors uint8 hex, radius)"""
    if len(points) == 0:
        return np.empty((0, 6), np.uint8), np.empty(0, np.float32)
    ix = np.clip(points[:, 0].astype(np.int32), 0, W - 1)
    iy = np.clip(points[:, 1].astype(np.int32), 0, H - 1)
    # 2x2 area average
    x1 = np.clip(ix + 1, 0, W - 1); y1 = np.clip(iy + 1, 0, H - 1)
    c = (arr[iy, ix] + arr[iy, x1] + arr[y1, ix] + arr[y1, x1]) * 0.25
    l = (lum[iy, ix] + lum[iy, x1] + lum[y1, ix] + lum[y1, x1]) * 0.25
    e = (edge[iy, ix] + edge[iy, x1] + edge[y1, ix] + edge[y1, x1]) * 0.25
    return (np.clip(c, 0, 1) * 255).astype(np.uint8), l, e

rng = np.random.default_rng(20060)

CW, CH = int(W), int(H)

# ---- layout 1: hex lattice, radius from luminance ----
rows = int(CH // PITCH_Y)
cols = int(CW // CELL)
pts = []
for r in range(rows):
    off = (CELL * 0.5) if (r % 2) else 0.0
    for c in range(cols):
        pts.append((c * CELL + CELL * 0.5 + off, r * PITCH_Y + PITCH_Y * 0.5))
pts = np.array(pts, np.float32)
col1, l1, _ = sample(pts)
rad1 = CELL * (0.06 + 0.62 * l1 ** 0.90)   # dark canvas: light IS the ink

# ---- layout 2: edge-weighted scatter (features traced) ----
N_EDGE = 5200
# importance map: edges + a floor so shading is still present
w2 = (edge + 0.11) ** 1.9
w2 = w2 / w2.sum()
idx2 = rng.choice(H * W, size=N_EDGE, p=w2.ravel())
y2i, x2i = np.unravel_index(idx2, (H, W))
pts2 = np.stack([x2i + rng.uniform(-0.4, 0.4, N_EDGE),
                 y2i + rng.uniform(-0.4, 0.4, N_EDGE)], axis=1).astype(np.float32)
col2, l2, e2 = sample(pts2)
rad2 = CELL * (0.14 + 0.32 * e2 + 0.22 * (1.0 - l2))

# ---- layout 3: contour bands (isolines of brightness) ----
lv = np.clip((lum * 13).astype(np.int32), 0, 12)
boundary = np.zeros_like(lum, bool)
boundary[:, :-1] |= np.abs(np.diff(lv, axis=1)) > 0
boundary[:-1, :] |= np.abs(np.diff(lv, axis=0)) > 0
by, bx = np.nonzero(boundary)
N_CON = 6400
keep = rng.choice(len(by), size=N_CON, replace=False)
pts3 = np.stack([bx[keep] + rng.uniform(-0.6, 0.6, N_CON),
                 by[keep] + rng.uniform(-0.6, 0.6, N_CON)], axis=1).astype(np.float32)
col3, l3, e3 = sample(pts3)
rad3 = CELL * (0.18 + 0.34 * (1.0 - l3))

# ---- layout 4: neural mesh (edge points + k-NN links) ----
mesh_idx = rng.choice(H * W, size=1600, p=w2.ravel())
my, mx = np.unravel_index(mesh_idx, (H, W))
pts4 = np.stack([mx + rng.uniform(-0.4, 0.4, 1600), my + rng.uniform(-0.4, 0.4, 1600)],
                axis=1).astype(np.float32)
k = 4
d = np.sqrt(((pts4[:, None, :] - pts4[None, :, :]) ** 2).sum(-1))
np.fill_diagonal(d, np.inf)
link = np.argmin(d, axis=1)
edges_pairs = [(i, link[i]) for i in range(len(pts4)) if d[i, link[i]] < 46]

LAYOUTS = []
def pack(p, c, r):
    return {
        "x": np.round(p[:, 0], 1).tolist(),
        "y": np.round(p[:, 1], 1).tolist(),
        "r": np.round(r, 2).tolist(),
        "c": [f"{v[0]:02x}{v[1]:02x}{v[2]:02x}" for v in c],
    }

c4, l4, e4 = sample(pts4)
rad4 = CELL * (0.18 + 0.30 * e4 + 0.22 * (1.0 - l4))
LAYOUTS = {
    "lattice": pack(pts, col1, rad1),
    "edge": pack(pts2, col2, rad2),
    "contour": pack(pts3, col3, rad3),
    "mesh": pack(pts4, c4, rad4),
}

payload = {
    "w": CW, "h": CH, "cell": CELL,
    "bg": "%02x%02x%02x" % BG, "accent": "%02x%02x%02x" % ACCENT,
    "links": [[int(a_), int(b_)] for a_, b_ in edges_pairs],
    "layouts": LAYOUTS,
}
json.dump(payload, open(f"{OUT}/dots.json", "w"), separators=(",", ":"))
sz = os.path.getsize(f"{OUT}/dots.json")
print(f"dots.json {sz/1024:.0f} KiB  lattice={len(pts)} edge={len(pts2)} contour={len(pts3)} mesh={len(pts4)} links={len(edges_pairs)}")

# ---------------------------------------------------------------- static PNG
def render(p, c, r, ss=3, glow=None):
    img = Image.new("RGB", (CW * ss, CH * ss), BG)
    dr = ImageDraw.Draw(img)
    hexes = [f"{v[0]:02x}{v[1]:02x}{v[2]:02x}" for v in c]
    for (x, y), rr, h in zip(p, r, hexes):
        col = (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
        dr.ellipse([(x - rr) * ss, (y - rr) * ss, (x + rr) * ss, (y + rr) * ss], fill=col)
    img = img.resize((CW, CH), Image.LANCZOS)
    if glow:
        halo = Image.new("RGB", (CW, CH), BG)
        hd = ImageDraw.Draw(halo)
        for (x, y), rr, h in zip(p, r, hexes):
            if rr > np.quantile(r, 0.93):
                hd.ellipse([x - rr, y - rr, x + rr, y + rr],
                           fill=(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)))
        halo = halo.filter(ImageFilter.GaussianBlur(9))
        img = Image.blend(img, halo, 0.22)
    return img

hero = render(pts, col1, rad1)
# head + shoulders composition (the matte frame is already tighter than the raw screenshot)
box = (int(W * 0.16), int(H * 0.20), int(W * 0.84), int(H * 0.97)) if matted \
    else (int(W * 0.10), int(H * 0.22), int(W * 0.92), int(H * 0.92))
hero.crop(box).save(f"{OUT}/portrait-hero.png", optimize=True)
hero.save(f"{OUT}/portrait-full.png", optimize=True)
render(pts2, col2, rad2, glow=True).crop(box).save(f"{OUT}/portrait-edge.png", optimize=True)
print("png ok")

# ------------------------------------------------------------- animated SVG
def svg_animated(p, c, r, w, h, step=9):
    n = len(p)
    order = np.argsort(p[:, 1] * 0.55 + p[:, 0] * 0.45 + rng.normal(0, 60, n))
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">',
             f'<rect x="0" y="0" width="{w}" height="{h}" fill="#{payload["bg"]}"/>',
             '<g>']
    hexes = [f"{v[0]:02x}{v[1]:02x}{v[2]:02x}" for v in c]
    for k, i in enumerate(order):
        x, y, rr = p[i]
        t = k * step
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{rr:.2f}" fill="#{hexes[i]}" opacity="0">'
            f'<set attributeName="opacity" to="1" begin="{t}ms" dur="indefinite"/></circle>')
    parts.append(f'<animate attributeName="opacity" values="0;1" dur="{n*step+400}ms" fill="freeze"/>')
    parts.append('</g></svg>')
    return "".join(parts)

# static SVG (no script/SMIL timing so it also works as a plain image file)
def svg_static(p, c, r, w, h):
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">',
             f'<rect x="0" y="0" width="{w}" height="{h}" fill="#{payload["bg"]}"/>']
    hexes = [f"{v[0]:2x}" for v in c]
    for (x, y), rr, h6 in zip(p, r, [f"{v[0]:02x}{v[1]:02x}{v[2]:02x}" for v in c]):
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{rr:.2f}" fill="#{h6}"/>')
    parts.append('</svg>')
    return "".join(parts)

svg = svg_static(pts, col1, rad1, CW, CH)
open(f"{OUT}/portrait-dots.svg", "w").write(svg)
print(f"svg {os.path.getsize(f'{OUT}/portrait-dots.svg')/1024:.0f} KiB")

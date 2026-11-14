# 🧬 Generative Portrait Engine

Everything in `../assets` and `../docs` is computed from a single photo — no manual drawing, no filters.

## Pipeline

| Stage | What happens |
|---|---|
| 1. Crop | Letterbox bars detected from row/column variance, tightest content block kept |
| 2. Tone | Contrast + saturation pass, unsharp mask, shadow lift so the black suit survives the `#0D1117` canvas |
| 3. Field | Sobel gradient magnitude → edge importance map; brightness isolines → contour map |
| 4. Sampling | 4 layouts from the same image: **hex lattice** (14,336 dots, radius = luminance), **edge tracing** (5,200 dots weighted onto gradients), **contour isolines** (6,400 dots), **neural mesh** (1,600 dots + k-NN links) |
| 5. Output | `portrait-dots.svg` (1 circle per dot), `dots.json` (packed arrays for the browser), PNG renders at 3× supersampling |

## Run it yourself

```bash
pip install pillow numpy
python dotty.py /path/to/photo.jpg --out ../assets
```

Re-crop for a different input by editing `x0, x1, y0, y1` near the top of the script.

The web page (`../docs/index.html`) reads `dots.json`, springs 27,536 particles between the four layouts on a canvas, and lets the cursor repel them.

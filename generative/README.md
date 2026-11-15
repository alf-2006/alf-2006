# 🧬 Generative Portrait Engine

Everything in `../assets` and `../docs` is computed from a single photo — no
manual drawing, no filters. Three scripts, one learned model, ~35k dots.

## Pipeline

| Stage | Script | What happens |
|---|---|---|
| 0. Segmentation | `matting.py` | `rembg · isnet-general-use` (a saliency/matting CNN) cuts the subject out of the cluttered mural background. This alone is the difference between "static noise" and "that's clearly his face" — the mural contributed ~40 % of dot density in early renders. |
| 1. Dot field | `dotty.py` | Composite matte onto `#0D1117` → tone curve → Sobel edge field → 4 layouts from the same image: **hex lattice** (14,224 dots, radius = luminance), **edge tracing** (5,200 dots weighted onto gradients), **contour isolines** (6,400), **neural mesh** (1,600 + k-NN links). Outputs `portrait-dots.svg`, `dots.json`, PNGs at 3× supersampling. |
| 2. Terminal | `dotty_terminal.py` | Neofetch-style window whose *logo slot* is the matte rendered as ~7.5k real-color halftone dots (brightness drives radius — light is ink on a dark canvas). One file, pure SMIL: command types itself, portrait paints in 24 bands top-down, stats print, cursor blinks, 16 s loop. |

## Run it yourself

```bash
pip install pillow numpy "rembg[cpu]"

python matting.py /path/to/photo.jpg --out ../assets      # → portrait-matted.png (RGBA)
python dotty.py ../assets/portrait-matted.png --out ../assets
python dotty_terminal.py ../assets/portrait-matted.png --out ../assets
cp ../assets/dots.json ../docs/dots.json                  # powers the Pages demo
```

Raw photos work too — `dotty.py` skips letterbox detection automatically when
given the RGBA matte; on an RGB screenshot it crops the phone bars first.
Tweak the crop boxes (`box`, `bx0..by1`) for a different framing.

The web page (`../docs/index.html`) reads `dots.json`, springs 27,424
particles between the four layouts on a canvas, and lets the cursor repel them.

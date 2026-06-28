# Vera — GPU-native Astronomical Source Extraction

A Rust pipeline for detecting and measuring astronomical sources (stars, galaxies) in survey images, inspired by [SExtractor](https://sextractor.readthedocs.io/) but designed from the ground up for GPU acceleration and modern hardware.

Built on public DECam data from the [DESI Legacy Survey DR10](https://www.legacysurvey.org/dr10/), targeting the Virgo Cluster field (RA ≈ 187.7°, Dec ≈ +12.4°) — the same sky region imaged in the Vera C. Rubin Observatory first-light image ([noirlab2521a](https://noirlab.edu/public/images/noirlab2521a/)).

> **Author:** Hugues LAMBERT  
> **Developed with:** [Claude Sonnet](https://anthropic.com) (Anthropic) — AI-assisted development under the direction of Hugues LAMBERT  
> **Target publication:** [Journal of Open Source Software (JOSS)](https://joss.theoj.org)

---

## Motivation

SExtractor is a C program from 1996. Rubin Observatory pipelines run on HPC clusters. This project bridges the gap: a scientifically rigorous source extraction pipeline that runs on consumer hardware (tested on RTX 4070 Ti, i9-10850K, 64 GB RAM), written in safe Rust with portable GPU compute via `wgpu`.

When Rubin DP1 public data becomes available, the only change needed is the input file path — the FITS format and calibration convention are identical to Legacy Survey DR10.

---

## Progress

### ✅ Phase 1 — FITS I/O (`vera-fits`)

- Reads plain FITS and RICE tile-compressed `.fits.fz` transparently
- Extracts WCS header (CD matrix, CRPIX, CRVAL) for pixel ↔ sky conversion
- Handles all pixel types (F32, F64, I16, I32, I64, U8)
- `vera-inspect` binary: prints image dimensions, flux statistics, WCS info

```
$ vera-inspect legacysurvey-1877p122-image-r.fits.fz

┌── Image ──────────────────────────────────────
│  Taille   : 3600 x 3600 px  (12 Mpx)
│  Flux     : min=-0.2031  max=62.5469  mean=0.0220  (nanomaggies)
│  WCS centre : RA=187.7983°  Dec=12.2500°
│  FOV        : 0.262° x 0.262°
│  Plate scale: 0.2620 arcsec/px
└───────────────────────────────────────────────
```

### ✅ Phase 2 — Sky Background Estimation (`vera-pipeline`)

SExtractor-equivalent background algorithm:

1. Divide image into a mesh of cells (default 64×64 px)
2. Each cell: iterative κ-σ clipping (MAD-based, Gaussian-unbiased)
3. Mode estimate following SExtractor: `bg = 2.5·median − 1.5·mean` when skewed
4. Bilinear interpolation of the low-resolution grid to full image size
5. Background subtraction → residual image

All cell estimations run in parallel via Rayon.

```
$ vera-background legacysurvey-1877p122-image-r.fits.fz

BG estimation : 12.8ms  (grille 57 x 57 cellules de 64px)

┌── Fond de ciel ───────────────────────────────
│  BG  : min=-0.025  max=0.203  mean=0.004  (nanomaggies)
│  RMS : min=0.004   max=2.510  mean=0.092
│  Residual median=0.003  RMS=0.329
└───────────────────────────────────────────────
```

Background estimation over a 3600×3600 brick: **~13 ms** (release build).

### 🔲 Phase 3 — Source Detection (`vera-gpu` + `vera-pipeline`)

- [ ] Gaussian convolution matched to PSF (GPU, `wgpu` compute shader)
- [ ] Pixel thresholding: `flux > N × RMS_map` (GPU)
- [ ] Connected-component labeling (Union-Find, CPU)
- [ ] Basic deblending (threshold tree)
- [ ] `DetectionList` struct

### 🔲 Phase 4 — Photometric Measurements (`vera-pipeline`)

- [ ] Centroid (flux-weighted 1st moments)
- [ ] Isophotal flux + Kron elliptical aperture flux
- [ ] 2nd-order moments → semi-axes, position angle
- [ ] Star/galaxy classification
- [ ] Quality flags (saturation, edge, deblend)

### 🔲 Phase 5 — Catalog Output (`vera-catalog`)

- [ ] FITS binary table (standard column names)
- [ ] Parquet export
- [ ] Comparison with Tractor catalog from same brick

### 🔲 Phase 6 — Interactive Viewer (`vera-viewer`)

- [ ] GPU tile streaming (wgpu + winit)
- [ ] ZScale / asinh stretch
- [ ] Source ellipse overlays (click → info panel)
- [ ] WCS coordinate display
- [ ] 60 fps pan/zoom

### 🔲 Phase 7 — Multi-brick Pipeline

- [ ] CLI `vera run --data-dir ./fits --output catalogue.fits`
- [ ] Cross-brick duplicate resolution (< 1 arcsec matching)
- [ ] Full Virgo Cluster mosaic from 28 DR10 bricks

---

## Data

28 DECam bricks (r-band) from DESI Legacy Survey DR10, centered on M87 / Virgo Cluster core.

Each brick: 3600×3600 px · 0.262 arcsec/px · calibrated in nanomaggies.

Files (not included in repo — ~1 GB total):

```
fits/legacysurvey-<brick>-image-r.fits.fz      # science image (RICE-compressed)
fits/legacysurvey-<brick>-invvar-r.fits.fz     # inverse variance
fits/legacysurvey-<brick>-maskbits.fits.fz     # bitmask
```

Download with the script in `scripts/download_bricks.ps1` (TODO).

---

## Build

```bash
cargo build --release -p vera-fits
cargo build --release -p vera-pipeline
```

Requires: Rust 2024 edition (≥ 1.85).

No C dependencies — pure Rust, including FITS RICE decompression via [fitsrs](https://crates.io/crates/fitsrs).

---

## Workspace Structure

```
vera/
├── crates/
│   ├── vera-fits/        # FITS I/O + WCS (no GPU)
│   ├── vera-pipeline/    # background, detection, measurement algorithms
│   ├── vera-gpu/         # wgpu compute shaders (Phase 3+)
│   ├── vera-catalog/     # catalog serialization (Phase 5)
│   └── vera-viewer/      # interactive viewer (Phase 6)
└── fits/                 # data directory (gitignored)
```

---

## Compatibility

Legacy Survey DR10 and Rubin DP1 share the same FITS format and calibration convention (nanomaggies, RICE compression, standard WCS). Switching to Rubin data requires only changing the input path.

---

## Attribution

This project was developed by **Hugues LAMBERT** with the assistance of **Claude Sonnet** (Anthropic) as an AI pair-programming tool. All scientific decisions, design choices, and project direction are by Hugues LAMBERT.

Any publication derived from this work (including the planned JOSS paper) will explicitly credit:
- Hugues LAMBERT — author and scientific director
- Claude Sonnet (Anthropic) — AI-assisted development

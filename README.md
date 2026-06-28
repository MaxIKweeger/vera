# Vera — GPU-native Astronomical Source Extraction

A complete Rust pipeline for detecting and measuring astronomical sources (stars, galaxies)
in survey images.  Inspired by [SExtractor](https://sextractor.readthedocs.io/) but designed
for modern hardware: parallel CPU via Rayon, GPU rendering via wgpu/egui, pure-Rust FITS I/O
with no C dependencies.

Validated on 28 DECam bricks from [DESI Legacy Survey DR10](https://www.legacysurvey.org/dr10/)
around the Virgo Cluster core (M87 / RA ≈ 187.7°, Dec ≈ +12.4°) — the same field imaged in
the Vera C. Rubin Observatory first-light image
([noirlab2521a](https://noirlab.edu/public/images/noirlab2521a/)).

> **Author:** MaxKweeger  
> **Developed with:** [Claude Sonnet](https://anthropic.com) (Anthropic) — AI-assisted  
> **Target publication:** [Journal of Open Source Software (JOSS)](https://joss.theoj.org)

---

## Pipeline overview

```
FITS image (.fits.fz)
    │
    ▼
vera-fits       — FITS + RICE decompression, WCS extraction
    │
    ▼
BackgroundMap   — 64-px mesh · κ-σ clipping · bilinear interpolation
    │
    ▼
detect()        — Gaussian PSF match · SNR threshold · Union-Find labeling
    │
    ▼
measure_all()   — centroid · Kron flux · 2nd moments · WCS coords
    │
    ▼
vera-catalog    — FITS binary table + CSV export
    │
    ▼
vera-viewer     — interactive GPU viewer (eframe 0.35 / egui 0.35)
```

For multi-brick processing, `vera-run` runs all steps in parallel across bricks (Rayon)
and cross-matches sources at brick boundaries to remove duplicates.

---

## Performance — Virgo Cluster field, 28 bricks, band r

Measured on **Intel i9-10850K (10c / 20t) · RTX 4070 Ti · 64 GB RAM · Windows 11**.
Release build (`cargo build --release`).  Each single-brick time is the median of 3 runs.
GPU temperature monitored throughout via `nvidia-smi` — flat at **32 °C** (idle):
the pipeline is CPU-only; the GPU is used exclusively by `vera-viewer` for rendering.

### Single brick — `legacysurvey-1877p122-image-r.fits.fz` (3 600 × 3 600 px, ~11 MB)

| Stage | Binary | Wall time |
|-------|--------|----------:|
| FITS read + RICE decompress | `vera-background` | ~190 ms |
| Background estimation (64 px mesh, κ-σ × 10, bilinear) | `vera-background` | **13 ms** |
| Gaussian convolution (σ = 1.6 px, rayon rows) | `vera-detect` | ~440 ms |
| SNR map + Union-Find 8-connectivity | `vera-detect` | ~210 ms |
| Photometric measurements (centroid, moments, Kron) | `vera-measure` | **230 ms** |
| **Full pipeline (I/O + BG + detect + measure)** | `vera-measure` | **~1.1 s** |
| + Catalog write (FITS binary table + CSV) | `vera-catalog` | **~1.1 s** |
| Sources detected | — | **3 576** |

### Scale-out — 28 bricks in parallel (vera-run)

| Metric | Value |
|--------|------:|
| Bricks processed | 28 |
| Rayon threads | 20 |
| Sources detected (raw) | 102 629 |
| Duplicate pairs removed at brick boundaries (1″ tol.) | 90 |
| **Final catalog size** | **102 539 sources** |
| Catalog files | 7.6 MB FITS + 10.3 MB CSV |
| Median flux_auto | 3.24 nanomaggies |
| Brightest source (M87 halo) | 148 403 nanomaggies |
| **Total wall time** | **6.9 s** |
| Throughput | ~14 900 sources / s |

M87 true nucleus position (SIMBAD J2000): RA = 187.706°, Dec = +12.391°.
The flux-weighted centroid of the detected blob may differ due to the halo
being truncated at brick boundaries.

---

## Build

Requires **Rust 2024 edition (≥ 1.85)**.  No C dependencies — pure Rust.

```bash
# Build everything
cargo build --release

# Build a specific crate
cargo build --release -p vera-run
```

---

## Usage

### Single brick — full pipeline + catalog

```bash
vera-catalog fits/legacysurvey-1877p122-image-r.fits.fz
```

```
┌── Catalogue ──────────────────────────────────────────────────
│  Image      : 3600 x 3600 px
│  Pipeline   : 1.1s  (BG + detect + measure)
│  N sources  : 3576
│  flux_auto  : médiane=3.28  max=148402.7  (nanomaggies)
│
│  Fichiers écrits :
│    vera-1877p122-r.fits  (~240 kB)
│    vera-1877p122-r.csv   (~340 kB)
└───────────────────────────────────────────────────────────────
```

### Multi-brick pipeline — full Virgo field

```bash
vera-run fits/ r vera-virgo
```

```
┌── Vera multi-brick pipeline ───────────────────────────────────
│  Bricks  : 28   Threads : 20   Dedup tol : 1"
│  ...
│  Pipeline complet  : 6.6s
│  Sources brutes    : 102 629
│  Doublons supprimés: 90
│  Sources finales   : 102 539
│  Fichiers écrits   : vera-virgo.fits (7.6 MB)  vera-virgo.csv (10.3 MB)
│  Total             : 6.9s
└───────────────────────────────────────────────────────────────
```

### Interactive viewer

```bash
vera-viewer fits/legacysurvey-1877p122-image-r.fits.fz
```

- Loads image + runs full pipeline before window opens (~1.1 s)
- ZScale stretch with live asinh sliders
- Scroll-wheel zoom to cursor, drag to pan, double-click to fit
- Green ellipses = detected sources; orange = flagged; yellow = selected
- Click any source → RA, Dec, flux_auto, semi-axes, Kron radius, flags
- RA/Dec coordinate readout under the cursor

### Individual diagnostic binaries

```bash
vera-inspect    fits/legacysurvey-1877p122-image-r.fits.fz   # image stats + WCS
vera-background fits/legacysurvey-1877p122-image-r.fits.fz   # background map stats
vera-detect     fits/legacysurvey-1877p122-image-r.fits.fz   # detection stats
vera-measure    fits/legacysurvey-1877p122-image-r.fits.fz   # full measurement table
```

---

## Workspace structure

```
vera/
├── Cargo.toml                  ← workspace (Rust 2024, resolver = 2)
├── crates/
│   ├── vera-fits/              ← FITS/RICE I/O, WCS (fitsrs 0.4)
│   │   └── src/bin/vera-inspect
│   ├── vera-pipeline/          ← background · detect · measure (rayon)
│   │   └── src/bin/{vera-background, vera-detect, vera-measure}
│   ├── vera-catalog/           ← FITS binary table + CSV writer (pure Rust)
│   │   └── src/bin/vera-catalog
│   ├── vera-viewer/            ← interactive GPU viewer (eframe 0.35)
│   │   └── src/main.rs → vera-viewer
│   └── vera-run/               ← multi-brick parallel pipeline
│       └── src/main.rs → vera-run
└── fits/                       ← data directory (gitignored, ~1 GB)
```

### Key algorithms

| Module | Algorithm |
|--------|-----------|
| `background.rs` | κ-σ clipping (MAD × 1.4826, 3σ, 10 iter), SExtractor mode, bilinear interpolation |
| `convolve.rs` | Separated Gaussian, rayon-parallel rows |
| `detect.rs` | SNR map, Union-Find 8-connectivity connected components |
| `measure.rs` | Flux-weighted centroid, 2nd-order moments → eigenvalues, Kron radius & flux |
| `fits_write.rs` | Pure-Rust FITS binary table, big-endian, 2880-byte blocks, no cfitsio |
| `vera-run/main.rs` | O(N log N) RA-sorted deduplication across bricks |

---

## Catalog column schema

| Column | Format | Unit | Description |
|--------|--------|------|-------------|
| BRICK | 12A | — | Brick identifier (multi-brick only) |
| RA | D | deg | Right ascension (WCS) |
| DEC | D | deg | Declination (WCS) |
| X_IMAGE | D | pix | Centroid column (0-indexed) |
| Y_IMAGE | D | pix | Centroid row (0-indexed) |
| FLUX_ISO | E | nmg | Isophotal flux |
| FLUX_AUTO | E | nmg | Kron elliptical aperture flux |
| A_IMAGE | E | pix | Semi-major axis |
| B_IMAGE | E | pix | Semi-minor axis |
| THETA | E | deg | Position angle (x-axis CCW) |
| ELLIP | E | — | Ellipticity (1 − b/a) |
| KRON_RAD | E | — | Kron radius |
| NPIX | J | pix | Number of pixels in isophote |
| FLAGS | I | — | 0x01 = edge, 0x04 = saturated |

Calibration: nanomaggies (1 nmg = 3.631 × 10⁻³² W m⁻² Hz⁻¹),
consistent with DESI Legacy Survey DR10 and Rubin DP1 conventions.

---

## Data

28 DECam r-band bricks from DESI Legacy Survey DR10.
Each brick: 3600 × 3600 px · 0.262 arcsec/px · ~10–13 MB compressed.

```
fits/legacysurvey-<brick>-image-r.fits.fz     # science image (RICE-compressed)
fits/legacysurvey-<brick>-invvar-r.fits.fz    # inverse variance
fits/legacysurvey-<brick>-maskbits.fits.fz    # pixel bitmask
```

Files not included (gitignored, ~1 GB). Download via the Legacy Survey data access at
`https://www.legacysurvey.org/dr10/files/`.

---

## Compatibility

Legacy Survey DR10 and Rubin LSST DP1 share the same FITS format,
RICE compression, and nanomaggie calibration.  Switching to Rubin data
requires only changing the input file path.

---

## Attribution

Developed by **MaxKweeger** with **Claude Sonnet** (Anthropic) as AI pair-programmer.
All scientific decisions, algorithm choices, and project direction by MaxKweeger.

Any publication (including the planned JOSS paper) will credit:
- **MaxKweeger** — author and scientific director
- **Claude Sonnet** (Anthropic) — AI-assisted development

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

## Results — Virgo Cluster field, 28 bricks, band r

| Metric | Value |
|--------|-------|
| Total processing time (28 bricks, 20 threads) | **7.7 s** |
| Sources detected (raw) | 116 635 |
| Duplicate pairs removed at brick boundaries | 105 |
| **Final catalog size** | **116 530 sources** |
| Median flux_auto | 4.06 nanomaggies |
| Brightest source (M87 halo) | 101 723 nanomaggies |
| M87 true nucleus position | RA = 187.706°  Dec = +12.391° (J2000) |
| M87 detected blob centroid | RA ≈ 187.79°  Dec ≈ +12.38° (flux-weighted, truncated at brick edge) |
| M87 semi-major axis | 1 057 px (~4.6 arcmin) |
| Per-brick pipeline (single thread) | ~1.2 s |
| Background estimation alone | ~13 ms |
| Catalog files (28 bricks) | 8.9 MB FITS + 12 MB CSV |

Hardware: RTX 4070 Ti · i9-10850K (10c/20t) · 64 GB RAM.

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
│  Pipeline   : 1.2s  (BG + detect + measure)
│  N sources  : 5025
│  flux_auto  : médiane=4.28  max=89822.8  (nanomaggies)
│
│  Fichiers écrits :
│    vera-1877p122-r.fits  (335 kB)
│    vera-1877p122-r.csv   (474 kB)
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
│  Pipeline complet  : 7.3s
│  Sources brutes    : 116 635
│  Doublons supprimés: 105
│  Sources finales   : 116 530
│  Fichiers écrits   : vera-virgo.fits (8.9 MB)  vera-virgo.csv (12 MB)
│  Total             : 7.7s
└───────────────────────────────────────────────────────────────
```

### Interactive viewer

```bash
vera-viewer fits/legacysurvey-1877p122-image-r.fits.fz
```

- Loads image + runs full pipeline before window opens (~1.2 s)
- ZScale stretch with live asinh sliders
- Scroll-wheel zoom to cursor, drag to pan, double-click to fit
- Green ellipses = detected sources; orange = flagged; yellow = selected
- Click any source → RA, Dec, flux_auto, semi-axes, Kron radius, flags
- RA/Dec coordinate readout under the cursor

### Individual diagnostic binaries

```bash
vera-inspect   fits/legacysurvey-1877p122-image-r.fits.fz   # image stats + WCS
vera-background fits/legacysurvey-1877p122-image-r.fits.fz  # background map stats
vera-detect    fits/legacysurvey-1877p122-image-r.fits.fz   # detection stats
vera-measure   fits/legacysurvey-1877p122-image-r.fits.fz   # full measurement table
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

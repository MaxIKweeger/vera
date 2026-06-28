---
title: 'Vera: GPU-native astronomical source extraction in Rust'
tags:
  - astronomy
  - source extraction
  - photometry
  - GPU computing
  - Rust
  - FITS
authors:
  - name: MaxKweeger
    affiliation: Independent
date: 28 June 2026
bibliography: paper.bib
---

# Summary

Vera is a pure-Rust pipeline for detecting and measuring astronomical sources — stars
and galaxies — in optical survey images. Starting from compressed FITS images
[@wells1981], Vera estimates and subtracts the sky background, matches sources to
the point spread function (PSF) via Gaussian convolution, detects connected regions
above a signal-to-noise threshold, and produces a photometric catalogue (positions,
fluxes, shapes) in FITS binary table and CSV format. Gaussian convolution is
offloaded to the GPU via WebGPU compute shaders [@wgpu2024], achieving a 7.2×
speedup over the CPU implementation on an NVIDIA RTX 4070 Ti.

Vera was validated on 28 DECam *r*-band bricks from the DESI Legacy Imaging Survey
DR10 [@dey2019] covering the Virgo Cluster core (M87; RA ≈ 187.7°, Dec ≈ +12.4°),
the same field captured in the Vera C. Rubin Observatory first-light image. The
pipeline processes all 28 bricks (28 × 3600 × 3600 pixels) in **4.8 seconds** and
produces a catalogue of 102 539 sources on a consumer desktop (Intel i9-10850K,
RTX 4070 Ti, 64 GB RAM, Windows 11).

# Statement of Need

Source extraction is a fundamental step in nearly every optical astronomy pipeline.
The canonical tool, SExtractor [@bertin1996; @bertin2010], was written in C in 1996
and remains widely used. While extremely capable, SExtractor and its successors
carry significant installation friction (cfitsio, ATLAS, FFTW dependencies), no
native GPU acceleration, and no built-in multi-image parallelism.

The Vera C. Rubin Observatory Legacy Survey of Space and Time (LSST) will generate
~15 TB of imaging data per night [@ivezic2019], processing millions of sources per
exposure. Community tools for this scale must be fast, portable, and free of legacy
C dependencies. Similarly, the DESI Legacy Imaging Survey DR10 — already publicly
available — covers ~20 000 deg² and is underutilised by researchers who lack access
to institutional computing clusters.

Vera addresses this gap: it runs on a consumer GPU without any C dependencies
(pure Rust, zero unsafe code in library crates), installs with a single
`cargo build --release`, and processes a full DECam pointing in under a second.
Its FITS output is directly compatible with TOPCAT, astropy, and Rubin DP1
conventions, and its source extraction results are quantitatively consistent
with SExtractor on the same data.

# Algorithms

## Background Estimation

Vera divides each image into a grid of cells (default: 64 × 64 pixels) and
estimates the sky background in each cell using iterative κ-σ clipping
[@chauvenet1863]. In each iteration, pixels deviating by more than κ = 3 median
absolute deviations (MAD × 1.4826) from the cell median are rejected; convergence
is reached when no further pixels are clipped (maximum 10 iterations). The
background level is estimated via the SExtractor mode approximation:
mode ≈ 2.5 × median − 1.5 × mean when the clipped distribution is mildly skewed
(|mean − median| / σ < 0.3), falling back to the median for crowded fields
[@bertin1996]. The per-cell estimates are bilinearly interpolated to full image
resolution.

## Source Detection

PSF matching is performed by convolving the background-subtracted residual with a
separable Gaussian kernel of width σ_PSF = 1.6 pixels (corresponding to ~1 arcsec
seeing at the DECam plate scale of 0.262 arcsec/pixel). On GPU-equipped systems,
this convolution runs as two sequential WGSL compute shader passes (horizontal then
vertical) with 16 × 16 workgroups on the RTX 4070 Ti, falling back to a
Rayon-parallelised CPU implementation otherwise. A signal-to-noise ratio (SNR) map
is constructed by dividing the convolved residual by the interpolated RMS map.
Pixels exceeding a threshold of 1.5σ are extracted, and connected components with
8-connectivity are identified using a two-pass Union–Find algorithm with path
compression [@rosenfeld1966].

## Photometric Measurements

For each detected component, Vera computes: the flux-weighted centroid (first-order
moments); the second-order moments → semi-major axis *a*, semi-minor axis *b*, and
position angle θ; the ellipticity ε = 1 − *b*/*a*; isophotal flux (sum over all
connected pixels); and the automatic aperture flux *f*_auto using the Kron radius
[@kron1980], defined as the first-moment radius within an elliptical aperture of
2.5 × *a*_IMAGE × *b*_IMAGE. World coordinate system (WCS) positions (RA, Dec) are
computed from pixel centroids using the TAN projection encoded in the FITS header.
Source flags identify edge truncations (0x01) and saturated pixels (0x04).

# Performance

On 28 DECam *r*-band bricks (3600 × 3600 pixels each, RICE-compressed):

| Stage | CPU-only | GPU (RTX 4070 Ti) |
|---|---:|---:|
| Gaussian convolution (per brick) | 304 ms | **42 ms** (7.2×) |
| Full pipeline (28 bricks, 20 threads) | 6.9 s | **4.8 s** (1.4×) |
| Final catalogue | 102 539 sources | 102 539 sources |

The modest overall speedup (1.4×) relative to the convolution speedup (7.2×) reflects
that background estimation, CCL, and photometry remain on the CPU, and that
20 Rayon threads compete for a single GPU. Source counts are identical between the
CPU and GPU paths, confirming numerical equivalence.

# Acknowledgements

Vera was developed by MaxKweeger with Claude Sonnet (Anthropic) as AI pair-programmer.
All scientific decisions, algorithm choices, and project direction by MaxKweeger.

Validation data: DESI Legacy Imaging Survey DR10 [@dey2019], provided by the
Legacy Survey collaboration. The Rubin Observatory first-light image (NOIRLab
image noirlab2521a) motivated the choice of the Virgo Cluster field.

# References

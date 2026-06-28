use std::collections::HashMap;
use std::f32::consts::PI;

use ndarray::Array2;
use rayon::prelude::*;

use crate::background::BackgroundMap;
use crate::detect::DetectionList;
use vera_fits::WcsHeader;

// ── Configuration ─────────────────────────────────────────────────────────────

pub struct MeasureConfig {
    /// Kron aperture multiplier (SExtractor default: 2.5).
    pub kron_factor: f32,
    /// Minimum Kron radius in units of ellipse scale (SExtractor: 3.5 px effective).
    pub kron_min_radius: f32,
}

impl Default for MeasureConfig {
    fn default() -> Self {
        Self { kron_factor: 2.5, kron_min_radius: 3.5 }
    }
}

// ── Output struct ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Measurement {
    pub label: u32,
    pub npix: u32,

    /// Flux-weighted centroid (column, 0-indexed pixel).
    pub x_c: f64,
    /// Flux-weighted centroid (row, 0-indexed pixel).
    pub y_c: f64,
    pub ra: Option<f64>,
    pub dec: Option<f64>,

    /// Semi-major axis from 2nd moments (pixels).
    pub a: f32,
    /// Semi-minor axis from 2nd moments (pixels).
    pub b: f32,
    /// Position angle (degrees, from +col axis, counter-clockwise).
    pub theta: f32,
    /// Ellipticity = 1 - b/a.
    pub ellipticity: f32,

    /// Kron radius in units of the ellipse scale.
    pub kron_radius: f32,
    /// Sum of flux within the detection isophote (nanomaggies).
    pub flux_iso: f32,
    /// Flux within kron_factor × kron_radius ellipse (nanomaggies).
    pub flux_auto: f32,

    /// Quality flag bitmask.
    pub flags: u16,
}

pub const FLAG_EDGE: u16      = 0x01; // bbox touches image border
pub const FLAG_SATURATED: u16 = 0x04; // peak pixel > saturation limit

// ── Public entry point ────────────────────────────────────────────────────────

/// Compute photometric measurements for all detections in parallel.
pub fn measure_all(
    image: &Array2<f32>,
    bg_map: &BackgroundMap,
    detections: &DetectionList,
    wcs: Option<&WcsHeader>,
    config: &MeasureConfig,
) -> Vec<Measurement> {
    let residual = bg_map.subtract(image);
    let (nrows, ncols) = image.dim();

    // One-pass scan: build sorted pixel list per label.
    let mut pixels_by_label: HashMap<u32, Vec<(usize, usize)>> = HashMap::new();
    for r in 0..nrows {
        for c in 0..ncols {
            let l = detections.label_map[[r, c]];
            if l > 0 {
                pixels_by_label.entry(l).or_default().push((r, c));
            }
        }
    }

    detections
        .detections
        .par_iter()
        .map(|det| {
            let pixels = pixels_by_label
                .get(&det.label)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            measure_one(
                det.label,
                det.npix,
                &det.bbox,
                pixels,
                &residual,
                wcs,
                config,
                nrows,
                ncols,
            )
        })
        .collect()
}

// ── Single-source measurement ─────────────────────────────────────────────────

fn measure_one(
    label: u32,
    npix: u32,
    bbox: &[usize; 4],
    pixels: &[(usize, usize)],
    image: &Array2<f32>,
    wcs: Option<&WcsHeader>,
    config: &MeasureConfig,
    nrows: usize,
    ncols: usize,
) -> Measurement {
    // ── Isophotal flux + flux-weighted centroid ───────────────────────────────
    let mut flux_iso = 0.0f32;
    let mut sum_w = 0.0f64;
    let mut sum_wx = 0.0f64;
    let mut sum_wy = 0.0f64;

    for &(r, c) in pixels {
        let v = image[[r, c]];
        if !v.is_finite() {
            continue;
        }
        flux_iso += v;
        let w = v.max(0.0) as f64; // positive-flux weighting for centroid
        sum_w  += w;
        sum_wx += w * c as f64;
        sum_wy += w * r as f64;
    }

    let (x_c, y_c) = if sum_w > 0.0 {
        (sum_wx / sum_w, sum_wy / sum_w)
    } else {
        (
            (bbox[2] + bbox[3]) as f64 / 2.0,
            (bbox[0] + bbox[1]) as f64 / 2.0,
        )
    };

    // ── 2nd-order moments → semi-axes ────────────────────────────────────────
    let mut mxx = 0.0f64;
    let mut myy = 0.0f64;
    let mut mxy = 0.0f64;
    let mut sum_w2 = 0.0f64;

    for &(r, c) in pixels {
        let v = image[[r, c]];
        if !v.is_finite() {
            continue;
        }
        let w = v.max(0.0) as f64;
        let dx = c as f64 - x_c;
        let dy = r as f64 - y_c;
        mxx   += w * dx * dx;
        myy   += w * dy * dy;
        mxy   += w * dx * dy;
        sum_w2 += w;
    }

    let (a, b, theta) = if sum_w2 > 0.0 {
        mxx /= sum_w2;
        myy /= sum_w2;
        mxy /= sum_w2;
        moments_to_axes(mxx as f32, myy as f32, mxy as f32)
    } else {
        let r = ((npix as f32) / PI).sqrt();
        (r, r, 0.0f32)
    };

    // Clamp: b <= a, both >= 0.5 px (avoids division by zero).
    let a = a.max(0.5);
    let b = b.max(0.5).min(a);
    let ellipticity = 1.0 - b / a;

    // ── Kron radius (dimensionless, in units of ellipse scale) ───────────────
    let kron_radius = {
        let raw = compute_kron_radius(pixels, image, x_c, y_c, a, b, theta);
        raw.max(config.kron_min_radius)
    };

    // ── Kron (auto) flux ─────────────────────────────────────────────────────
    let flux_auto = kron_flux(
        image,
        x_c,
        y_c,
        a,
        b,
        theta,
        kron_radius * config.kron_factor,
        bbox,
        nrows,
        ncols,
    );

    // ── Sky coordinates ───────────────────────────────────────────────────────
    let (ra, dec) = match wcs {
        Some(w) => {
            let (ra, dec) = w.pix_to_sky(x_c, y_c);
            (Some(ra), Some(dec))
        }
        None => (None, None),
    };

    // ── Flags ─────────────────────────────────────────────────────────────────
    let mut flags = 0u16;
    if bbox[0] == 0 || bbox[1] + 1 >= nrows || bbox[2] == 0 || bbox[3] + 1 >= ncols {
        flags |= FLAG_EDGE;
    }

    Measurement {
        label,
        npix,
        x_c,
        y_c,
        ra,
        dec,
        a,
        b,
        theta,
        ellipticity,
        kron_radius,
        flux_iso,
        flux_auto,
        flags,
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Diagonalise symmetric 2×2 matrix of 2nd-order moments → (a, b, theta).
fn moments_to_axes(mxx: f32, myy: f32, mxy: f32) -> (f32, f32, f32) {
    let half_diff = (mxx - myy) / 2.0;
    let disc = (half_diff * half_diff + mxy * mxy).sqrt();
    let a = ((mxx + myy) / 2.0 + disc).max(0.0).sqrt();
    let b = ((mxx + myy) / 2.0 - disc).max(0.0).sqrt();
    let theta = 0.5 * mxy.atan2(half_diff) * 180.0 / PI;
    (a, b, theta)
}

/// Kron radius: Σ(r_ell × I) / Σ(I) summed over the isophotal footprint.
/// Note: SExtractor computes the same sum over a larger 6×a_image aperture;
/// restricting to the footprint is equivalent for compact sources and avoids
/// scanning empty sky pixels far from the object.
fn compute_kron_radius(
    pixels: &[(usize, usize)],
    image: &Array2<f32>,
    x_c: f64,
    y_c: f64,
    a: f32,
    b: f32,
    theta: f32,
) -> f32 {
    let ct = (theta * PI / 180.0).cos() as f64;
    let st = (theta * PI / 180.0).sin() as f64;
    let a = a as f64;
    let b = b as f64;

    let mut sum_ri = 0.0f64;
    let mut sum_i  = 0.0f64;

    for &(r, c) in pixels {
        let v = image[[r, c]];
        if !v.is_finite() || v <= 0.0 {
            continue;
        }
        let dx = c as f64 - x_c;
        let dy = r as f64 - y_c;
        let u =  dx * ct + dy * st;
        let w = -dx * st + dy * ct;
        let r_ell = ((u / a).powi(2) + (w / b).powi(2)).sqrt();
        sum_ri += r_ell * v as f64;
        sum_i  += v as f64;
    }

    if sum_i > 0.0 { (sum_ri / sum_i) as f32 } else { 1.0 }
}

/// Sum flux within the Kron ellipse (radius = kron_factor × kron_radius).
fn kron_flux(
    image: &Array2<f32>,
    x_c: f64,
    y_c: f64,
    a: f32,
    b: f32,
    theta: f32,
    aperture: f32,
    bbox: &[usize; 4],
    nrows: usize,
    ncols: usize,
) -> f32 {
    let ct = (theta * PI / 180.0).cos() as f64;
    let st = (theta * PI / 180.0).sin() as f64;
    let a  = a as f64;
    let b  = b as f64;
    let ap = aperture as f64;

    // Expand bbox by aperture radius.
    let margin = (ap * a.max(b)).ceil() as usize + 1;
    let r0 = bbox[0].saturating_sub(margin);
    let r1 = (bbox[1] + margin + 1).min(nrows);
    let c0 = bbox[2].saturating_sub(margin);
    let c1 = (bbox[3] + margin + 1).min(ncols);

    let mut flux = 0.0f32;
    for r in r0..r1 {
        for c in c0..c1 {
            let dx = c as f64 - x_c;
            let dy = r as f64 - y_c;
            let u =  dx * ct + dy * st;
            let w = -dx * st + dy * ct;
            if (u / a).powi(2) + (w / b).powi(2) <= ap * ap {
                let v = image[[r, c]];
                if v.is_finite() {
                    flux += v;
                }
            }
        }
    }
    flux
}

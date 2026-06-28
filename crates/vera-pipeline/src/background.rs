use ndarray::Array2;
use rayon::prelude::*;

pub struct BackgroundConfig {
    /// Cell size in pixels (SExtractor default: 64).
    pub mesh_size: usize,
    /// κ for iterative sigma-clipping (SExtractor default: 3.0).
    pub sigma_clip: f32,
    /// Maximum clipping iterations.
    pub max_iter: usize,
}

impl Default for BackgroundConfig {
    fn default() -> Self {
        Self { mesh_size: 64, sigma_clip: 3.0, max_iter: 10 }
    }
}

pub struct BackgroundMap {
    pub mesh_size: usize,
    /// Low-resolution background grid [grid_rows × grid_cols].
    pub bg_grid: Array2<f32>,
    /// Low-resolution RMS grid [grid_rows × grid_cols].
    pub rms_grid: Array2<f32>,
    /// Original image shape (nrows, ncols).
    pub img_shape: (usize, usize),
}

impl BackgroundMap {
    /// Estimate background and RMS from an image using sigma-clipped cell medians.
    pub fn estimate(image: &Array2<f32>, config: &BackgroundConfig) -> Self {
        let (nrows, ncols) = image.dim();
        let m = config.mesh_size;

        let grid_rows = nrows.div_ceil(m);
        let grid_cols = ncols.div_ceil(m);
        let n_cells = grid_rows * grid_cols;

        // Parallel estimation: each cell -> (bg, rms)
        let cells: Vec<(f32, f32)> = (0..n_cells)
            .into_par_iter()
            .map(|idx| {
                let gr = idx / grid_cols;
                let gc = idx % grid_cols;
                let r0 = gr * m;
                let c0 = gc * m;
                let r1 = (r0 + m).min(nrows);
                let c1 = (c0 + m).min(ncols);

                let mut px: Vec<f32> = image
                    .slice(ndarray::s![r0..r1, c0..c1])
                    .iter()
                    .copied()
                    .filter(|v| v.is_finite())
                    .collect();

                sigma_clip_stats(&mut px, config.sigma_clip, config.max_iter)
            })
            .collect();

        let bg_flat: Vec<f32> = cells.iter().map(|&(bg, _)| bg).collect();
        let rms_flat: Vec<f32> = cells.iter().map(|&(_, rms)| rms).collect();

        let bg_grid = Array2::from_shape_vec((grid_rows, grid_cols), bg_flat).unwrap();
        let rms_grid = Array2::from_shape_vec((grid_rows, grid_cols), rms_flat).unwrap();

        Self { mesh_size: m, bg_grid, rms_grid, img_shape: (nrows, ncols) }
    }

    /// Interpolate the background grid to full image resolution.
    pub fn background(&self) -> Array2<f32> {
        interp_to_image(&self.bg_grid, self.img_shape, self.mesh_size)
    }

    /// Interpolate the RMS grid to full image resolution.
    pub fn rms(&self) -> Array2<f32> {
        interp_to_image(&self.rms_grid, self.img_shape, self.mesh_size)
    }

    /// Subtract the interpolated background from an image.
    pub fn subtract(&self, image: &Array2<f32>) -> Array2<f32> {
        image - &self.background()
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Iterative σ-clipping: returns (background_estimate, sigma).
fn sigma_clip_stats(px: &mut Vec<f32>, kappa: f32, max_iter: usize) -> (f32, f32) {
    if px.is_empty() {
        return (f32::NAN, f32::NAN);
    }

    for _ in 0..max_iter {
        let n = px.len();
        px.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let median = px[n / 2];

        // MAD-based sigma estimate (robust, unbiased for Gaussian: MAD * 1.4826)
        let mut deviations: Vec<f32> = px.iter().map(|&x| (x - median).abs()).collect();
        deviations.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let mad = deviations[n / 2];
        let sig = mad * 1.4826_f32;

        if sig == 0.0 {
            break;
        }

        let lo = median - kappa * sig;
        let hi = median + kappa * sig;
        let before = px.len();
        px.retain(|&x| x >= lo && x <= hi);
        if px.len() == before {
            break;
        }
    }

    if px.is_empty() {
        return (f32::NAN, f32::NAN);
    }

    let n = px.len();
    px.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let median = px[n / 2];
    let mean: f32 = px.iter().sum::<f32>() / n as f32;
    let var: f32 = px.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n as f32;
    let sigma = var.sqrt();

    // SExtractor mode estimate: use the Pearson approximation when the clipped
    // distribution is mildly asymmetric (|mean−median|/σ < 0.3); fall back to
    // the median when the field is so crowded that the approximation breaks down.
    let bg = if sigma > 0.0 && (mean - median).abs() / sigma < 0.3 {
        2.5 * median - 1.5 * mean
    } else {
        median
    };

    (bg, sigma)
}

/// Bilinear interpolation of a low-resolution grid to full image size.
/// The center of grid cell (gr, gc) maps to pixel (gr*m + m/2, gc*m + m/2).
fn interp_to_image(grid: &Array2<f32>, img_shape: (usize, usize), mesh: usize) -> Array2<f32> {
    let (nrows, ncols) = img_shape;
    let m = mesh as f32;

    Array2::from_shape_fn((nrows, ncols), |(r, c)| {
        // Fractional grid coordinate of this pixel's center
        let gr = (r as f32 + 0.5) / m - 0.5;
        let gc = (c as f32 + 0.5) / m - 0.5;
        bilinear(grid, gr, gc)
    })
}

fn bilinear(grid: &Array2<f32>, r: f32, c: f32) -> f32 {
    let (nr, nc) = grid.dim();
    let clamp_r = |i: isize| i.clamp(0, nr as isize - 1) as usize;
    let clamp_c = |j: isize| j.clamp(0, nc as isize - 1) as usize;

    let r0 = r.floor() as isize;
    let c0 = c.floor() as isize;
    let dr = r - r0 as f32;
    let dc = c - c0 as f32;

    let v00 = grid[[clamp_r(r0),     clamp_c(c0)    ]];
    let v01 = grid[[clamp_r(r0),     clamp_c(c0 + 1)]];
    let v10 = grid[[clamp_r(r0 + 1), clamp_c(c0)    ]];
    let v11 = grid[[clamp_r(r0 + 1), clamp_c(c0 + 1)]];

    v00 * (1.0 - dr) * (1.0 - dc)
        + v01 * (1.0 - dr) * dc
        + v10 * dr * (1.0 - dc)
        + v11 * dr * dc
}

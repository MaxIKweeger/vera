use ndarray::Array2;
use rayon::prelude::*;

/// Separable Gaussian smoothing: horizontal pass then vertical pass (via transpose).
pub fn gaussian_smooth(image: &Array2<f32>, sigma: f32) -> Array2<f32> {
    let kernel = gaussian_1d(sigma, 3.0);
    let tmp = convolve_rows(image, &kernel);
    // Transpose, convolve rows (== convolve cols of original), transpose back.
    let tmp_t = tmp.t().to_owned();
    let out_t = convolve_rows(&tmp_t, &kernel);
    out_t.t().to_owned()
}

/// 1-D normalised Gaussian kernel, truncated at `truncate * sigma` on each side.
fn gaussian_1d(sigma: f32, truncate: f32) -> Vec<f32> {
    let radius = (truncate * sigma).ceil() as usize;
    let s2 = 2.0 * sigma * sigma;
    let mut k: Vec<f32> = (0..=2 * radius)
        .map(|i| {
            let x = i as f32 - radius as f32;
            (-x * x / s2).exp()
        })
        .collect();
    let sum: f32 = k.iter().sum();
    k.iter_mut().for_each(|v| *v /= sum);
    k
}

/// Convolve each row independently (parallel via rayon).
/// NaN/Inf pixels are treated as 0 so they don't propagate into neighbours.
fn convolve_rows(image: &Array2<f32>, kernel: &[f32]) -> Array2<f32> {
    let (nrows, ncols) = image.dim();
    let radius = kernel.len() / 2;

    let rows: Vec<Vec<f32>> = (0..nrows)
        .into_par_iter()
        .map(|r| {
            let row = image.row(r);
            (0..ncols)
                .map(|c| {
                    let mut sum = 0.0f32;
                    for (ki, &kv) in kernel.iter().enumerate() {
                        let ic = (c as isize + ki as isize - radius as isize)
                            .clamp(0, ncols as isize - 1) as usize;
                        let px = row[ic];
                        if px.is_finite() {
                            sum += px * kv;
                        }
                    }
                    sum
                })
                .collect()
        })
        .collect();

    let flat: Vec<f32> = rows.into_iter().flatten().collect();
    Array2::from_shape_vec((nrows, ncols), flat).unwrap()
}

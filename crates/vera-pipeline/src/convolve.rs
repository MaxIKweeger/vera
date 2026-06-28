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
pub fn gaussian_1d(sigma: f32, truncate: f32) -> Vec<f32> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn gaussian_1d_sums_to_one() {
        for &sigma in &[0.5f32, 1.0, 1.6, 3.0] {
            let k = gaussian_1d(sigma, 3.0);
            let sum: f32 = k.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "sigma={sigma}: kernel sum={sum}");
        }
    }

    #[test]
    fn gaussian_1d_is_symmetric() {
        let k = gaussian_1d(2.0, 3.0);
        let n = k.len();
        for i in 0..n / 2 {
            assert!((k[i] - k[n - 1 - i]).abs() < 1e-7, "kernel not symmetric at i={i}");
        }
    }

    #[test]
    fn smooth_uniform_field_unchanged() {
        let img = Array2::from_elem((32, 32), 5.0f32);
        let out = gaussian_smooth(&img, 1.6);
        for &v in out.iter() {
            assert!((v - 5.0).abs() < 1e-3, "expected 5.0, got {v}");
        }
    }

    #[test]
    fn smooth_preserves_flux() {
        // A single bright pixel — total flux should be ~conserved after smoothing.
        let mut img: Array2<f32> = Array2::zeros((64, 64));
        img[[32, 32]] = 1.0;
        let out = gaussian_smooth(&img, 2.0);
        let total: f32 = out.iter().sum();
        assert!((total - 1.0).abs() < 0.05, "flux conservation: total={total}");
    }

    #[test]
    fn smooth_peak_remains_at_center() {
        let mut img: Array2<f32> = Array2::zeros((32, 32));
        img[[16, 16]] = 1.0;
        let out = gaussian_smooth(&img, 1.6);
        let center = out[[16, 16]];
        let corner = out[[0, 0]];
        assert!(center > corner, "center={center} should exceed corner={corner}");
    }
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

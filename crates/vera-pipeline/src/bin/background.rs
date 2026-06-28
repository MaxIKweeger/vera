use std::path::Path;
use vera_fits::read_image_f32;
use vera_pipeline::background::{BackgroundConfig, BackgroundMap};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path_str = args.get(1).cloned().unwrap_or_else(|| {
        eprintln!("Usage: vera-background <fichier.fits[.fz]> [mesh_size] [sigma]");
        std::process::exit(1);
    });
    let mesh_size: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(64);
    let sigma_clip: f32  = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(3.0);

    let path = Path::new(&path_str);
    println!("Brick       : {}", path.display());

    let (image, wcs) = match read_image_f32(path) {
        Ok(v)  => v,
        Err(e) => { eprintln!("Erreur FITS : {e}"); std::process::exit(1); }
    };
    let (nrows, ncols) = image.dim();
    println!("Taille      : {} x {} px", ncols, nrows);

    let config = BackgroundConfig { mesh_size, sigma_clip, max_iter: 10 };

    let t0 = std::time::Instant::now();
    let bg_map = BackgroundMap::estimate(&image, &config);
    println!("BG estimation : {:.1?}  (grille {} x {} cellules de {}px)",
        t0.elapsed(),
        bg_map.bg_grid.dim().1, bg_map.bg_grid.dim().0,
        mesh_size);

    // Grid statistics
    let bg_vals: Vec<f32> = bg_map.bg_grid.iter().copied().filter(|v| v.is_finite()).collect();
    let rms_vals: Vec<f32> = bg_map.rms_grid.iter().copied().filter(|v| v.is_finite()).collect();

    let bg_min  = bg_vals.iter().cloned().fold(f32::INFINITY,     f32::min);
    let bg_max  = bg_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let bg_mean = bg_vals.iter().sum::<f32>() / bg_vals.len() as f32;

    let rms_min  = rms_vals.iter().cloned().fold(f32::INFINITY,     f32::min);
    let rms_max  = rms_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let rms_mean = rms_vals.iter().sum::<f32>() / rms_vals.len() as f32;

    println!();
    println!("┌── Fond de ciel (grille basse résolution) ─────────────────────");
    println!("│  BG  : min={bg_min:.5}  max={bg_max:.5}  mean={bg_mean:.5}  (nanomaggies)");
    println!("│  RMS : min={rms_min:.5}  max={rms_max:.5}  mean={rms_mean:.5}");
    println!("│");

    // Interpolated background then subtract
    let t1 = std::time::Instant::now();
    let residual = bg_map.subtract(&image);
    println!("│  Interpolation + soustraction : {:.1?}", t1.elapsed());

    let res: Vec<f32> = residual.iter().copied().filter(|v| v.is_finite()).collect();
    let res_mean = res.iter().sum::<f32>() / res.len() as f32;
    let res_var  = res.iter().map(|&x| (x - res_mean).powi(2)).sum::<f32>() / res.len() as f32;
    let res_rms  = res_var.sqrt();

    let mut res_sorted = res.clone();
    res_sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let res_median = res_sorted[res_sorted.len() / 2];

    println!("│");
    println!("│  Image résiduelle après soustraction du fond :");
    println!("│    mean={res_mean:.5}  median={res_median:.5}  RMS={res_rms:.5}");
    println!("│    (idéalement : mean≈0, médian≈0, RMS ≈ bruit de Poisson)");

    if let Some(w) = &wcs {
        let (ra, dec) = w.pix_to_sky(ncols as f64 / 2.0, nrows as f64 / 2.0);
        println!("│");
        println!("│  WCS centre : RA={ra:.4}°  Dec={dec:.4}°");
    }

    println!("└────────────────────────────────────────────────────────────────");
}

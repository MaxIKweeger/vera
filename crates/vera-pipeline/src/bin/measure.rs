use std::path::Path;
use vera_fits::read_image_f32;
use vera_pipeline::background::{BackgroundConfig, BackgroundMap};
use vera_pipeline::detect::{detect, detect_gpu, DetectConfig};
use vera_pipeline::gpu::GpuContext;
use vera_pipeline::measure::{measure_all, MeasureConfig};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path_str = args.get(1).cloned().unwrap_or_else(|| {
        eprintln!("Usage: vera-measure <fichier.fits[.fz]> [sigma_psf] [thresh] [minarea]");
        std::process::exit(1);
    });
    let sigma_psf: f32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1.6);
    let thresh: f32    = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1.5);
    let minarea: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(5);

    let path = Path::new(&path_str);
    println!("Brick  : {}", path.display());

    let (image, wcs) = match read_image_f32(path) {
        Ok(v)  => v,
        Err(e) => { eprintln!("Erreur FITS : {e}"); std::process::exit(1); }
    };
    let (nrows, ncols) = image.dim();
    println!("Taille : {} x {} px\n", ncols, nrows);

    let gpu = GpuContext::new();
    if gpu.is_some() {
        println!("GPU    : initialisé (convolution wgpu)\n");
    } else {
        println!("GPU    : non disponible (convolution CPU)\n");
    }

    let t0 = std::time::Instant::now();
    let bg_map = BackgroundMap::estimate(&image, &BackgroundConfig::default());
    let t_bg = t0.elapsed();

    let config = DetectConfig { sigma_psf, thresh, minarea };

    let t1 = std::time::Instant::now();
    let det_result = match gpu.as_ref() {
        Some(ctx) => detect_gpu(&image, &bg_map, &config, ctx),
        None      => detect(&image, &bg_map, &config),
    };
    let t_det = t1.elapsed();

    let t2 = std::time::Instant::now();
    let mut measurements = measure_all(
        &image, &bg_map, &det_result, wcs.as_ref(), &MeasureConfig::default(),
    );
    let t_mes = t2.elapsed();

    measurements.sort_by(|a, b| b.flux_auto.partial_cmp(&a.flux_auto).unwrap());

    let n = measurements.len();
    let flux_vals: Vec<f32> = measurements.iter().map(|m| m.flux_auto).collect();
    let flux_mean = flux_vals.iter().sum::<f32>() / n as f32;
    let mut fs = flux_vals.clone();
    fs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let flux_median = fs[fs.len() / 2];

    let conv_label = if gpu.is_some() { "GPU (wgpu)" } else { "CPU (rayon)" };

    println!("┌── Pipeline complet ──────────────────────────────────────────");
    println!("│  Background  : {t_bg:.1?}");
    println!("│  Détection ({conv_label}) : {t_det:.1?}");
    println!("│  Mesures     : {t_mes:.1?}");
    println!("│  TOTAL       : {:.1?}", t0.elapsed());
    println!("│");
    println!("│  N sources    : {n}");
    println!("│  flux_auto    : médiane={flux_median:.4}  moy={flux_mean:.4}  (nanomaggies)");

    let n_edge = measurements.iter().filter(|m| m.flags & vera_pipeline::measure::FLAG_EDGE != 0).count();
    println!("│  Flags edge   : {n_edge} sources");
    println!("│");
    println!("│  Top 15 sources par flux_auto (nanomaggies) :");
    println!("│  {:>4}  {:>10}  {:>10}  {:>9}  {:>7}  {:>6}  {:>6}  {:>5}  {:>5}",
        "#", "RA (°)", "Dec (°)", "flux_auto", "flux_iso", "a(px)", "b(px)", "θ(°)", "flags");
    println!("│  {}", "─".repeat(78));

    for (i, m) in measurements.iter().take(15).enumerate() {
        let ra_s  = m.ra.map(|v| format!("{v:10.4}")).unwrap_or_else(|| format!("{:>10}", "—"));
        let dec_s = m.dec.map(|v| format!("{v:10.4}")).unwrap_or_else(|| format!("{:>10}", "—"));
        println!(
            "│  {:>4}  {}  {}  {:>9.3}  {:>7.3}  {:>6.1}  {:>6.1}  {:>5.1}  0x{:02X}",
            i + 1, ra_s, dec_s,
            m.flux_auto, m.flux_iso,
            m.a, m.b, m.theta,
            m.flags
        );
    }

    println!("└───────────────────────────────────────────────────────────────");
}

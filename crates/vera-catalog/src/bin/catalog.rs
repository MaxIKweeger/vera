use std::path::Path;
use vera_fits::read_image_f32;
use vera_pipeline::background::{BackgroundConfig, BackgroundMap};
use vera_pipeline::detect::{detect, detect_gpu, DetectConfig};
use vera_pipeline::gpu::GpuContext;
use vera_pipeline::measure::{measure_all, measure_all_gpu, MeasureConfig};
use vera_catalog::{csv_write, fits_write};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path_str = args.get(1).cloned().unwrap_or_else(|| {
        eprintln!("Usage: vera-catalog <brick.fits.fz> [output_stem] [band]");
        std::process::exit(1);
    });

    let path = Path::new(&path_str);

    // Derive brick name from filename: "legacysurvey-1877p122-image-r.fits.fz" → "1877p122"
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("unknown");
    let brick = stem.split('-').nth(1).unwrap_or(stem);
    let band  = args.get(3).map(|s| s.as_str()).unwrap_or("r");

    let out_stem = args.get(2).cloned()
        .unwrap_or_else(|| format!("vera-{brick}-{band}"));

    println!("Brick   : {brick}  band={band}");
    println!("Input   : {}", path.display());
    println!("Output  : {out_stem}.{{fits,csv}}");

    let gpu = GpuContext::new();

    let t0 = std::time::Instant::now();

    let (image, wcs) = match read_image_f32(path) {
        Ok(v)  => v,
        Err(e) => { eprintln!("Erreur FITS : {e}"); std::process::exit(1); }
    };
    let (nrows, ncols) = image.dim();

    let bg_map = BackgroundMap::estimate(&image, &BackgroundConfig::default());
    let det = match gpu.as_ref() {
        Some(ctx) => detect_gpu(&image, &bg_map, &DetectConfig::default(), ctx),
        None      => detect(&image, &bg_map, &DetectConfig::default()),
    };
    let mut meas = match gpu.as_ref() {
        Some(ctx) => measure_all_gpu(&image, &bg_map, &det, wcs.as_ref(), &MeasureConfig::default(), ctx),
        None      => measure_all(&image, &bg_map, &det, wcs.as_ref(), &MeasureConfig::default()),
    };

    // Sort by flux_auto descending (standard catalogue ordering).
    meas.sort_by(|a, b| b.flux_auto.partial_cmp(&a.flux_auto).unwrap());

    let t_pipeline = t0.elapsed();

    // Write FITS binary table.
    let fits_path = format!("{out_stem}.fits");
    let t1 = std::time::Instant::now();
    fits_write::write(Path::new(&fits_path), &meas, brick, band)
        .unwrap_or_else(|e| eprintln!("FITS write error: {e}"));
    let t_fits = t1.elapsed();

    // Write CSV.
    let csv_path = format!("{out_stem}.csv");
    let t2 = std::time::Instant::now();
    csv_write::write(Path::new(&csv_path), &meas, brick, band)
        .unwrap_or_else(|e| eprintln!("CSV write error: {e}"));
    let t_csv = t2.elapsed();

    let n = meas.len();
    let flux_vals: Vec<f32> = meas.iter().map(|m| m.flux_auto).collect();
    let flux_med = {
        let mut s = flux_vals.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        s[s.len() / 2]
    };
    let flux_max = flux_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let fits_size = std::fs::metadata(&fits_path).map(|m| m.len()).unwrap_or(0);
    let csv_size  = std::fs::metadata(&csv_path).map(|m| m.len()).unwrap_or(0);

    println!();
    println!("┌── Catalogue ──────────────────────────────────────────────────");
    println!("│  Image      : {} x {} px", ncols, nrows);
    println!("│  Pipeline   : {t_pipeline:.1?}  (BG + detect + measure)");
    println!("│  N sources  : {n}");
    println!("│  flux_auto  : médiane={flux_med:.4}  max={flux_max:.1}  (nanomaggies)");
    println!("│");
    println!("│  Fichiers écrits :");
    println!("│    {fits_path}  ({:.1} kB, {t_fits:.1?})", fits_size as f64 / 1024.0);
    println!("│    {csv_path}  ({:.1} kB, {t_csv:.1?})", csv_size as f64 / 1024.0);
    println!("│");
    println!("│  Colonnes FITS : RA DEC X_IMAGE Y_IMAGE FLUX_ISO FLUX_AUTO");
    println!("│                  A_IMAGE B_IMAGE THETA ELLIP KRON_RAD NPIX FLAGS");
    println!("└───────────────────────────────────────────────────────────────");
}

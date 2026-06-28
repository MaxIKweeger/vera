use std::path::Path;
use vera_fits::read_image_f32;
use vera_pipeline::background::{BackgroundConfig, BackgroundMap};
use vera_pipeline::detect::{detect, DetectConfig};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path_str = args.get(1).cloned().unwrap_or_else(|| {
        eprintln!("Usage: vera-detect <fichier.fits[.fz]> [sigma_psf] [thresh] [minarea]");
        std::process::exit(1);
    });
    let sigma_psf: f32  = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1.6);
    let thresh: f32     = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1.5);
    let minarea: usize  = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(5);

    let path = Path::new(&path_str);
    println!("Brick  : {}", path.display());

    let (image, wcs) = match read_image_f32(path) {
        Ok(v)  => v,
        Err(e) => { eprintln!("Erreur FITS : {e}"); std::process::exit(1); }
    };
    let (nrows, ncols) = image.dim();
    println!("Taille : {} x {} px\n", ncols, nrows);

    // Background
    let t = std::time::Instant::now();
    let bg_map = BackgroundMap::estimate(&image, &BackgroundConfig::default());
    let t_bg = t.elapsed();

    // Detection
    let config = DetectConfig { sigma_psf, thresh, minarea };
    let t = std::time::Instant::now();
    let result = detect(&image, &bg_map, &config);
    let t_det = t.elapsed();

    let dets = &result.detections;
    let n = dets.len();

    println!("┌── Détection ─────────────────────────────────────────────────");
    println!("│  σ_PSF={sigma_psf}px  seuil={thresh}σ  minarea={minarea}px");
    println!("│  Background : {t_bg:.1?}");
    println!("│  Détection  : {t_det:.1?}  (convolution + seuillage + labeling)");
    println!("│");
    println!("│  N sources  : {n}");

    if n > 0 {
        let mut sizes: Vec<u32> = dets.iter().map(|d| d.npix).collect();
        sizes.sort_unstable();
        let size_min    = sizes[0];
        let size_max    = *sizes.last().unwrap();
        let size_median = sizes[sizes.len() / 2];
        let size_mean   = sizes.iter().sum::<u32>() as f32 / n as f32;

        let snr_max  = dets[0].peak_snr;
        let snr_min  = dets.last().unwrap().peak_snr;
        let snr_mean = dets.iter().map(|d| d.peak_snr).sum::<f32>() / n as f32;

        println!("│  Taille (px): min={size_min}  max={size_max}  médiane={size_median}  moy={size_mean:.0}");
        println!("│  SNR peak   : min={snr_min:.1}  max={snr_max:.1}  moy={snr_mean:.1}");
        println!("│");
        println!("│  Top 10 sources (SNR décroissant) :");

        for (i, d) in dets.iter().take(10).enumerate() {
            let [r, c] = d.peak_pos;
            let coords = match &wcs {
                Some(w) => {
                    let (ra, dec) = w.pix_to_sky(c as f64, r as f64);
                    format!("RA={ra:.4}° Dec={dec:.4}°")
                }
                None => format!("px=({r},{c})"),
            };
            let bbox = d.bbox;
            let ext_r = bbox[1] - bbox[0] + 1;
            let ext_c = bbox[3] - bbox[2] + 1;
            println!(
                "│    #{:>3}  SNR={:>7.1}  npix={:>6}  bbox={}×{}  {}",
                i + 1, d.peak_snr, d.npix, ext_r, ext_c, coords
            );
        }
    }

    println!("└───────────────────────────────────────────────────────────────");
}

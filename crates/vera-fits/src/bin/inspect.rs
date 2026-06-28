use std::path::Path;
use vera_fits::read_image_f32;

fn main() {
    let path_str = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("Usage: vera-inspect <fichier.fits[.fz]>");
        std::process::exit(1);
    });

    let path = Path::new(&path_str);
    println!("Lecture : {}", path.display());

    let (img, wcs) = match read_image_f32(path) {
        Ok(v)  => v,
        Err(e) => { eprintln!("Erreur : {e}"); std::process::exit(1); }
    };

    let (nrows, ncols) = img.dim();
    let pixels = img.as_slice().unwrap();
    let valid: Vec<f32> = pixels.iter().copied().filter(|v| v.is_finite()).collect();

    let n_valid = valid.len();
    let n_bad   = pixels.len() - n_valid;
    let min     = valid.iter().cloned().fold(f32::INFINITY,     f32::min);
    let max     = valid.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean    = valid.iter().sum::<f32>() / n_valid as f32;

    let mut sorted = valid.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2];

    // Background estimate: median of central patch
    let cy = nrows / 2;
    let cx = ncols / 2;
    let patch = 100.min(nrows / 4);
    let bg_samples: Vec<f32> = (cy - patch..cy + patch)
        .flat_map(|r| (cx - patch..cx + patch).map(move |c| (r, c)))
        .map(|(r, c)| img[[r, c]])
        .filter(|v| v.is_finite())
        .collect();
    let mut bg = bg_samples.clone();
    bg.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let sky_median = bg.get(bg.len() / 2).copied().unwrap_or(f32::NAN);

    println!();
    println!("┌── Image ──────────────────────────────────────");
    println!("│  Taille   : {} x {} px  ({:.1} Mpx)", ncols, nrows, nrows * ncols / 1_000_000);
    println!("│  Valides  : {}  |  NaN/Inf : {}", n_valid, n_bad);
    println!("│  Flux     : min={min:.4}  max={max:.4}  mean={mean:.4}  median={median:.4}  (nanomaggies)");
    println!("│  Sky (patch centrale) : {sky_median:.4}");
    println!("│");

    if let Some(wcs) = &wcs {
        let (ra_c, dec_c) = wcs.pix_to_sky(ncols as f64 / 2.0, nrows as f64 / 2.0);
        let (ra_tl, _)    = wcs.pix_to_sky(0.0, 0.0);
        let (ra_br, _)    = wcs.pix_to_sky(ncols as f64, nrows as f64);
        let plate_scale   = (wcs.cd[0][0].hypot(wcs.cd[1][0])).abs() * 3600.0;
        let fov_deg       = (ra_br - ra_tl).abs();
        println!("│  WCS centre : RA={ra_c:.4}°  Dec={dec_c:.4}°");
        println!("│  FOV        : {fov_deg:.3}° x {fov_deg:.3}°");
        println!("│  Plate scale: {plate_scale:.4} arcsec/px");
    } else {
        println!("│  WCS : non disponible dans ce fichier");
    }

    println!("└───────────────────────────────────────────────");
}

//! vera-run — process all bricks in a directory and produce a unified catalog.
//!
//! Usage: vera-run <data-dir> [band] [output-stem]
//!
//! Finds all `legacysurvey-*-image-<band>.fits.fz` files in <data-dir>,
//! runs the full pipeline on each brick in parallel (Rayon), cross-matches
//! sources at brick boundaries and removes duplicates (< 1 arcsec), then
//! writes a combined FITS binary table + CSV.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use rayon::prelude::*;

use vera_catalog::{csv_write, fits_write};
use vera_fits::read_image_f32;
use vera_pipeline::background::{BackgroundConfig, BackgroundMap};
use vera_pipeline::detect::{detect, DetectConfig};
use vera_pipeline::measure::{measure_all, Measurement, MeasureConfig};

// ── Brick discovery ───────────────────────────────────────────────────────────

fn find_bricks(data_dir: &Path, band: &str) -> Vec<PathBuf> {
    let suffix = format!("-image-{band}.fits.fz");
    let mut paths: Vec<PathBuf> = std::fs::read_dir(data_dir)
        .expect("Cannot read data directory")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with("legacysurvey-") && n.ends_with(&suffix))
                .unwrap_or(false)
        })
        .collect();
    paths.sort();
    paths
}

fn brick_name(path: &Path) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .split('-')
        .nth(1)
        .unwrap_or("unknown")
        .to_string()
}

// ── Single-brick pipeline ─────────────────────────────────────────────────────

fn process_brick(path: &Path) -> Option<(String, Vec<Measurement>)> {
    let name = brick_name(path);
    let (image, wcs) = read_image_f32(path).ok()?;
    let bg   = BackgroundMap::estimate(&image, &BackgroundConfig::default());
    let det  = detect(&image, &bg, &DetectConfig::default());
    let meas = measure_all(&image, &bg, &det, wcs.as_ref(), &MeasureConfig::default());
    Some((name, meas))
}

// ── Cross-matching deduplication ──────────────────────────────────────────────
//
// Two sources from different bricks are considered the same object if their
// angular separation is < tolerance_arcsec.  We keep the detection with more
// pixels (better S/N proxy).  Algorithm: sort by RA, then slide a window of
// ΔRA < tol / cos(Dec) to find candidates in O(N log N) expected time.

fn deduplicate(entries: &mut Vec<(String, Measurement)>, tolerance_arcsec: f64) -> usize {
    // Sort by RA (NaN-aware: NaN entries go to the end and will be skipped).
    entries.sort_by(|(_, a), (_, b)| {
        a.ra.unwrap_or(f64::INFINITY)
            .partial_cmp(&b.ra.unwrap_or(f64::INFINITY))
            .unwrap()
    });

    let n = entries.len();
    let mut keep = vec![true; n];
    let tol_deg = tolerance_arcsec / 3600.0;

    for i in 0..n {
        if !keep[i] { continue; }
        let (brick_i, m_i) = &entries[i];
        let (ra_i, dec_i) = match (m_i.ra, m_i.dec) {
            (Some(ra), Some(dec)) => (ra, dec),
            _ => continue,
        };
        let cos_dec = dec_i.to_radians().cos().abs().max(0.01);
        let ra_win  = tol_deg / cos_dec; // ΔRA window (degrees)

        for j in (i + 1)..n {
            if !keep[j] { continue; }
            let (brick_j, m_j) = &entries[j];
            let ra_j = match m_j.ra { Some(v) => v, None => continue };

            if ra_j - ra_i > ra_win { break; } // outside RA window
            if brick_i == brick_j  { continue; } // same brick — not a duplicate

            let dec_j = match m_j.dec { Some(v) => v, None => continue };
            let d_ra  = (ra_j - ra_i) * cos_dec;
            let d_dec = dec_j - dec_i;
            let sep   = (d_ra * d_ra + d_dec * d_dec).sqrt() * 3600.0;

            if sep < tolerance_arcsec {
                // Keep the detection with the most pixels.
                if m_i.npix >= m_j.npix {
                    keep[j] = false;
                } else {
                    keep[i] = false;
                    break; // i is gone; move to i+1
                }
            }
        }
    }

    let removed = keep.iter().filter(|&&k| !k).count();
    let mut idx = 0;
    entries.retain(|_| { let k = keep[idx]; idx += 1; k });
    removed
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let data_dir = PathBuf::from(args.get(1).cloned().unwrap_or_else(|| {
        eprintln!("Usage: vera-run <data-dir> [band] [output-stem]");
        std::process::exit(1);
    }));
    let band       = args.get(2).cloned().unwrap_or_else(|| "r".into());
    let out_stem   = args.get(3).cloned().unwrap_or_else(|| "vera-virgo".into());
    let tol_arcsec = 1.0_f64; // cross-match tolerance

    let bricks = find_bricks(&data_dir, &band);
    if bricks.is_empty() {
        eprintln!("No bricks found in {}", data_dir.display());
        std::process::exit(1);
    }

    println!("┌── Vera multi-brick pipeline ───────────────────────────────────");
    println!("│  Data dir  : {}", data_dir.display());
    println!("│  Band      : {band}");
    println!("│  Bricks    : {}", bricks.len());
    println!("│  Threads   : {} (Rayon)", rayon::current_num_threads());
    println!("│  Dedup tol : {tol_arcsec}\"");
    println!("│");

    let counter = AtomicUsize::new(0);
    let total   = bricks.len();
    let t0      = Instant::now();

    // Parallel brick processing → collect per-brick results, then flatten.
    let brick_results: Vec<(String, Vec<Measurement>)> = bricks
        .par_iter()
        .filter_map(|path| {
            let t      = Instant::now();
            let result = process_brick(path);
            let n      = counter.fetch_add(1, Ordering::Relaxed) + 1;
            match &result {
                Some((brick, meas)) =>
                    eprintln!("  [{n:2}/{total}] {brick}  → {:5} sources  ({:.2?})",
                              meas.len(), t.elapsed()),
                None =>
                    eprintln!("  [{n:2}/{total}] {} FAILED", brick_name(path)),
            }
            result
        })
        .collect();

    let mut all_raw: Vec<(String, Measurement)> = brick_results
        .into_iter()
        .flat_map(|(brick, meas)| meas.into_iter().map(move |m| (brick.clone(), m)))
        .collect();

    let t_pipeline = t0.elapsed();
    let n_raw      = all_raw.len();

    println!("│");
    println!("│  Pipeline complet : {t_pipeline:.1?}");
    println!("│  Sources brutes   : {n_raw}");

    // Deduplication.
    let t1      = Instant::now();
    let removed = deduplicate(&mut all_raw, tol_arcsec);
    let t_dedup = t1.elapsed();

    let n_final = all_raw.len();
    println!("│  Doublons supprimés : {removed}  ({t_dedup:.1?})");
    println!("│  Sources finales    : {n_final}");

    // Sort final catalog by RA.
    all_raw.sort_by(|(_, a), (_, b)| {
        a.ra.unwrap_or(f64::INFINITY)
            .partial_cmp(&b.ra.unwrap_or(f64::INFINITY))
            .unwrap()
    });

    // Statistics.
    let fluxes: Vec<f32> = all_raw.iter().map(|(_, m)| m.flux_auto).filter(|v| v.is_finite()).collect();
    let (flux_med, flux_max) = if fluxes.is_empty() {
        (0.0f32, 0.0f32)
    } else {
        let mut s = fluxes.clone();
        s.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        (*s.get(s.len() / 2).unwrap_or(&0.0), *s.last().unwrap_or(&0.0))
    };

    // Write combined catalog.
    let fits_path = format!("{out_stem}.fits");
    let csv_path  = format!("{out_stem}.csv");

    let t2 = Instant::now();
    fits_write::write_multi(Path::new(&fits_path), &all_raw, &band)
        .unwrap_or_else(|e| eprintln!("FITS write error: {e}"));
    let t_fits = t2.elapsed();

    let t3 = Instant::now();
    csv_write::write_multi(Path::new(&csv_path), &all_raw, &band)
        .unwrap_or_else(|e| eprintln!("CSV write error: {e}"));
    let t_csv = t3.elapsed();

    let fits_kb = std::fs::metadata(&fits_path).map(|m| m.len()).unwrap_or(0) as f64 / 1024.0;
    let csv_kb  = std::fs::metadata(&csv_path).map(|m| m.len()).unwrap_or(0)  as f64 / 1024.0;

    println!("│");
    println!("│  Statistiques flux_auto (nanomaggies) :");
    println!("│    médiane = {flux_med:.4}   max = {flux_max:.1}");
    println!("│");
    println!("│  Fichiers écrits :");
    println!("│    {fits_path}  ({fits_kb:.0} kB, {t_fits:.1?})");
    println!("│    {csv_path}  ({csv_kb:.0} kB, {t_csv:.1?})");
    println!("│");
    println!("│  Total : {:.1?}", t0.elapsed());
    println!("└───────────────────────────────────────────────────────────────");
}

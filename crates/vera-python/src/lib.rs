use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use std::path::{Path, PathBuf};

use vera_fits::read_image_f32;
use vera_pipeline::background::{BackgroundConfig, BackgroundMap};
use vera_pipeline::detect::{detect, detect_gpu, DetectConfig};
use vera_pipeline::gpu::GpuContext;
use vera_pipeline::measure::{measure_all, measure_all_gpu, Measurement, MeasureConfig};

// ── Conversion ────────────────────────────────────────────────────────────────

fn to_pyobject(py: Python<'_>, m: &Measurement) -> Py<PyAny> {
    let d = PyDict::new(py);
    d.set_item("label",       m.label).unwrap();
    d.set_item("npix",        m.npix).unwrap();
    d.set_item("x",           m.x_c).unwrap();
    d.set_item("y",           m.y_c).unwrap();
    d.set_item("ra",          m.ra).unwrap();
    d.set_item("dec",         m.dec).unwrap();
    d.set_item("a",           m.a).unwrap();
    d.set_item("b",           m.b).unwrap();
    d.set_item("theta",       m.theta).unwrap();
    d.set_item("ellipticity", m.ellipticity).unwrap();
    d.set_item("kron_radius", m.kron_radius).unwrap();
    d.set_item("flux_iso",    m.flux_iso).unwrap();
    d.set_item("flux_auto",   m.flux_auto).unwrap();
    d.set_item("flags",       m.flags).unwrap();
    d.into_any().unbind()
}

// ── Pipeline helpers ──────────────────────────────────────────────────────────

/// Single brick — GPU conv + GPU Kron (full GPU path, suited for one image at a time).
fn pipeline_single(path: &Path, gpu: Option<&GpuContext>) -> Option<Vec<Measurement>> {
    let (image, wcs) = read_image_f32(path).ok()?;
    let bg  = BackgroundMap::estimate(&image, &BackgroundConfig::default());
    let det = match gpu {
        Some(ctx) => detect_gpu(&image, &bg, &DetectConfig::default(), ctx),
        None      => detect(&image, &bg, &DetectConfig::default()),
    };
    let meas = match gpu {
        Some(ctx) => measure_all_gpu(&image, &bg, &det, wcs.as_ref(), &MeasureConfig::default(), ctx),
        None      => measure_all(&image, &bg, &det, wcs.as_ref(), &MeasureConfig::default()),
    };
    Some(meas)
}

/// Multi-brick — GPU conv + CPU Kron to avoid GPU queue contention
/// when 20 Rayon threads submit simultaneously.
fn pipeline_parallel(path: &Path, gpu: Option<&GpuContext>) -> Option<Vec<Measurement>> {
    let (image, wcs) = read_image_f32(path).ok()?;
    let bg  = BackgroundMap::estimate(&image, &BackgroundConfig::default());
    let det = match gpu {
        Some(ctx) => detect_gpu(&image, &bg, &DetectConfig::default(), ctx),
        None      => detect(&image, &bg, &DetectConfig::default()),
    };
    Some(measure_all(&image, &bg, &det, wcs.as_ref(), &MeasureConfig::default()))
}

fn find_bricks(data_dir: &Path, band: &str) -> Vec<PathBuf> {
    let suffix = format!("-image-{band}.fits.fz");
    let mut paths: Vec<PathBuf> = std::fs::read_dir(data_dir)
        .unwrap_or_else(|e| panic!("Cannot read directory: {e}"))
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

fn deduplicate(entries: &mut Vec<Measurement>, tolerance_arcsec: f64) -> usize {
    entries.sort_by(|a, b| {
        a.ra.unwrap_or(f64::INFINITY)
            .partial_cmp(&b.ra.unwrap_or(f64::INFINITY))
            .unwrap()
    });
    let n = entries.len();
    let mut keep = vec![true; n];
    let tol_deg = tolerance_arcsec / 3600.0;
    for i in 0..n {
        if !keep[i] { continue; }
        let (ra_i, dec_i) = match (entries[i].ra, entries[i].dec) {
            (Some(ra), Some(dec)) => (ra, dec),
            _ => continue,
        };
        let cos_dec = dec_i.to_radians().cos().abs().max(0.01);
        let ra_win  = tol_deg / cos_dec;
        for j in (i + 1)..n {
            if !keep[j] { continue; }
            let ra_j = match entries[j].ra { Some(v) => v, None => continue };
            if ra_j - ra_i > ra_win { break; }
            let dec_j = match entries[j].dec { Some(v) => v, None => continue };
            let d_ra  = (ra_j - ra_i) * cos_dec;
            let d_dec = dec_j - dec_i;
            if (d_ra * d_ra + d_dec * d_dec).sqrt() * 3600.0 < tolerance_arcsec {
                if entries[i].npix >= entries[j].npix { keep[j] = false; }
                else { keep[i] = false; break; }
            }
        }
    }
    let removed = keep.iter().filter(|&&k| !k).count();
    let mut idx = 0;
    entries.retain(|_| { let k = keep[idx]; idx += 1; k });
    removed
}

// ── Public Python functions ───────────────────────────────────────────────────

/// Process a single FITS brick and return a list of source dicts.
///
/// Parameters
/// ----------
/// path : str
///     Path to a FITS or FITS.fz image.
///
/// Returns
/// -------
/// list[dict]
///     One dict per detected source.  Keys: label, npix, x, y, ra, dec,
///     a, b, theta, ellipticity, kron_radius, flux_iso, flux_auto, flags.
///
/// Examples
/// --------
/// >>> import vera, pandas as pd
/// >>> sources = vera.process_brick("legacysurvey-1877p122-image-r.fits.fz")
/// >>> df = pd.DataFrame(sources)
/// >>> print(df[["ra", "dec", "flux_auto"]].head())
#[pyfunction]
fn process_brick(py: Python<'_>, path: &str) -> PyResult<Vec<Py<PyAny>>> {
    let gpu  = GpuContext::new();
    let meas = pipeline_single(Path::new(path), gpu.as_ref())
        .ok_or_else(|| pyo3::exceptions::PyIOError::new_err(
            format!("Failed to process '{path}'")
        ))?;
    Ok(meas.iter().map(|m| to_pyobject(py, m)).collect())
}

/// Process all FITS bricks in a directory and return a merged, deduplicated
/// source catalogue (same logic as the `vera-run` binary).
///
/// Parameters
/// ----------
/// data_dir : str
///     Directory containing `legacysurvey-*-image-<band>.fits.fz` files.
/// band : str, optional
///     Photometric band (default: "r").
/// dedup_arcsec : float, optional
///     Cross-match tolerance for duplicate removal in arcseconds (default: 1.0).
///
/// Returns
/// -------
/// list[dict]
///     One dict per source.  Same keys as ``process_brick``.
///
/// Examples
/// --------
/// >>> import vera, pandas as pd
/// >>> sources = vera.run("./fits/", band="r")
/// >>> df = pd.DataFrame(sources)
/// >>> print(f"{len(df)} sources detected")
#[pyfunction]
#[pyo3(signature = (data_dir, band="r", dedup_arcsec=1.0))]
fn run(py: Python<'_>, data_dir: &str, band: &str, dedup_arcsec: f64) -> PyResult<Vec<Py<PyAny>>> {
    let dir    = Path::new(data_dir);
    let bricks = find_bricks(dir, band);
    if bricks.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("No bricks found in '{data_dir}' for band '{band}'")
        ));
    }
    let gpu = GpuContext::new();
    let mut all_meas: Vec<Measurement> = bricks.par_iter()
        .filter_map(|path| pipeline_parallel(path, gpu.as_ref()))
        .flatten()
        .collect();
    deduplicate(&mut all_meas, dedup_arcsec);
    Ok(all_meas.iter().map(|m| to_pyobject(py, m)).collect())
}

// ── Module ────────────────────────────────────────────────────────────────────

#[pymodule]
fn _vera(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(process_brick, m)?)?;
    m.add_function(wrap_pyfunction!(run, m)?)?;
    Ok(())
}

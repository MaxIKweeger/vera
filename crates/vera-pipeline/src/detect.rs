use ndarray::Array2;
use std::collections::HashMap;

use crate::background::BackgroundMap;
use crate::convolve::gaussian_smooth;

pub struct DetectConfig {
    /// PSF sigma in pixels. DECam r-band ~1" seeing / 0.262"/px ≈ 1.6px.
    pub sigma_psf: f32,
    /// Detection threshold in units of local RMS.
    pub thresh: f32,
    /// Minimum number of connected pixels to keep as a detection.
    pub minarea: usize,
}

impl Default for DetectConfig {
    fn default() -> Self {
        Self { sigma_psf: 1.6, thresh: 1.5, minarea: 5 }
    }
}

#[derive(Debug, Clone)]
pub struct Detection {
    pub label: u32,
    /// Number of pixels above threshold.
    pub npix: u32,
    /// Bounding box [row_min, row_max, col_min, col_max].
    pub bbox: [usize; 4],
    /// Peak SNR within this detection.
    pub peak_snr: f32,
    /// Pixel position of peak SNR [row, col].
    pub peak_pos: [usize; 2],
}

pub struct DetectionList {
    /// Detections sorted by peak SNR descending.
    pub detections: Vec<Detection>,
    /// Labeled image (0 = background, label >= 1 corresponds to detections vec index + 1).
    pub label_map: Array2<u32>,
}

/// Full detection pipeline: background subtraction → CPU convolution → threshold → label → filter.
pub fn detect(
    image: &Array2<f32>,
    bg_map: &BackgroundMap,
    config: &DetectConfig,
) -> DetectionList {
    let residual = bg_map.subtract(image);
    let rms_map  = bg_map.rms();
    let filtered = gaussian_smooth(&residual, config.sigma_psf);
    detect_from_filtered(&filtered, &rms_map, config)
}

/// Like `detect` but uses GPU Gaussian convolution.
pub fn detect_gpu(
    image: &Array2<f32>,
    bg_map: &BackgroundMap,
    config: &DetectConfig,
    ctx: &crate::gpu::GpuContext,
) -> DetectionList {
    let residual = bg_map.subtract(image);
    let rms_map  = bg_map.rms();
    let filtered = ctx.gaussian_smooth(&residual, config.sigma_psf);
    detect_from_filtered(&filtered, &rms_map, config)
}

pub fn detect_from_filtered(
    filtered: &Array2<f32>,
    rms_map: &Array2<f32>,
    config: &DetectConfig,
) -> DetectionList {
    let (nrows, ncols) = filtered.dim();

    let snr_map = Array2::from_shape_fn((nrows, ncols), |(r, c)| {
        let rms = rms_map[[r, c]];
        if rms > 0.0 { filtered[[r, c]] / rms } else { 0.0 }
    });

    let mask        = snr_map.mapv(|v| v > config.thresh);
    let label_map   = label_connected_components(&mask);
    let detections  = collect_detections(&label_map, &snr_map, config.minarea);

    DetectionList { detections, label_map }
}

// ── Connected-component labeling ──────────────────────────────────────────────

/// Two-pass connected-component labeling with 8-connectivity and Union-Find.
fn label_connected_components(mask: &Array2<bool>) -> Array2<u32> {
    let (nrows, ncols) = mask.dim();
    let mut labels: Array2<u32> = Array2::zeros((nrows, ncols));
    // parent[0] = 0 (background sentinel)
    let mut parent: Vec<u32> = vec![0];
    let mut next_label = 1u32;

    // First pass: assign provisional labels and record equivalences.
    for r in 0..nrows {
        for c in 0..ncols {
            if !mask[[r, c]] {
                continue;
            }

            // 8-connected neighbours that have already been visited.
            let mut nbr_labels = Vec::with_capacity(4);
            if r > 0 {
                push_if(&labels, r - 1, c,     &mut nbr_labels);
                if c > 0           { push_if(&labels, r - 1, c - 1, &mut nbr_labels); }
                if c + 1 < ncols   { push_if(&labels, r - 1, c + 1, &mut nbr_labels); }
            }
            if c > 0 { push_if(&labels, r, c - 1, &mut nbr_labels); }

            if nbr_labels.is_empty() {
                labels[[r, c]] = next_label;
                parent.push(next_label);
                next_label += 1;
            } else {
                // Find the minimum root among neighbours.
                let roots: Vec<u32> = nbr_labels.iter().map(|&l| find(&parent, l)).collect();
                let min_root = *roots.iter().min().unwrap();
                labels[[r, c]] = min_root;
                for &root in &roots {
                    if root != min_root {
                        union(&mut parent, root, min_root);
                    }
                }
            }
        }
    }

    // Flatten all chains to their root.
    let roots: Vec<u32> = (0..parent.len())
        .map(|i| find(&parent, i as u32))
        .collect();

    // Sequential relabeling: root → compact label.
    let mut remap = HashMap::<u32, u32>::new();
    let mut next = 1u32;
    for r in 0..nrows {
        for c in 0..ncols {
            let l = labels[[r, c]];
            if l == 0 {
                continue;
            }
            let root = roots[l as usize];
            let new_l = *remap.entry(root).or_insert_with(|| {
                let v = next;
                next += 1;
                v
            });
            labels[[r, c]] = new_l;
        }
    }

    labels
}

#[inline]
fn push_if(labels: &Array2<u32>, r: usize, c: usize, out: &mut Vec<u32>) {
    let l = labels[[r, c]];
    if l > 0 {
        out.push(l);
    }
}

fn find(parent: &[u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        x = parent[x as usize];
    }
    x
}

fn union(parent: &mut Vec<u32>, x: u32, y: u32) {
    let rx = find(parent, x);
    let ry = find(parent, y);
    if rx != ry {
        // Always point larger root to smaller root for determinism.
        if rx < ry {
            parent[ry as usize] = rx;
        } else {
            parent[rx as usize] = ry;
        }
    }
}

// ── Per-detection statistics ──────────────────────────────────────────────────

fn collect_detections(
    label_map: &Array2<u32>,
    snr_map: &Array2<f32>,
    minarea: usize,
) -> Vec<Detection> {
    let (nrows, ncols) = label_map.dim();
    let mut stats: HashMap<u32, Detection> = HashMap::new();

    for r in 0..nrows {
        for c in 0..ncols {
            let l = label_map[[r, c]];
            if l == 0 {
                continue;
            }
            let snr = snr_map[[r, c]];
            let e = stats.entry(l).or_insert(Detection {
                label: l,
                npix: 0,
                bbox: [r, r, c, c],
                peak_snr: f32::NEG_INFINITY,
                peak_pos: [r, c],
            });
            e.npix += 1;
            e.bbox[0] = e.bbox[0].min(r);
            e.bbox[1] = e.bbox[1].max(r);
            e.bbox[2] = e.bbox[2].min(c);
            e.bbox[3] = e.bbox[3].max(c);
            if snr > e.peak_snr {
                e.peak_snr = snr;
                e.peak_pos = [r, c];
            }
        }
    }

    let mut result: Vec<Detection> = stats
        .into_values()
        .filter(|d| d.npix >= minarea as u32)
        .collect();
    result.sort_by(|a, b| b.peak_snr.partial_cmp(&a.peak_snr).unwrap());
    result
}

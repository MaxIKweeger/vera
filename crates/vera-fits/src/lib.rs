use fitsrs::{Fits, HDU};
use fitsrs::hdu::data::image::Pixels;
use fitsrs::hdu::data::bintable::data::BinaryTableData;
use fitsrs::hdu::data::bintable::tile_compressed::pixels::Pixels as TilePixels;
use ndarray::Array2;
use thiserror::Error;
use std::io::{BufReader, Read, Seek};
use std::path::Path;
use std::fs::File;

#[derive(Debug, Error)]
pub enum FitsError {
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("FITS parse: {0}")]
    Fits(String),
    #[error("no image data found")]
    NoImage,
    #[error("unexpected shape: {0:?}")]
    BadShape(Vec<usize>),
}

pub type Result<T> = std::result::Result<T, FitsError>;

/// Minimal WCS for pixel <-> sky conversion (linear approximation, good for < 1 deg).
#[derive(Debug, Clone)]
pub struct WcsHeader {
    pub crpix: [f64; 2],
    pub crval: [f64; 2],
    /// CD matrix [[cd1_1, cd1_2], [cd2_1, cd2_2]] in degrees/pixel.
    pub cd: [[f64; 2]; 2],
}

impl WcsHeader {
    /// 0-indexed (col, row) -> (RA, Dec) degrees.
    pub fn pix_to_sky(&self, col: f64, row: f64) -> (f64, f64) {
        let dx = col + 1.0 - self.crpix[0];
        let dy = row + 1.0 - self.crpix[1];
        let ra  = self.crval[0] + self.cd[0][0] * dx + self.cd[0][1] * dy;
        let dec = self.crval[1] + self.cd[1][0] * dx + self.cd[1][1] * dy;
        (ra, dec)
    }
}

/// A single-band brick: science image + optional calibration arrays.
pub struct FitsBrick {
    pub image: Array2<f32>,
    pub invvar: Option<Array2<f32>>,
    pub maskbits: Option<Array2<u32>>,
    pub wcs: Option<WcsHeader>,
    /// E.g. "1877p122"
    pub brick: String,
}

/// Read a 2-D float image from a FITS file.
/// Handles both plain FITS and RICE-compressed .fits.fz transparently.
pub fn read_image_f32(path: &Path) -> Result<(Array2<f32>, Option<WcsHeader>)> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    read_image_from_reader(reader)
}

fn read_image_from_reader<R>(reader: R) -> Result<(Array2<f32>, Option<WcsHeader>)>
where
    R: Read + Seek + std::fmt::Debug + 'static,
{
    let mut hdu_list = Fits::from_reader(reader);

    while let Some(hdu_result) = hdu_list.next() {
        let hdu = hdu_result.map_err(|e| FitsError::Fits(e.to_string()))?;

        match hdu {
            // --- Plain image (Primary or IMAGE extension) ---
            HDU::Primary(ref h) | HDU::XImage(ref h) => {
                let xtension = h.get_header().get_xtension();
                let naxis = xtension.get_naxis();
                if naxis.len() < 2 {
                    continue;
                }
                let ncols = naxis[0] as usize;
                let nrows = naxis[1] as usize;
                if ncols == 0 || nrows == 0 {
                    continue;
                }

                let wcs = extract_wcs_image(h.get_header());
                let image_data = hdu_list.get_data(h);
                let flat: Vec<f32> = match image_data.pixels() {
                    Pixels::F32(it) => it.collect(),
                    Pixels::I16(it) => it.map(|v| v as f32).collect(),
                    Pixels::I32(it) => it.map(|v| v as f32).collect(),
                    Pixels::F64(it) => it.map(|v| v as f32).collect(),
                    Pixels::U8(it)  => it.map(|v| v as f32).collect(),
                    Pixels::I64(it) => it.map(|v| v as f32).collect(),
                };
                let arr = Array2::from_shape_vec((nrows, ncols), flat)
                    .map_err(|_| FitsError::BadShape(vec![nrows, ncols]))?;
                return Ok((arr, wcs));
            }

            // --- Tile-compressed image (.fits.fz, RICE) stored as binary table ---
            HDU::XBinaryTable(ref h) => {
                let header = h.get_header();

                // ZNAXIS1/ZNAXIS2 hold the uncompressed image dimensions.
                let ncols = match header.get_parsed::<i64>("ZNAXIS1") {
                    Ok(v) if v > 0 => v as usize,
                    _ => continue,
                };
                let nrows = match header.get_parsed::<i64>("ZNAXIS2") {
                    Ok(v) if v > 0 => v as usize,
                    _ => continue,
                };

                let wcs = extract_wcs_bintable(header);
                let data = hdu_list.get_data(h);

                let flat: Vec<f32> = match data {
                    BinaryTableData::TileCompressed(pixels) => match pixels {
                        TilePixels::F32(it) => it.collect(),
                        // F64 iterator yields f32 (dequantized)
                        TilePixels::F64(it) => it.collect(),
                        TilePixels::I32(it) => it.map(|v| v as f32).collect(),
                        TilePixels::I16(it) => it.map(|v| v as f32).collect(),
                        TilePixels::U8(it)  => it.map(|v| v as f32).collect(),
                    },
                    BinaryTableData::Table(_) => continue,
                };

                if flat.len() != nrows * ncols {
                    return Err(FitsError::BadShape(vec![flat.len(), nrows, ncols]));
                }
                let arr = Array2::from_shape_vec((nrows, ncols), flat)
                    .map_err(|_| FitsError::BadShape(vec![nrows, ncols]))?;
                return Ok((arr, wcs));
            }

            HDU::XASCIITable(_) => continue,
        }
    }

    Err(FitsError::NoImage)
}

// ── WCS helpers ──────────────────────────────────────────────────────────────

use fitsrs::hdu::header::extension::image::Image;
use fitsrs::hdu::header::extension::bintable::BinTable;
use fitsrs::hdu::header::Header;

fn extract_wcs_image(h: &Header<Image>) -> Option<WcsHeader> {
    let crpix1 = h.get_parsed::<f64>("CRPIX1").ok()?;
    let crpix2 = h.get_parsed::<f64>("CRPIX2").ok()?;
    let crval1 = h.get_parsed::<f64>("CRVAL1").ok()?;
    let crval2 = h.get_parsed::<f64>("CRVAL2").ok()?;
    let cd1_1  = h.get_parsed::<f64>("CD1_1").ok()?;
    let cd1_2  = h.get_parsed::<f64>("CD1_2").ok()?;
    let cd2_1  = h.get_parsed::<f64>("CD2_1").ok()?;
    let cd2_2  = h.get_parsed::<f64>("CD2_2").ok()?;
    Some(WcsHeader {
        crpix: [crpix1, crpix2],
        crval: [crval1, crval2],
        cd: [[cd1_1, cd1_2], [cd2_1, cd2_2]],
    })
}

fn extract_wcs_bintable(h: &Header<BinTable>) -> Option<WcsHeader> {
    let crpix1 = h.get_parsed::<f64>("CRPIX1").ok()?;
    let crpix2 = h.get_parsed::<f64>("CRPIX2").ok()?;
    let crval1 = h.get_parsed::<f64>("CRVAL1").ok()?;
    let crval2 = h.get_parsed::<f64>("CRVAL2").ok()?;
    let cd1_1  = h.get_parsed::<f64>("CD1_1").ok()?;
    let cd1_2  = h.get_parsed::<f64>("CD1_2").ok()?;
    let cd2_1  = h.get_parsed::<f64>("CD2_1").ok()?;
    let cd2_2  = h.get_parsed::<f64>("CD2_2").ok()?;
    Some(WcsHeader {
        crpix: [crpix1, crpix2],
        crval: [crval1, crval2],
        cd: [[cd1_1, cd1_2], [cd2_1, cd2_2]],
    })
}

//! Pure-Rust FITS binary table writer (no cfitsio dependency).
//!
//! Produces a 2-HDU file: empty primary + BINTABLE extension.
//! All integers/floats are written big-endian as required by the FITS standard.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use vera_pipeline::measure::Measurement;

const BLOCK: usize = 2880;

// ── Column schema ─────────────────────────────────────────────────────────────

struct Col {
    name: &'static str,
    tform: &'static str,
    tunit: &'static str,
    nbytes: usize,
}

fn schema() -> Vec<Col> {
    vec![
        Col { name: "RA",        tform: "1D", tunit: "deg", nbytes: 8 },
        Col { name: "DEC",       tform: "1D", tunit: "deg", nbytes: 8 },
        Col { name: "X_IMAGE",   tform: "1D", tunit: "pix", nbytes: 8 },
        Col { name: "Y_IMAGE",   tform: "1D", tunit: "pix", nbytes: 8 },
        Col { name: "FLUX_ISO",  tform: "1E", tunit: "nmg", nbytes: 4 },
        Col { name: "FLUX_AUTO", tform: "1E", tunit: "nmg", nbytes: 4 },
        Col { name: "A_IMAGE",   tform: "1E", tunit: "pix", nbytes: 4 },
        Col { name: "B_IMAGE",   tform: "1E", tunit: "pix", nbytes: 4 },
        Col { name: "THETA",     tform: "1E", tunit: "deg", nbytes: 4 },
        Col { name: "ELLIP",     tform: "1E", tunit: "",    nbytes: 4 },
        Col { name: "KRON_RAD",  tform: "1E", tunit: "",    nbytes: 4 },
        Col { name: "NPIX",      tform: "1J", tunit: "pix", nbytes: 4 },
        Col { name: "FLAGS",     tform: "1I", tunit: "",    nbytes: 2 },
    ]
}

// ── Public entry points ───────────────────────────────────────────────────────

pub fn write(
    path: &Path,
    measurements: &[Measurement],
    brick: &str,
    band: &str,
) -> io::Result<()> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);
    write_primary(&mut w, brick, band)?;
    write_bintable(&mut w, measurements, brick, band)?;
    Ok(())
}

/// Multi-brick combined catalog — adds a BRICK column (12-char ASCII).
pub fn write_multi(
    path: &Path,
    entries: &[(String, Measurement)],
    band: &str,
) -> io::Result<()> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);
    write_primary_multi(&mut w, band)?;
    write_bintable_multi(&mut w, entries, band)?;
    Ok(())
}

fn write_primary_multi(w: &mut impl Write, band: &str) -> io::Result<()> {
    let cards = vec![
        kv_bool("SIMPLE",  true,  "Standard FITS"),
        kv_int ("BITPIX",  8,     "Character data"),
        kv_int ("NAXIS",   0,     "No image"),
        kv_bool("EXTEND",  true,  "Extensions allowed"),
        kv_str ("ORIGIN",  "vera","github.com/MaxIKweeger/vera"),
        kv_str ("TELESCOP","DECam","Dark Energy Camera"),
        kv_str ("FILTER",  band,  "Photometric band"),
        kv_str ("AUTHOR",  "MaxKweeger",     "Scientific director"),
        kv_str ("SOFTWARE","vera / Claude Sonnet (Anthropic)", "AI-assisted"),
        kv_str ("REFERENC","joss.theoj.org", "Target journal: JOSS"),
    ];
    write_header_block(w, &cards)
}

fn write_bintable_multi(
    w: &mut impl Write,
    entries: &[(String, Measurement)],
    band: &str,
) -> io::Result<()> {
    const BRICK_BYTES: usize = 12;
    let per_row: usize = BRICK_BYTES + schema().iter().map(|c| c.nbytes).sum::<usize>();
    let naxis2 = entries.len();
    let ncols  = 1 + schema().len();

    let mut cards = vec![
        kv_str ("XTENSION", "BINTABLE", "Binary table extension"),
        kv_int ("BITPIX",   8,          ""),
        kv_int ("NAXIS",    2,          "2-D table"),
        kv_int ("NAXIS1",   per_row as i64, "Bytes per row"),
        kv_int ("NAXIS2",   naxis2 as i64,  "Number of rows"),
        kv_int ("PCOUNT",   0,          "No heap"),
        kv_int ("GCOUNT",   1,          ""),
        kv_int ("TFIELDS",  ncols as i64,   "Number of columns"),
        kv_str ("FILTER",   band,       "Photometric band"),
        kv_str ("AUTHOR",   "MaxKweeger",    "Scientific director"),
        kv_str ("SOFTWARE", "vera / Claude Sonnet (Anthropic)", "AI-assisted"),
        kv_str ("REFERENC", "joss.theoj.org","Target journal: JOSS"),
        // BRICK column (1)
        kv_str ("TTYPE1",   "BRICK",    "Brick identifier"),
        kv_str ("TFORM1",   "12A",      "12-char ASCII string"),
    ];
    for (i, col) in schema().iter().enumerate() {
        let n = i + 2; // offset by 1 for BRICK
        cards.push(kv_str(&format!("TTYPE{n}"), col.name,  "Column name"));
        cards.push(kv_str(&format!("TFORM{n}"), col.tform, "Data format"));
        if !col.tunit.is_empty() {
            cards.push(kv_str(&format!("TUNIT{n}"), col.tunit, "Unit"));
        }
    }
    write_header_block(w, &cards)?;

    let mut data: Vec<u8> = Vec::with_capacity(per_row * naxis2);
    for (brick, m) in entries {
        // BRICK column: 12 bytes, space-padded
        let bk = brick.as_bytes();
        let blen = bk.len().min(BRICK_BYTES);
        data.extend_from_slice(&bk[..blen]);
        for _ in blen..BRICK_BYTES { data.push(b' '); }
        // Measurement columns (same as single-brick writer)
        data.extend_from_slice(&m.ra.unwrap_or(f64::NAN).to_be_bytes());
        data.extend_from_slice(&m.dec.unwrap_or(f64::NAN).to_be_bytes());
        data.extend_from_slice(&m.x_c.to_be_bytes());
        data.extend_from_slice(&m.y_c.to_be_bytes());
        data.extend_from_slice(&m.flux_iso.to_be_bytes());
        data.extend_from_slice(&m.flux_auto.to_be_bytes());
        data.extend_from_slice(&m.a.to_be_bytes());
        data.extend_from_slice(&m.b.to_be_bytes());
        data.extend_from_slice(&m.theta.to_be_bytes());
        data.extend_from_slice(&m.ellipticity.to_be_bytes());
        data.extend_from_slice(&m.kron_radius.to_be_bytes());
        data.extend_from_slice(&(m.npix as i32).to_be_bytes());
        data.extend_from_slice(&(m.flags as i16).to_be_bytes());
    }
    let rem = data.len() % BLOCK;
    if rem != 0 { data.extend(std::iter::repeat(0u8).take(BLOCK - rem)); }
    w.write_all(&data)
}

// ── Primary HDU (empty image) ─────────────────────────────────────────────────

fn write_primary(w: &mut impl Write, brick: &str, band: &str) -> io::Result<()> {
    let cards = vec![
        kv_bool("SIMPLE",  true,  "Standard FITS"),
        kv_int ("BITPIX",  8,     "Character data"),
        kv_int ("NAXIS",   0,     "No image"),
        kv_bool("EXTEND",  true,  "Extensions allowed"),
        kv_str ("ORIGIN",  "vera","github.com/MaxIKweeger/vera"),
        kv_str ("TELESCOP","DECam","Dark Energy Camera"),
        kv_str ("FILTER",  band,  "Photometric band"),
        kv_str ("OBJECT",  brick, "Brick identifier"),
        kv_str ("AUTHOR",  "MaxKweeger", "Scientific director"),
        kv_str ("SOFTWARE","vera / Claude Sonnet (Anthropic)", "AI-assisted"),
    ];
    write_header_block(w, &cards)
}

// ── BINTABLE HDU ──────────────────────────────────────────────────────────────

fn write_bintable(
    w: &mut impl Write,
    rows: &[Measurement],
    brick: &str,
    band: &str,
) -> io::Result<()> {
    let cols = schema();
    let naxis1: usize = cols.iter().map(|c| c.nbytes).sum();
    let naxis2 = rows.len();
    let tfields = cols.len();

    let mut cards = vec![
        kv_str ("XTENSION", "BINTABLE", "Binary table extension"),
        kv_int ("BITPIX",   8,          ""),
        kv_int ("NAXIS",    2,          "2-D table"),
        kv_int ("NAXIS1",   naxis1 as i64, "Bytes per row"),
        kv_int ("NAXIS2",   naxis2 as i64, "Number of rows"),
        kv_int ("PCOUNT",   0,          "No heap"),
        kv_int ("GCOUNT",   1,          ""),
        kv_int ("TFIELDS",  tfields as i64, "Number of columns"),
        kv_str ("OBJECT",   brick,      "Brick identifier"),
        kv_str ("FILTER",   band,       "Photometric band"),
        kv_str ("AUTHOR",   "MaxKweeger", "Scientific director"),
        kv_str ("SOFTWARE", "vera / Claude Sonnet (Anthropic)", "AI-assisted"),
        kv_str ("REFERENC", "joss.theoj.org", "Target journal: JOSS"),
    ];
    for (i, col) in cols.iter().enumerate() {
        let n = i + 1;
        cards.push(kv_str(&format!("TTYPE{n}"), col.name,  "Column name"));
        cards.push(kv_str(&format!("TFORM{n}"), col.tform, "Data format"));
        if !col.tunit.is_empty() {
            cards.push(kv_str(&format!("TUNIT{n}"), col.tunit, "Unit"));
        }
    }
    write_header_block(w, &cards)?;

    // Data block
    let mut data: Vec<u8> = Vec::with_capacity(naxis1 * naxis2);
    for m in rows {
        data.extend_from_slice(&m.ra.unwrap_or(f64::NAN).to_be_bytes());
        data.extend_from_slice(&m.dec.unwrap_or(f64::NAN).to_be_bytes());
        data.extend_from_slice(&m.x_c.to_be_bytes());
        data.extend_from_slice(&m.y_c.to_be_bytes());
        data.extend_from_slice(&m.flux_iso.to_be_bytes());
        data.extend_from_slice(&m.flux_auto.to_be_bytes());
        data.extend_from_slice(&m.a.to_be_bytes());
        data.extend_from_slice(&m.b.to_be_bytes());
        data.extend_from_slice(&m.theta.to_be_bytes());
        data.extend_from_slice(&m.ellipticity.to_be_bytes());
        data.extend_from_slice(&m.kron_radius.to_be_bytes());
        data.extend_from_slice(&(m.npix as i32).to_be_bytes());
        data.extend_from_slice(&(m.flags as i16).to_be_bytes());
    }
    // Pad data to 2880-byte multiple
    let rem = data.len() % BLOCK;
    if rem != 0 {
        data.extend(std::iter::repeat(0u8).take(BLOCK - rem));
    }
    w.write_all(&data)
}

// ── Header helpers ────────────────────────────────────────────────────────────

fn write_header_block(w: &mut impl Write, cards: &[String]) -> io::Result<()> {
    let mut buf: Vec<u8> = Vec::new();
    for text in cards {
        buf.extend_from_slice(&card(text));
    }
    // END card
    buf.extend_from_slice(&card("END"));
    // Pad to 2880-byte multiple with spaces
    let rem = buf.len() % BLOCK;
    if rem != 0 {
        buf.extend(std::iter::repeat(b' ').take(BLOCK - rem));
    }
    w.write_all(&buf)
}

fn card(text: &str) -> [u8; 80] {
    let mut c = [b' '; 80];
    for (i, b) in text.bytes().take(80).enumerate() {
        c[i] = b;
    }
    c
}

fn kv_bool(key: &str, val: bool, comment: &str) -> String {
    let v = if val { "T" } else { "F" };
    kv_raw(key, &format!("{v:>20}"), comment)
}

fn kv_int(key: &str, val: i64, comment: &str) -> String {
    kv_raw(key, &format!("{val:>20}"), comment)
}

fn kv_str(key: &str, val: &str, comment: &str) -> String {
    // String value: single-quoted, left-justified, at least 8 chars wide inside quotes.
    let padded = format!("{val:<8}");
    let padded = if val.len() > 8 { val.to_string() } else { padded };
    kv_raw(key, &format!("'{padded}'"), comment)
}

fn kv_raw(key: &str, value: &str, comment: &str) -> String {
    if comment.is_empty() {
        format!("{key:<8}= {value}")
    } else {
        format!("{key:<8}= {value:<20} / {comment}")
    }
}

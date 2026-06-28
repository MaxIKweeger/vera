use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use vera_pipeline::measure::Measurement;

pub fn write(path: &Path, measurements: &[Measurement], brick: &str, band: &str) -> io::Result<()> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    writeln!(w, "# Vera source catalog — brick={brick} band={band}")?;
    writeln!(w, "# Author: MaxKweeger | Software: vera / Claude Sonnet (Anthropic)")?;
    writeln!(w, "# Target journal: Journal of Open Source Software (joss.theoj.org)")?;
    writeln!(
        w,
        "ra,dec,x_image,y_image,flux_iso,flux_auto,a_image,b_image,theta,ellip,kron_rad,npix,flags"
    )?;

    for m in measurements {
        writeln!(
            w,
            "{},{},{:.4},{:.4},{:.6},{:.6},{:.3},{:.3},{:.2},{:.4},{:.3},{},{}",
            m.ra.map(|v| format!("{v:.7}")).unwrap_or_else(|| "nan".into()),
            m.dec.map(|v| format!("{v:.7}")).unwrap_or_else(|| "nan".into()),
            m.x_c,
            m.y_c,
            m.flux_iso,
            m.flux_auto,
            m.a,
            m.b,
            m.theta,
            m.ellipticity,
            m.kron_radius,
            m.npix,
            m.flags,
        )?;
    }

    Ok(())
}

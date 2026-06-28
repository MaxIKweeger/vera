"""
Vera — GPU-native astronomical source extraction, Python bindings.

Install
-------
    pip install maturin
    cd crates/vera-python
    maturin develop --release   # development install (editable)
    # or: maturin build --release && pip install target/wheels/*.whl

Quick start
-----------
    import vera

    # Single FITS brick
    sources = vera.process_brick("legacysurvey-1877p122-image-r.fits.fz")

    # Full directory (28 bricks, parallel, deduplicated)
    sources = vera.run("./fits/", band="r")

    # Convert to pandas DataFrame
    import pandas as pd
    df = pd.DataFrame(sources)
    print(df[["ra", "dec", "flux_auto", "a", "b"]].head(10))

    # Convert to astropy Table
    from astropy.table import Table
    t = Table(rows=sources)
    t["flux_auto"].unit = "nanomaggy"

Column schema
-------------
    label        — internal detection label (int)
    npix         — number of pixels in isophote (int)
    x, y         — flux-weighted centroid, 0-indexed pixels (float)
    ra, dec      — sky coordinates in degrees, WCS TAN projection (float or None)
    a, b         — semi-major / semi-minor axis in pixels (float)
    theta        — position angle in degrees, CCW from x-axis (float)
    ellipticity  — 1 − b/a (float)
    kron_radius  — Kron radius in units of ellipse scale (float)
    flux_iso     — isophotal flux in nanomaggies (float)
    flux_auto    — Kron elliptical aperture flux in nanomaggies (float)
    flags        — bitmask: 0x01 = edge truncation, 0x04 = saturated (int)
"""

from ._vera import process_brick, run

__all__ = ["process_brick", "run"]
__version__ = "0.1.0"

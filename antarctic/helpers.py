"""Reconstructions of the helper functions referenced but never delivered.

``Ann_block_ave`` and ``exportHybridMap`` are called by
``Analyse_PCA_Antarctica.m`` but are not in the delivery. Both are rebuilt here
from how they are used.
"""

from __future__ import annotations

import numpy as np


def ann_block_ave(t_decimal_year, y, min_count=1):
    """Annual block average of an irregular/monthly series.

    Reconstruction of ``[xd, yd] = Ann_block_ave(x, y)``. Every call site
    passes a decimal-year time axis and a monthly series, then immediately
    filters ``xd`` with year comparisons and fits ``polyfit(xd, yd, 1)``, so
    ``xd`` must come back as whole years and ``yd`` as the mean within each.

    Parameters
    ----------
    t_decimal_year : array_like
        Time in decimal years.
    y : array_like
        Values, same length.
    min_count : int
        Years with fewer than this many samples yield NaN.

    Returns
    -------
    (years, means) : both 1-D ndarrays, one entry per calendar year present.
    """
    t = np.asarray(t_decimal_year, dtype=float).ravel()
    v = np.asarray(y, dtype=float).ravel()
    if t.size != v.size:
        raise ValueError(f"length mismatch: {t.size} vs {v.size}")

    yr = np.floor(t).astype(int)
    years = np.unique(yr[np.isfinite(t)])
    means = np.full(years.shape, np.nan)
    for i, Y in enumerate(years):
        sel = (yr == Y) & np.isfinite(v)
        if sel.sum() >= min_count:
            means[i] = v[sel].mean()
    return years.astype(float), means


def matlab_datenum_to_decimal_year(dt64):
    """Decimal year from numpy datetime64, the way the MATLAB *meant* to.

    The original computes ``datenum(Date)./365`` in one place and
    ``datenum(Date)./365.25`` in another. MATLAB's ``datenum`` counts days from
    year 0, so dividing by 365.25 lands within a fraction of a year of the
    calendar year, whereas dividing by 365 is off by about +1.5 years. See
    ``scripts/analyse_pca_antarctica.py`` for how that is handled.
    """
    dt64 = np.asarray(dt64, dtype="datetime64[D]")
    year = dt64.astype("datetime64[Y]").astype(int) + 1970
    start = dt64.astype("datetime64[Y]").astype("datetime64[D]")
    nxt = (dt64.astype("datetime64[Y]") + 1).astype("datetime64[D]")
    frac = (dt64 - start).astype(float) / (nxt - start).astype(float)
    return year + frac


def matlab_datenum(dt64):
    """MATLAB ``datenum`` (days since year 0) from numpy datetime64."""
    dt64 = np.asarray(dt64, dtype="datetime64[D]")
    # datenum(1970,1,1) == 719529
    return dt64.astype("datetime64[D]").astype(float) + 719529.0


def linfit_with_p(x, y):
    """Least-squares slope/intercept plus the correlation p-value.

    Mirrors the ``polyfit`` + ``corrcoef`` pair used throughout the MATLAB:
    the reported p-value is the two-sided p for Pearson r between x and y,
    which for a simple linear fit is the same as the p-value on the slope.

    Returns
    -------
    slope, intercept, pvalue, n
        All NaN (n possibly 0) if fewer than 3 finite pairs.
    """
    from scipy import stats

    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = x.size
    if n < 3 or np.ptp(x) == 0:
        return np.nan, np.nan, np.nan, n
    res = stats.linregress(x, y)
    return res.slope, res.intercept, res.pvalue, n


def export_hybrid_map(fig, path, raster_axes=None, dpi=600):
    """Reconstruction of ``exportHybridMap``.

    The MATLAB call is ``exportHybridMap(Z, latlims, lonlims, overlayFcn,
    'antarctic_map.pdf', mapsettings)`` and the file's own cells (lines
    232-282 and 318-345) show what it was for: rasterise the heavy filled
    field so the PDF stays small, keep coastlines, scatter and text as true
    vectors on top.

    Matplotlib does this natively with per-artist ``set_rasterized``, so this
    helper just rasterises the pcolormesh/contourf artists in the given axes
    and writes a vector PDF.

    Parameters
    ----------
    fig : matplotlib Figure
    path : output path (.pdf)
    raster_axes : list of Axes, or None for all axes in the figure
    dpi : resolution of the rasterised layer only
    """
    from matplotlib.collections import QuadMesh, PathCollection
    from matplotlib.image import AxesImage

    axes = fig.axes if raster_axes is None else raster_axes
    for ax in axes:
        for arts in (ax.collections, ax.images):
            for a in arts:
                # Rasterise filled fields; leave scatter (PathCollection) vector.
                if isinstance(a, (QuadMesh, AxesImage)) and not isinstance(a, PathCollection):
                    a.set_rasterized(True)
    fig.savefig(path, format="pdf", dpi=dpi, bbox_inches="tight")
    return path

"""Trend, neighbour-merging and PCA machinery for ``Analyse_PCA_Antarctica.m``."""

from __future__ import annotations

import numpy as np
from scipy import stats


# ------------------------------------------------------------- trends ------
def trend_1d(x, y, year_start=None, year_stop=None):
    """Slope, p-value and n for one series, with the MATLAB's windowing.

    The original pattern is, every time:

        y = y(x >= Year_start & x < Year_stop);   % note: stop is exclusive
        x = x(...); x = x(~isnan(y)); y = y(~isnan(y));
        p = polyfit(x, y, 1); [r, P] = corrcoef(x, y);

    so the window is half-open and NaNs are dropped pairwise.
    """
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    if x.size != y.size:
        n = min(x.size, y.size)
        x, y = x[:n], y[:n]
    if year_start is not None:
        sel = (x >= year_start) & (x < year_stop)
        x, y = x[sel], y[sel]
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3 or np.ptp(x) == 0:
        return np.nan, np.nan, x.size
    r = stats.linregress(x, y)
    return r.slope, r.pvalue, x.size


def trend_map(years, cube, year_start, year_stop, min_years=10):
    """Vectorised per-gridpoint linear trend.

    Parameters
    ----------
    years : (T,) array of calendar years
    cube  : (T, ...) array
    year_start, year_stop : half-open window [start, stop)
    min_years : gridpoints with fewer valid years give NaN

    Returns
    -------
    slope, pvalue : arrays shaped like ``cube.shape[1:]``, slope in units/year
    """
    years = np.asarray(years, float)
    cube = np.asarray(cube, float)
    sel = (years >= year_start) & (years < year_stop)
    x = years[sel]
    Y = cube[sel]

    shape = Y.shape[1:]
    Y2 = Y.reshape(len(x), -1)

    good = np.isfinite(Y2)
    n = good.sum(axis=0)

    # Masked sums so partially-missing columns still work.
    Yf = np.where(good, Y2, 0.0)
    X = np.repeat(x[:, None], Y2.shape[1], axis=1)
    Xf = np.where(good, X, 0.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        sx = Xf.sum(axis=0)
        sy = Yf.sum(axis=0)
        sxx = (Xf * Xf).sum(axis=0)
        sxy = (Xf * Yf).sum(axis=0)
        syy = (Yf * Yf).sum(axis=0)

        denom = n * sxx - sx * sx
        slope = (n * sxy - sx * sy) / denom

        # Pearson r -> two-sided p via the t distribution, matching corrcoef.
        rnum = n * sxy - sx * sy
        rden = np.sqrt(denom * (n * syy - sy * sy))
        r = np.clip(rnum / rden, -1.0, 1.0)
        with np.errstate(all="ignore"):
            t = r * np.sqrt((n - 2) / np.maximum(1e-12, 1 - r * r))
        p = 2.0 * stats.t.sf(np.abs(t), np.maximum(n - 2, 1))

    bad = (n < min_years) | ~np.isfinite(denom) | (denom == 0)
    slope = np.where(bad, np.nan, slope)
    p = np.where(bad, np.nan, p)
    return slope.reshape(shape), p.reshape(shape)


# --------------------------------------------------- ice-core merging ------
def haversine_km(lat1, lon1, lat2, lon2, radius=6371.0088):
    """Great-circle distance in km.

    The MATLAB uses ``distance(..., wgs84Ellipsoid('km'))``, i.e. geodesic on
    WGS84. The spherical approximation differs by well under 0.5%, far below
    the 200 km grouping threshold, so it changes no grouping decision here.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def merge_neighbours(age, values, lat, lon, threshold_km=200.0):
    """Greedy spatial merge of ice-core records, as in the MATLAB.

    Walks the records in order; for each still-unassigned record it collects
    every unassigned record within ``threshold_km``, interpolates them all onto
    the union of their age axes (no extrapolation) and averages with
    ``omitnan``. The group takes the coordinates of its *seed* record.

    This is order-dependent by construction -- reordering the input can change
    the grouping. That is faithful to the original, not an accident.

    Returns
    -------
    dict with keys ``age``, ``value`` (lists of arrays), ``lat``, ``lon``
    (arrays) and ``members`` (list of index arrays).
    """
    n = len(age)
    assigned = np.zeros(n, bool)
    out = {"age": [], "value": [], "lat": [], "lon": [], "members": []}

    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)

    for i in range(n):
        if assigned[i]:
            continue
        d = haversine_km(lat[i], lon[i], lat, lon)
        members = np.flatnonzero((d <= threshold_km) & ~assigned)

        common = np.unique(np.concatenate([np.asarray(age[j]).ravel()
                                           for j in members if np.size(age[j])]))
        stack = np.full((common.size, members.size), np.nan)
        for k, j in enumerate(members):
            a = np.asarray(age[j], float).ravel()
            v = np.asarray(values[j], float).ravel()
            if a.size == 0 or v.size == 0:
                continue
            if a.size != v.size:
                # MATLAB warns and skips.
                continue
            order = np.argsort(a)
            a, v = a[order], v[order]
            interp = np.interp(common, a, v, left=np.nan, right=np.nan)
            stack[:, k] = interp

        with np.errstate(invalid="ignore"):
            mean = np.nanmean(stack, axis=1) if stack.size else np.array([])
        out["age"].append(common)
        out["value"].append(mean)
        out["lat"].append(lat[i])
        out["lon"].append(lon[i])
        out["members"].append(members)
        assigned[members] = True

    out["lat"] = np.array(out["lat"])
    out["lon"] = np.array(out["lon"])
    return out


# ---------------------------------------------------------------- PCA ------
def pca_fields(cube, n_components=None):
    """PCA in the orientation the MATLAB uses.

    ``data2D = reshape(T2my - nanmean(T2my,3), nx*ny, nz)`` then ``pca(data2D)``
    treats **grid cells as observations and years as variables**. That is the
    transpose of the usual EOF convention, and it means:

    * ``score``  (n_cells x n_comp) reshapes back into spatial *maps*;
    * ``coeff``  (n_years x n_comp) is the *temporal* loading, which is what
      the later station-correlation cell uses.

    This reproduces that orientation exactly.

    Parameters
    ----------
    cube : (n_years, ny, nx) array of annual anomalies
    n_components : int or None

    Returns
    -------
    dict with ``score`` (n_comp, ny, nx) spatial patterns, ``coeff``
    (n_years, n_comp) temporal loadings, ``explained`` (%), ``cumulative`` (%).
    """
    from sklearn.decomposition import PCA

    cube = np.asarray(cube, float)
    nt = cube.shape[0]
    spatial = cube.shape[1:]

    # Anomalies about the time mean, as nanmean(T2my,3) does.
    anom = cube - np.nanmean(cube, axis=0, keepdims=True)
    data2D = anom.reshape(nt, -1).T          # (n_cells, n_years)

    finite = np.all(np.isfinite(data2D), axis=1)
    if finite.sum() < 3:
        raise ValueError("not enough finite grid cells for PCA")
    X = data2D[finite]

    n_comp = n_components or min(X.shape)
    p = PCA(n_components=n_comp)
    scores = p.fit_transform(X)              # (n_good_cells, n_comp)

    full = np.full((data2D.shape[0], n_comp), np.nan)
    full[finite] = scores
    score_maps = full.T.reshape((n_comp,) + spatial)

    return {
        "score": score_maps,
        "coeff": p.components_.T,            # (n_years, n_comp)
        "explained": p.explained_variance_ratio_ * 100.0,
        "cumulative": np.cumsum(p.explained_variance_ratio_) * 100.0,
        "mask": finite.reshape(spatial),
    }

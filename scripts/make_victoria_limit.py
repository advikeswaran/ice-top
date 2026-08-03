"""Reconstruct the missing ``LatitudeLimofVictoriaLand.txt``.

The original file is a 2-column table (latitude, longitude-limit) that
``regions_Ant2K.m`` feeds to ``interp1`` to get, for every query latitude, the
longitude that separates the East Antarctic Plateau from Victoria Land / the
Ross Sea sector. It is not on disk, so it is rebuilt here from topography.

Why this reconstruction is the right shape
------------------------------------------
Between lon 145 and 190 the plateau rule in ``regions_Ant2K.m`` is

    (vlon >= 145) & vlon < 190 & velev >= 0 & vlon < vlonlim   ->  region 1

i.e. inside that longitude band the boundary carries no elevation test of its
own -- ``vlonlim`` *is* doing the job of the 2000 m contour. The source comment
says as much: "a bit complicated because of the high mountains we want to
exclude", meaning the Transantarctic Mountains rise above 2000 m but belong to
Victoria Land, not the plateau.

So the limit is the *interior* plateau edge: scanning eastward from 145 deg,
the longitude past which the surface stays below the plateau altitude. Using
the 60 km DEM is an advantage here -- it smooths out the narrow TAM ridges that
a high-resolution grid would (wrongly) re-admit as plateau.

Run:  python scripts/make_victoria_limit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from antarctic.config import AUX_DIR, VICTORIA_LIMIT_TXT
from antarctic.elevation import default_elevation

LON_MIN, LON_MAX = 145.0, 190.0
PLATEAU_ALTI = 2000.0
TAPER_MARGIN = 200.0  # plateau must clear PLATEAU_ALTI by this much to be trusted
MIN_DROP_DEG = 2.0    # the drop below PLATEAU_ALTI must persist this far east
LAT_STEP = 0.25
LON_STEP = 0.05
SMOOTH_WINDOW = 9  # samples of LAT_STEP, ~2.25 deg of latitude


def plateau_edge_longitude(elev, lat):
    """East edge of the *contiguous* plateau, scanning eastward from LON_MIN.

    Returns NaN where no plateau reaches this latitude inside the band, or
    where it only barely clears PLATEAU_ALTI. That second gate matters: north
    of about 71S the plateau tapers to a sliver that only just tops 2000 m, so
    the contour position swings tens of degrees on a few tens of metres of DEM
    error. Those latitudes are filled by extrapolation instead.

    Taking the first drop below PLATEAU_ALTI after the plateau starts -- rather
    than the last high sample -- is deliberate: it discards any detached
    Transantarctic Mountain highs further east, which is precisely the
    "high mountains we want to exclude" of the original comment.
    """
    lons = np.arange(LON_MIN, LON_MAX + LON_STEP, LON_STEP)
    lats = np.full_like(lons, lat)
    z = np.nan_to_num(elev(lats, lons), nan=0.0)

    high = z >= PLATEAU_ALTI
    if not high.any() or z.max() < PLATEAU_ALTI + TAPER_MARGIN:
        return np.nan
    first_high = int(np.flatnonzero(high)[0])

    # The drop below PLATEAU_ALTI must be *sustained* over MIN_DROP_DEG of
    # longitude to count as the plateau edge. At 1 km DEM resolution the
    # plateau is pitted with sub-kilometre dips and nunatak saddles that would
    # otherwise truncate the scan hundreds of km too early.
    run = max(1, int(round(MIN_DROP_DEG / LON_STEP)))
    low = ~high[first_high:]
    if not low.any():
        return LON_MAX                       # plateau spans the band (near pole)
    # cumulative run-length of consecutive lows
    idx = np.flatnonzero(
        np.convolve(low.astype(int), np.ones(run, int), mode="valid") == run)
    if idx.size == 0:
        return LON_MAX
    return float(lons[first_high + int(idx[0])])


def trim_north_taper(vals, drop_deg=6.0, window=5):
    """Discard the collapsing northern tail of the scan.

    Around 71S the plateau inside 145-190E breaks up into a fragmented sliver
    before disappearing. The scan then reports an edge that lurches back toward
    LON_MIN (about 149 then 147, against ~160 just to the south) -- an artefact
    of fragmentation, not a real boundary.

    Working from the northernmost valid latitude southward, drop any value that
    sits more than ``drop_deg`` below the median of the ``window`` latitudes
    immediately south of it, and stop at the first value that does not. Acting
    only on the northern tail leaves the genuine southern plunge from 190 --
    where the plateau really does span the whole band near the pole -- intact.
    """
    vals = vals.copy()
    while True:
        ok = np.flatnonzero(np.isfinite(vals))
        if ok.size <= window:
            break
        last = ok[-1]
        south = vals[ok[-(window + 1):-1]]
        if vals[last] < np.median(south) - drop_deg:
            vals[last] = np.nan
        else:
            break
    return vals


def fill_north(vals):
    """Hold the last valid value northward (and southward) across NaNs.

    North of about 71S the plateau simply does not reach into 145-190E, so the
    topographic scan is undefined there. The meaningful boundary at those
    latitudes is where the Wilkes Land coast gives way to the Victoria Land
    coast, which is the value the curve already has at its northern end -- so
    extend it rather than letting it collapse to LON_MIN.
    """
    vals = vals.copy()
    ok = np.flatnonzero(np.isfinite(vals))
    if ok.size == 0:
        raise RuntimeError("no plateau found at any latitude")
    vals[: ok[0]] = vals[ok[0]]
    vals[ok[-1] + 1:] = vals[ok[-1]]
    idx = np.arange(len(vals))
    return np.interp(idx, idx[np.isfinite(vals)], vals[np.isfinite(vals)])


def smooth(v, window):
    """Centred moving average with edge padding."""
    if window < 2:
        return v
    pad = window // 2
    vp = np.pad(v, pad, mode="edge")
    kern = np.ones(window) / window
    return np.convolve(vp, kern, mode="valid")


def main():
    elev = default_elevation()
    print(f"DEM: {elev.name}")

    lats = np.arange(-89.75, -59.99, LAT_STEP)
    raw = np.array([plateau_edge_longitude(elev, la) for la in lats])
    trimmed = trim_north_taper(raw)
    lim = np.clip(smooth(fill_north(trimmed), SMOOTH_WINDOW), LON_MIN, LON_MAX)

    AUX_DIR.mkdir(parents=True, exist_ok=True)
    header = (
        "Reconstructed LatitudeLimofVictoriaLand.txt\n"
        "col 1: latitude (deg, ascending)\n"
        "col 2: longitude limit (deg east, 0-360) separating East Antarctic\n"
        "       Plateau (lon < limit) from Victoria Land / Ross Sea (lon > limit)\n"
        f"derived from: {elev.name}, {PLATEAU_ALTI:.0f} m plateau contour,\n"
        f"scanned over lon [{LON_MIN:.0f}, {LON_MAX:.0f}], smoothed over "
        f"{SMOOTH_WINDOW * LAT_STEP:.2f} deg latitude\n"
        "NOT the original file -- an approximation. See scripts/make_victoria_limit.py"
    )
    np.savetxt(VICTORIA_LIMIT_TXT, np.column_stack([lats, lim]),
               fmt="%10.4f %10.4f", header=header)
    print(f"wrote {VICTORIA_LIMIT_TXT}  ({len(lats)} rows)")

    print("\n  lat     raw    smoothed")
    for la in np.arange(-88, -63.9, 2.0):
        i = int(np.argmin(np.abs(lats - la)))
        print(f"{lats[i]:7.2f} {raw[i]:8.2f} {lim[i]:9.2f}")


if __name__ == "__main__":
    main()

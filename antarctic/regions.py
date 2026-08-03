"""Port of ``regions_Ant2K.m``.

Assigns each (lat, lon, surface-elevation) point to one of the 7 PAGES Ant-2K
climate regions of Stenni et al. (2017):

    1 East Antarctic Plateau      5 WAIS
    2 Wilkes Land Coast           6 Victoria Land / Ross Sea
    3 Weddell Sea Coast           7 Dronning Maud Land Coast
    4 Antarctic Peninsula

The MATLAB original is a sequence of threshold rules where later rules
overwrite earlier ones; that ordering is reproduced exactly, statement for
statement, so the two can be diffed line by line.

One deliberate wart is preserved by default -- see ``edml_zero`` below.
"""

from __future__ import annotations

import numpy as np

from .config import VICTORIA_LIMIT_TXT

# Parameters, verbatim from the MATLAB.
PLATEAU_ALTI = 2000.0
PENI = -74.0    # latitude limit of the Peninsula
WED = 360 - 30  # 330; the source notes this "was 50"
B_AI = 67.0     # boundary between Atlantic and Indian sectors ("or 80")

_LIMIT_CACHE = {}


def load_victoria_limit(path=VICTORIA_LIMIT_TXT):
    """Load the (latitude, longitude-limit) table, cached.

    The original ``LatitudeLimofVictoriaLand.txt`` was not delivered; this
    reads the reconstruction produced by ``scripts/make_victoria_limit.py``.
    """
    key = str(path)
    if key not in _LIMIT_CACHE:
        tab = np.loadtxt(path)
        order = np.argsort(tab[:, 0])
        _LIMIT_CACHE[key] = (tab[order, 0], tab[order, 1])
    return _LIMIT_CACHE[key]


def _interp1(xp, fp, x):
    """MATLAB ``interp1`` semantics: linear inside, NaN outside."""
    out = np.interp(x, xp, fp, left=np.nan, right=np.nan)
    return out


def regions_ant2k(lat, lon, elev, limit_path=VICTORIA_LIMIT_TXT, edml_zero=True):
    """Region index for each point.

    Parameters
    ----------
    lat, lon, elev : array_like
        Same shape. ``lon`` is interpreted in the 0-360 convention used by the
        original; values in -180..180 are converted automatically.
        ``elev`` is surface elevation in metres (see :mod:`antarctic.elevation`).
    limit_path : path
        Victoria Land longitude-limit table.
    edml_zero : bool, default True
        Reproduce a quirk of the MATLAB. Lines 47-50 of ``regions_Ant2K.m``
        assign ``vregion = 0`` -- not a region in 1..7 -- to the plateau where
        ``lon < 30`` or ``lon >= 300``. Nothing later overwrites it, because
        every subsequent Dronning Maud / Weddell rule requires ``elev < 2000``.
        So a large wedge of the DML and Weddell *plateau* comes out as 0 rather
        than 1. It is invisible in the original figure only because
        ``caxis([1 7])`` clamps 0 to the plateau colour.

        Set ``edml_zero=False`` to leave those points as region 1, which is
        almost certainly what was meant.

    Returns
    -------
    ndarray of float
        Region numbers, NaN where unclassified. Float, not int, because
        unclassified points are NaN exactly as in the MATLAB.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    elev = np.asarray(elev, dtype=float)
    if not (lat.shape == lon.shape == elev.shape):
        raise ValueError(
            f"lat/lon/elev must share a shape, got {lat.shape}, {lon.shape}, {elev.shape}"
        )

    shape = lat.shape
    vlat = lat.ravel()
    vlon = np.mod(lon.ravel(), 360.0)  # enforce the 0-360 convention
    velev = elev.ravel()
    reg = np.full(vlat.shape, np.nan)

    lim_lat, lim_lon = load_victoria_limit(limit_path)
    vlonlim = _interp1(lim_lat, lim_lon, vlat)

    # NaN-safe comparisons: NaN in elev or vlonlim must make a test False,
    # which is numpy's behaviour already, but it warns. Suppress that.
    with np.errstate(invalid="ignore"):
        # -- Plateau ------------------------------------------------------
        reg[(vlon < 145) & (velev > PLATEAU_ALTI)] = 1          # vs DML / Indian
        reg[(vlon >= 145) & (vlon < 190) & (velev >= 0)
            & (vlon < vlonlim)] = 1                             # vs Ross
        reg[(vlon >= 145) & (vlon < 350) & (vlat < -84)
            & (velev >= 0)] = 1                                 # vs WAIS

        # -- EDML carve-out (see edml_zero) -------------------------------
        edml_fill = 0.0 if edml_zero else 1.0
        reg[(vlon < 30) & (velev > PLATEAU_ALTI)] = edml_fill
        reg[(vlon >= 300) & (velev > PLATEAU_ALTI)] = edml_fill

        # -- Dronning Maud Land coast -------------------------------------
        reg[(vlon > WED) & (velev < PLATEAU_ALTI) & (vlat > -76)
            & (velev >= 0)] = 7
        reg[(vlon > 0) & (vlon < B_AI) & (velev < PLATEAU_ALTI)] = 7

        # -- Weddell Sea coast --------------------------------------------
        reg[(vlon > WED) & (velev < PLATEAU_ALTI) & (vlat < -75)
            & (velev >= 0)] = 3
        reg[(vlon >= 300) & (vlon < WED) & (velev < PLATEAU_ALTI)] = 3

        # -- Indian Ocean sector (Wilkes Land coast) ----------------------
        reg[(vlon >= B_AI) & (vlon < 145) & (velev < PLATEAU_ALTI)] = 2
        reg[(vlon >= 145) & (vlon < 160) & (velev >= 0) & (vlon < vlonlim)
            & (velev < PLATEAU_ALTI) & (vlat > -72)] = 2

        # -- Victoria Land sector -----------------------------------------
        reg[(vlon >= 145) & (vlon < 190) & (velev >= 0) & (vlat < -74)
            & (vlon > vlonlim)] = 6                             # near pole
        reg[(vlon >= 160) & (vlon < 190) & (velev >= 0)
            & (vlon > vlonlim)] = 6                             # near Talos

        # -- WAIS ----------------------------------------------------------
        reg[(vlon >= 190) & (vlon < 280) & (vlat > -85) & (velev >= 0)] = 5
        reg[(vlon >= 280) & (vlon < 300) & (vlat > -85) & (vlat < PENI + 0.5)
            & (velev >= 0)] = 5                                 # vs Peninsula

        # -- Peninsula ------------------------------------------------------
        reg[(vlon >= 280) & (vlon < 305) & (vlat > PENI) & (velev >= 0)] = 4

    return reg.reshape(shape)


def regions_at(lat, lon, elevation=None, **kw):
    """Convenience wrapper that looks up elevation for you."""
    from .elevation import default_elevation
    lat = np.atleast_1d(np.asarray(lat, float))
    lon = np.atleast_1d(np.asarray(lon, float))
    elev = (default_elevation() if elevation is None else elevation)(lat, lon)
    return regions_ant2k(lat, lon, elev, **kw)

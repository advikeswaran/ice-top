"""Cartopy replacements for the MATLAB Mapping Toolbox calls.

``worldmap([-90 -62],[0 360])`` -> South Polar Stereographic axes
``surfm``                       -> pcolormesh in PlateCarree
``plotm``/``load coast``        -> cartopy coastline feature
``bedmap2('coast'/'grounding line')`` -> see note in :func:`add_coast`
``cmocean(...)``                -> :func:`cmocean_like`
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm, ListedColormap

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from .config import REGION_COLOURS, REGION_NAMES

SOUTH_POLAR = ccrs.SouthPolarStereo(central_longitude=0.0)
PLATE = ccrs.PlateCarree()


def polar_axes(fig=None, rect=111, lat_max=-62.0, labels=True):
    """Axes equivalent to ``worldmap([-90 lat_max], [0 360])``."""
    if fig is None:
        fig = plt.figure(figsize=(9, 8))
    args = rect if isinstance(rect, tuple) else (rect,)
    ax = fig.add_subplot(*args, projection=SOUTH_POLAR)
    ax.set_extent([-180, 180, -90, lat_max], crs=PLATE)
    # circular boundary, as worldmap draws
    theta = np.linspace(0, 2 * np.pi, 200)
    verts = np.stack([np.cos(theta), np.sin(theta)], axis=1) * 0.5 + 0.5
    import matplotlib.path as mpath
    ax.set_boundary(mpath.Path(verts), transform=ax.transAxes)
    gl = ax.gridlines(draw_labels=labels, linewidth=0.4, color="0.6", alpha=0.7)
    gl.n_steps = 90
    return fig, ax


def add_coast(ax, lw=1.5, color="k"):
    """Coastline overlay.

    The MATLAB draws ``bedmap2('coast')`` and ``bedmap2('grounding line')``.
    The Bedmap2 toolbox is unavailable, so this uses Natural Earth's Antarctic
    coastline, which follows the ice front (comparable to the bedmap2 coast).
    The *grounding line* is a distinct feature -- the ice/bed contact under the
    shelves -- and Natural Earth has no equivalent, so it is not drawn.
    """
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=lw,
                   edgecolor=color)
    return ax


def cmocean_like(name, n):
    """Small stand-in for the ``cmocean`` colormaps used in the scripts.

    Only the ones actually called are provided: ``balance`` (diverging
    blue-white-red), ``-algae`` and ``rain``. These are perceptual
    approximations, not the exact cmocean lookup tables -- if you want the
    real ones, ``pip install cmocean`` and swap this out.
    """
    reverse = name.startswith("-")
    key = name.lstrip("-")
    tables = {
        "balance": ["#181c43", "#2a68a8", "#8bbcd9", "#f4f4f4",
                    "#e59b74", "#b3341f", "#3e0a12"],
        "algae":   ["#d7f2c0", "#93d08a", "#4fa963", "#1c7f4d",
                    "#125437", "#0b2c1e"],
        "rain":    ["#e8e2d8", "#9fc9b0", "#54a9a6", "#3a72a0",
                    "#3f3f8f", "#3b1a4d"],
    }
    if key not in tables:
        raise KeyError(f"no stand-in for cmocean('{name}'); available: {sorted(tables)}")
    cols = tables[key]
    if reverse:
        cols = cols[::-1]
    return LinearSegmentedColormap.from_list(f"cmo_{name}", cols, N=n)


def region_cmap():
    """The hard-coded 7-entry colormap from ``Mapper_region_colour.m``."""
    cols = [np.array(REGION_COLOURS[i]) / 255.0 for i in range(1, 8)]
    cmap = ListedColormap(cols, name="ant2k_regions")
    norm = BoundaryNorm(np.arange(0.5, 8.5, 1.0), cmap.N)
    return cmap, norm


def surfm(ax, lat, lon, z, **kw):
    """``surfm`` equivalent: pcolormesh of a lat/lon field on a polar axes."""
    kw.setdefault("shading", "auto")
    kw.setdefault("transform", PLATE)
    return ax.pcolormesh(lon, lat, z, **kw)


def scatterm(ax, lat, lon, s=150, c=None, **kw):
    """``scatterm`` equivalent."""
    kw.setdefault("transform", PLATE)
    kw.setdefault("zorder", 5)
    return ax.scatter(lon, lat, s=s, c=c, **kw)


def region_legend(ax, **kw):
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=np.array(REGION_COLOURS[i]) / 255.0,
                     edgecolor="k", linewidth=0.4,
                     label=f"{i} {REGION_NAMES[i]}") for i in range(1, 8)]
    kw.setdefault("loc", "center left")
    kw.setdefault("bbox_to_anchor", (1.02, 0.5))
    kw.setdefault("frameon", False)
    return ax.legend(handles=handles, **kw)

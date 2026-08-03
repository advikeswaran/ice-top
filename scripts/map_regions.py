"""Port of ``Mapper_region_colour.m``.

Builds the 0.1 deg lat/lon grid, looks up surface elevation, classifies with
``regions_ant2k`` and draws the 7 Ant-2K regions on a South Polar Stereographic
map using the hard-coded colours from the original.

Two panels are produced so the region-0 quirk is visible rather than hidden:
left reproduces the MATLAB exactly (where ``caxis([1 7])`` silently clamps the
0-wedge to the plateau colour), right shows the corrected classification.

Run:  python scripts/map_regions.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from antarctic.config import FIG_DIR, REGION_NAMES
from antarctic.elevation import default_elevation
from antarctic.maps import (PLATE, add_coast, polar_axes, region_cmap,
                            region_legend, surfm)
from antarctic.regions import regions_ant2k

GRID_STEP = 0.1


def main():
    elev = default_elevation()
    print(f"DEM: {elev.name}")

    # lati = repmat([-90:0.1:-60]',1,3601);  loni = repmat([0:0.1:360],301,1);
    lat = np.arange(-90.0, -60.0 + GRID_STEP / 2, GRID_STEP)
    lon = np.arange(0.0, 360.0 + GRID_STEP / 2, GRID_STEP)
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    print(f"grid {LA.shape}  ({LA.size:,} points)")

    Z = elev(LA, LO)
    cmap, norm = region_cmap()

    ri_raw = regions_ant2k(LA, LO, Z, edml_zero=True)
    ri_fix = regions_ant2k(LA, LO, Z, edml_zero=False)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(16, 8))

    # Panel 1: exactly what the MATLAB draws. caxis([1 7]) clamps the 0-wedge
    # to the plateau colour, so this is identical to the corrected map -- which
    # is precisely why the bug is invisible in the original figure.
    _, ax1 = polar_axes(fig, (1, 2, 1), lat_max=-61.0, labels=True)
    surfm(ax1, lat, lon, np.where(ri_raw == 0, 1.0, ri_raw),
          cmap=cmap, norm=norm, rasterized=True)
    add_coast(ax1)
    ax1.set_title("As drawn by the MATLAB", fontsize=11)

    # Panel 2: same map, with the cells the MATLAB actually labels 0 hatched.
    _, ax2 = polar_axes(fig, (1, 2, 2), lat_max=-61.0, labels=False)
    surfm(ax2, lat, lon, ri_fix, cmap=cmap, norm=norm, rasterized=True)
    zero = np.where(ri_raw == 0, 1.0, np.nan)
    ax2.contourf(lon, lat, zero, levels=[0.5, 1.5], colors="none",
                 hatches=["////"], transform=PLATE, zorder=4)
    ax2.contour(lon, lat, np.nan_to_num(zero), levels=[0.5], colors="k",
                linewidths=1.2, transform=PLATE, zorder=4)
    add_coast(ax2)
    n0 = int(np.sum((ri_raw == 0) & (Z > 10)))
    ax2.set_title(f"Hatched: cells the MATLAB labels 0, not 1\n"
                  f"({n0:,} land cells, {n0 / max(np.sum((ri_fix == 1) & (Z > 10)), 1):.0%} "
                  f"of the plateau)", fontsize=11)
    print(f"  region-0 land cells: {n0:,}")

    region_legend(ax2)
    fig.suptitle("PAGES Ant-2K regions (Stenni et al. 2017)", fontsize=15)
    out = FIG_DIR / "ant2k_regions.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"wrote {out}")

    # Region areas, weighted by cos(lat), land only.
    ri = regions_ant2k(LA, LO, Z, edml_zero=False)
    w = np.cos(np.deg2rad(LA)) * (Z > 10)
    tot = np.nansum(np.where(np.isfinite(ri), w, 0.0))
    print("\nland-area share by region:")
    for v in range(1, 8):
        share = np.nansum(np.where(ri == v, w, 0.0)) / tot
        print(f"  {v} {REGION_NAMES[v]:26s} {share:6.1%}")


if __name__ == "__main__":
    main()

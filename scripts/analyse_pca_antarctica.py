"""Port of ``Analyse_PCA_Antarctica.m``.

The MATLAB original is a scratchpad: cells are re-run out of order, ``figure(3)``
is reused for four different plots, and several cells read variables that were
left over from an earlier cell. This port keeps the science identical but makes
the order explicit and each section independent.

Deviations from the MATLAB, all deliberate, all reported at runtime:

1. Decimal-year axis. Line 74 uses ``datenum(Date)./365`` and line 429 uses
   ``./365.25``. Since ``datenum`` counts days from year 0, ``/365`` overshoots
   by about +1.5 years, so the "1980-2020" ERA5 trend in the original is really
   computed over roughly 1978.5-2018.5. This port uses true calendar years
   everywhere and prints both so the difference is visible.

2. ``Trendb`` is used at MATLAB line 241 but not computed until line 389. Here
   the Bromwich trend is computed before it is plotted.

3. The Bromwich trend cell inherits ``Year_start=1930``/``Year_stop=2019`` from
   the PALEO cell above it, while its figure title says 1980-2020. This port
   uses 1980-2020, matching the stated intent.

4. ``Stat_data`` (23 stations), ``Latsite``/``Longsite`` are never defined in
   the .m file. ``Stat_data`` is rebuilt from the 17 stations in ``dataAWS.mat``
   -- the other 6 are not in the delivery -- and the ice-core coordinates stand
   in for ``Latsite``/``Longsite``.

5. The 'new team ice cores' section needs ``dataPALEO_LDC.mat``, which is not
   in the delivery. That section is skipped with a message.

Run:  python scripts/analyse_pca_antarctica.py
      python scripts/analyse_pca_antarctica.py --only trends
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from antarctic.analysis import merge_neighbours, pca_fields, trend_1d, trend_map
from antarctic.config import FIG_DIR
from antarctic.elevation import default_elevation
from antarctic.datasets import (load_aws, load_bromwich, load_era5,
                                load_ice_cores, load_paleo_ldc)
from antarctic.helpers import ann_block_ave, export_hybrid_map
from antarctic.maps import (PLATE, add_coast, cmocean_like, polar_axes,
                            scatterm, surfm)

YEAR_START, YEAR_STOP = 1980, 2020         # trend window for ERA5 / Bromwich / AWS
CORE_START, CORE_STOP = 1950, 2020         # trend window for ice cores
TREND_LIM = 0.65                           # colour limit, degC/decade
TREND_LEVELS = np.arange(-0.65, 0.66, 0.1)  # discrete bands, as in the paper figure

_cache = {}


def era5_annual():
    if "era5" not in _cache:
        da = load_era5()
        print(f"  ERA5 source: {da.attrs['note']}")
        ann = da.groupby("time.year").mean("time")
        _cache["era5"] = (ann.year.values.astype(float), ann.lat.values,
                          ann.lon.values, ann.values.astype("float32"))
        _cache["era5_note"] = da.attrs["note"]
    return _cache["era5"]


def _trend_axes(fig, rect, title, cmap, labels=True):
    _, ax = polar_axes(fig, rect, lat_max=-62.0, labels=labels)
    add_coast(ax)
    ax.set_title(title, fontsize=11)
    return ax


# ------------------------------------------------------------ section 1 ----
def section_mean_map():
    """MATLAB cell 'First plot to check'."""
    years, lat, lon, cube = era5_annual()
    tmean = np.nanmean(cube, axis=0)

    fig = plt.figure(figsize=(8.5, 8))
    ax = _trend_axes(fig, 111,
                     f"ERA5 mean temperature {years[0]:.0f}-{years[-1]:.0f}",
                     None)
    m = surfm(ax, lat, lon, tmean, cmap=cmocean_like("balance", 20),
              vmin=-60, vmax=0, rasterized=True)
    fig.colorbar(m, ax=ax, shrink=0.7, label="Temperature (°C)")
    out = FIG_DIR / "01_era5_mean_temperature.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}   (range {np.nanmin(tmean):.1f} to {np.nanmax(tmean):.1f} degC)")


# ------------------------------------------------------------ section 2 ----
def aws_trends():
    """MATLAB cell 'Compute trends' -- per-station linear trends."""
    stations = load_aws()
    rows = []
    for st in stations:
        slope, p, n = trend_1d(st.year, st.temp, YEAR_START, YEAR_STOP)
        rows.append((st.name, st.lat, st.lon, slope, p, n))
    return rows


def report_aws(rows):
    print(f"  {'station':18s} {'lat':>7s} {'lon':>8s} {'trend/dec':>10s} {'p':>8s} {'n':>4s}")
    for name, la, lo, slope, p, n in rows:
        star = " *" if np.isfinite(p) and p < 0.05 else ""
        print(f"  {name:18s} {la:7.2f} {lo:8.2f} {10 * slope:10.3f} {p:8.3f} {n:4d}{star}")
    sig = sum(1 for r in rows if np.isfinite(r[4]) and r[4] < 0.05)
    print(f"  {sig}/{len(rows)} significant at p<0.05")


# ------------------------------------------------------------ section 3 ----
def section_era5_trend(aws_rows):
    """MATLAB cell 'ERA5 trend' plus the AWS scatter overlay."""
    years, lat, lon, cube = era5_annual()
    slope, pval = trend_map(years, cube, YEAR_START, YEAR_STOP)

    # Show what the /365 bug did: the same computation on the shifted axis.
    shifted = years + 1.5
    slope_bug, _ = trend_map(shifted, cube, YEAR_START, YEAR_STOP)
    diff = np.nanmean(np.abs(10 * (slope - slope_bug)))
    print(f"  mean |trend difference| from the datenum/365 axis bug: "
          f"{diff:.4f} degC/dec")

    cmap = cmocean_like("balance", 13)
    fig = plt.figure(figsize=(8.5, 8))
    ax = _trend_axes(fig, 111,
                     f"Temperature trends ERA5 {YEAR_START}-{YEAR_STOP}", cmap)
    m = ax.contourf(lon, lat, 10 * slope, levels=TREND_LEVELS, cmap=cmap,
                    extend="both", transform=PLATE)
    la = [r[1] for r in aws_rows]
    lo = [r[2] for r in aws_rows]
    tr = [10 * r[3] for r in aws_rows]
    scatterm(ax, la, lo, s=180, c=tr, cmap=cmap, vmin=-TREND_LIM,
             vmax=TREND_LIM, edgecolors="0.1", linewidths=2)
    fig.colorbar(m, ax=ax, shrink=0.7, label="Trend (°C/dec)")
    out = FIG_DIR / "03_era5_trend.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}   (grid trend range "
          f"{10 * np.nanmin(slope):.2f} to {10 * np.nanmax(slope):.2f} degC/dec, "
          f"{np.mean(pval < 0.05):.0%} of cells p<0.05)")
    return slope, pval


# ------------------------------------------------------------ section 4 ----
def section_ice_cores(era5_trend, scale=2.0):
    """MATLAB cell 'Compare with ice core' -- d18O trends, no merging."""
    cores = load_ice_cores()
    print(f"  {len(cores)} usable cores (1 empty placeholder dropped)")

    slopes, pvals, lats, lons, ns = [], [], [], [], []
    for a, d in zip(cores.age, cores.d18O):
        s, p, n = trend_1d(a, d * scale, CORE_START, CORE_STOP)
        slopes.append(s); pvals.append(p); ns.append(n)
    slopes = np.array(slopes); pvals = np.array(pvals); ns = np.array(ns)

    end = np.array([a.max() if a.size else np.nan for a in cores.age])
    print(f"  d18O->T scaling {scale}x; records end between "
          f"{np.nanmin(end):.0f} and {np.nanmax(end):.0f} "
          f"(so '{CORE_START}-{CORE_STOP}' is really to {np.nanmax(end):.0f} at best)")
    sig = np.isfinite(pvals) & (pvals < 0.05)
    print(f"  {sig.sum()}/{len(slopes)} cores significant at p<0.05")

    _plot_cores(era5_trend, cores.lat[sig], cores.lon[sig], 10 * slopes[sig],
                f"Ice-core d18O trends ({scale}x) on ERA5 {YEAR_START}-{YEAR_STOP}",
                "04_ice_core_trends.png", marker="D")
    return slopes, pvals


def _plot_cores(era5_trend, lat_s, lon_s, trend_s, title, fname, marker="D"):
    years, lat, lon, _ = era5_annual()
    cmap = cmocean_like("balance", 13)
    fig = plt.figure(figsize=(8.5, 8))
    ax = _trend_axes(fig, 111, title, cmap)
    m = ax.contourf(lon, lat, 10 * era5_trend, levels=TREND_LEVELS, cmap=cmap,
                    extend="both", transform=PLATE)
    if len(lat_s):
        scatterm(ax, lat_s, lon_s, s=200, c=trend_s, cmap=cmap,
                 vmin=-TREND_LIM, vmax=TREND_LIM, marker=marker,
                 edgecolors="0.75", linewidths=2)
    fig.colorbar(m, ax=ax, shrink=0.7, label="Trend (°C/dec)")
    out = FIG_DIR / fname
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


# ------------------------------------------------------------ section 5 ----
def section_ice_cores_merged(era5_trend, scale=1.5, threshold_km=200.0):
    """MATLAB cell 'Compare with ice core merging neighbours'."""
    cores = load_ice_cores()
    g = merge_neighbours(cores.age, cores.d18O, cores.lat, cores.lon,
                         threshold_km=threshold_km)
    sizes = [len(m) for m in g["members"]]
    print(f"  {len(cores)} cores -> {len(g['age'])} groups within {threshold_km:.0f} km "
          f"(largest group {max(sizes)}, {sum(1 for s in sizes if s > 1)} groups merged)")

    slopes, pvals = [], []
    for a, v in zip(g["age"], g["value"]):
        s, p, _ = trend_1d(a, np.asarray(v) * scale, CORE_START, CORE_STOP)
        slopes.append(s); pvals.append(p)
    slopes = np.array(slopes); pvals = np.array(pvals)
    sig = np.isfinite(pvals) & (pvals < 0.05)
    print(f"  d18O->T scaling {scale}x; {sig.sum()}/{len(slopes)} groups p<0.05")

    _plot_cores(era5_trend, g["lat"][sig], g["lon"][sig], 10 * slopes[sig],
                f"Merged ice-core trends ({scale}x, {threshold_km:.0f} km groups)",
                "05_ice_core_merged.png", marker="D")
    return g, slopes, pvals


# ------------------------------------------------------------ section 6 ----
def section_bromwich(aws_rows):
    """MATLAB cell computing ``Trendb`` and the Bromwich trend map."""
    ds = load_bromwich()
    rec = ds["RECON"]
    ann = rec.groupby("time.year").mean("time")
    years = ann.year.values.astype(float)
    cube = ann.values.astype("float32")
    slope, pval = trend_map(years, cube, YEAR_START, YEAR_STOP)

    lat2d = ds["lat"].values
    lon2d = ds["lon"].values
    land = ds["LANDMASK"].values > 0.5
    slope_masked = np.where(land, slope, np.nan)

    cmap = cmocean_like("balance", 13)
    fig = plt.figure(figsize=(8.5, 8))
    ax = _trend_axes(fig, 111,
                     f"Temperature trends Bromwich et al. {YEAR_START}-{YEAR_STOP}",
                     cmap)
    m = ax.pcolormesh(lon2d, lat2d, slope_masked * 10, cmap=cmap,
                      vmin=-TREND_LIM, vmax=TREND_LIM, transform=PLATE,
                      shading="auto", rasterized=True)
    la = [r[1] for r in aws_rows]
    lo = [r[2] for r in aws_rows]
    tr = [10 * r[3] for r in aws_rows]
    scatterm(ax, la, lo, s=180, c=tr, cmap=cmap, vmin=-TREND_LIM,
             vmax=TREND_LIM, edgecolors=(0.2, 0.75, 0.75), linewidths=2)
    fig.colorbar(m, ax=ax, shrink=0.7, label="Trend (°C/dec)")

    out = FIG_DIR / "06_bromwich_trend.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    export_hybrid_map(fig, FIG_DIR / "06_bromwich_trend_hybrid.pdf")
    plt.close(fig)
    print(f"  wrote {out.name} and 06_bromwich_trend_hybrid.pdf "
          f"(raster field + vector overlay)")
    print(f"  land trend range {10 * np.nanmin(slope_masked):.2f} to "
          f"{10 * np.nanmax(slope_masked):.2f} degC/dec")
    return slope, pval


# ------------------------------------------------------------ section 7 ----
def section_pca(n_show=5):
    """MATLAB cells 'Apply PCA' and 'Plot PCA'."""
    years, lat, lon, cube = era5_annual()
    res = pca_fields(cube, n_components=min(80, cube.shape[0]))
    exp, cum = res["explained"], res["cumulative"]
    print(f"  variance explained: PC1 {exp[0]:.1f}%, PC2 {exp[1]:.1f}%, "
          f"PC3 {exp[2]:.1f}%")
    for target in (90, 95, 99):
        k = int(np.searchsorted(cum, target) + 1)
        print(f"  {k} PCs reach {target}% cumulative variance")

    cmap = cmocean_like("balance", 20)
    fig = plt.figure(figsize=(20, 4.6))
    for i in range(n_show):
        ax = _trend_axes(fig, (1, n_show, i + 1),
                         f"PC {i + 1}  ({exp[i]:.1f}%)", cmap, labels=False)
        v = res["score"][i]
        lim = np.nanpercentile(np.abs(v), 99)
        surfm(ax, lat, lon, v, cmap=cmap, vmin=-lim, vmax=lim, rasterized=True)
    fig.suptitle("ERA5 annual-anomaly PCA: spatial patterns "
                 "(grid cells as observations, years as variables)", fontsize=13)
    out = FIG_DIR / "07_pca_patterns.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.2))
    for i in range(3):
        axs[0].plot(years, res["coeff"][:, i], label=f"PC{i + 1}")
    axs[0].set_xlabel("year"); axs[0].set_ylabel("loading")
    axs[0].set_title("Temporal loadings"); axs[0].legend(); axs[0].grid(alpha=.3)
    axs[1].plot(np.arange(1, len(cum) + 1), cum, "o-", ms=3)
    axs[1].axhline(95, color="r", ls="--", lw=1)
    axs[1].set_xlabel("number of PCs"); axs[1].set_ylabel("cumulative %")
    axs[1].set_title("Cumulative variance explained"); axs[1].grid(alpha=.3)
    out2 = FIG_DIR / "07_pca_variance.png"
    fig.savefig(out2, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out2.name}")
    return res


# ------------------------------------------------------------ section 8 ----
def build_stat_data(years):
    """Reconstruct ``Stat_data``: stations on a common annual axis.

    The MATLAB loops ``for j = 1:23`` but ``dataAWS.mat`` holds 17 stations,
    so 6 of the originals are not in the delivery. Returns the 17 available.
    """
    stations = load_aws()
    mat = np.full((len(stations), len(years)), np.nan)
    for i, st in enumerate(stations):
        idx = np.searchsorted(years, st.year)
        ok = (idx >= 0) & (idx < len(years))
        ok &= np.isin(st.year, years)
        for y, t in zip(st.year[ok], st.temp[ok]):
            mat[i, int(np.flatnonzero(years == y)[0])] = t
    return stations, mat


def section_station_correlation(pca_res):
    """MATLAB cells 'Identify stations' and 'Maps where correlated'."""
    years, lat, lon, cube = era5_annual()
    sel = (years >= 1950) & (years <= 2021)
    yrs = years[sel]

    stations, stat = build_stat_data(yrs)
    print(f"  Stat_data rebuilt: {stat.shape[0]} stations x {stat.shape[1]} years "
          f"(MATLAB assumed 23; {23 - stat.shape[0]} not in the delivery)")

    # -- which PCs track a station record
    coeff = pca_res["coeff"][sel]
    n_pc = coeff.shape[1]
    corr = np.full((n_pc, stat.shape[0]), np.nan)
    for i in range(n_pc):
        for j in range(stat.shape[0]):
            ok = np.isfinite(stat[j]) & np.isfinite(coeff[:, i])
            if ok.sum() > 5:
                corr[i, j] = np.corrcoef(stat[j][ok], coeff[ok, i])[0, 1]
    with np.errstate(invalid="ignore"):
        pcid = np.nanmax(np.abs(corr), axis=1) > 0.5
    print(f"  {int(pcid.sum())}/{n_pc} PCs correlate |r|>0.5 with at least one station "
          f"(PCs {', '.join(str(i + 1) for i in np.flatnonzero(pcid)[:10])}"
          f"{' ...' if pcid.sum() > 10 else ''})")

    # -- map of best station correlation, vectorised over the grid
    anom = cube[sel] - np.nanmean(cube[sel], axis=0, keepdims=True)
    T = anom.reshape(len(yrs), -1)
    best = np.full(T.shape[1], -np.inf)
    for j in range(stat.shape[0]):
        s = stat[j]
        ok = np.isfinite(s)
        if ok.sum() < 10:
            continue
        sv = s[ok] - s[ok].mean()
        Tv = T[ok] - T[ok].mean(axis=0, keepdims=True)
        denom = np.sqrt((sv @ sv) * (Tv * Tv).sum(axis=0))
        with np.errstate(invalid="ignore", divide="ignore"):
            r = (sv @ Tv) / denom
        # Guard: a constant or all-NaN column gives denom 0, and letting that
        # through leaves -inf in `best`, which squares to inf.
        best = np.fmax(best, np.where(np.isfinite(r), r, -np.inf))
    r2 = np.where(np.isfinite(best), best ** 2, np.nan).reshape(cube.shape[1:])

    cores = load_ice_cores()
    cmap = cmocean_like("rain", 10)
    fig = plt.figure(figsize=(8.5, 8))
    ax = _trend_axes(fig, 111, "Amount of variance explained", cmap)
    # Masked to the continent, as the published figure is.
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    land = default_elevation().land_mask(LA, LO)
    m = ax.contourf(lon, lat, np.where(land, r2, np.nan),
                    levels=np.arange(0, 1.01, 0.1), cmap=cmap, transform=PLATE)
    scatterm(ax, cores.lat, cores.lon, s=60, c="none", marker="s",
             edgecolors="k", linewidths=0.8)
    scatterm(ax, [s.lat for s in stations], [s.lon for s in stations], s=90,
             c=[(212 / 255, 23 / 255, 35 / 255)], marker="^", edgecolors="k")
    fig.colorbar(m, ax=ax, shrink=0.7, label="r$^2$")
    out = FIG_DIR / "08_station_correlation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}   (median r2 over land {np.nanmedian(np.where(land, r2, np.nan)):.2f})")


# ------------------------------------------------------------ section 9 ----
def section_paleo_ldc():
    """MATLAB cell 'Add New ice cores from the team'."""
    try:
        load_paleo_ldc()
    except FileNotFoundError as e:
        print(f"  SKIPPED: {e}")
        return
    print("  dataPALEO_LDC.mat found -- extend this section to use it.")


# ---------------------------------------------------------------- main -----
SECTIONS = ["mean", "aws", "trends", "cores", "merged", "bromwich",
            "pca", "correlation", "paleo"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=SECTIONS, default=SECTIONS)
    args = ap.parse_args()
    run = set(args.only)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    era5_trend = None
    aws_rows = None
    pca_res = None

    if "mean" in run:
        print("\n[1] ERA5 mean temperature map")
        section_mean_map()

    if run & {"aws", "trends", "cores", "merged", "bromwich"}:
        print("\n[2] AWS station trends 1980-2020")
        aws_rows = aws_trends()
        if "aws" in run:
            report_aws(aws_rows)

    if run & {"trends", "cores", "merged"}:
        print("\n[3] ERA5 gridded trends 1980-2020")
        era5_trend, _ = section_era5_trend(aws_rows)

    if "cores" in run:
        print("\n[4] Ice-core d18O trends (2x scaling)")
        section_ice_cores(era5_trend, scale=2.0)

    if "merged" in run:
        print("\n[5] Ice-core trends with 200 km neighbour merging (1.5x)")
        section_ice_cores_merged(era5_trend, scale=1.5)

    if "bromwich" in run:
        print("\n[6] Bromwich reconstruction trends 1980-2020")
        section_bromwich(aws_rows)

    if run & {"pca", "correlation"}:
        print("\n[7] PCA of ERA5 annual anomalies")
        pca_res = section_pca()

    if "correlation" in run:
        print("\n[8] PC / station correlation")
        section_station_correlation(pca_res)

    if "paleo" in run:
        print("\n[9] New team ice cores")
        section_paleo_ldc()

    print(f"\nfigures in {FIG_DIR}")


if __name__ == "__main__":
    main()

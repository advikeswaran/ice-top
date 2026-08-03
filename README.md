# Ant-2K MATLAB → Python port

Python port of the three MATLAB scripts in `~/Downloads/Mathieu/`, plus
reconstructions of the four missing dependencies.

```
antarctic/
  config.py      paths, region names, the hard-coded region colours
  elevation.py   bedmap2_interp replacement (pluggable DEM backend)
  regions.py     port of regions_Ant2K.m
  datasets.py    loaders for the 5 data files
  analysis.py    trends, ice-core neighbour merging, PCA
  helpers.py     Ann_block_ave + exportHybridMap reconstructions
  maps.py        cartopy replacements for the Mapping Toolbox calls
scripts/
  make_victoria_limit.py     rebuilds LatitudeLimofVictoriaLand.txt
  map_regions.py             port of Mapper_region_colour.m
  analyse_pca_antarctica.py  port of Analyse_PCA_Antarctica.m
data/LatitudeLimofVictoriaLand.txt   (generated)
figures/                             (generated)
```

Run in order:

```bash
python scripts/make_victoria_limit.py     # ~5 s
python scripts/map_regions.py             # ~60 s
python scripts/analyse_pca_antarctica.py  # ~27 s
```

Requires numpy, scipy, xarray, netCDF4, pandas, matplotlib, cartopy, pyproj,
scikit-learn — all already present in the `cfr-env` conda environment.

---

## Missing dependencies and what was done about them

| Missing | Resolution |
|---|---|
| `LatitudeLimofVictoriaLand.txt` | **Reconstructed** from topography — see below |
| `bedmap2` toolbox | Replaced by `antarctic/elevation.py`, backed by BedMachine Antarctica v3 (500 m) with a 60 km fallback |
| `cmocean` | `maps.cmocean_like()` — perceptual approximations of `balance`, `-algae`, `rain`. `pip install cmocean` and swap if you need the exact tables |
| `Ann_block_ave` | **Reconstructed** as `helpers.ann_block_ave` |
| `exportHybridMap` | **Reconstructed** as `helpers.export_hybrid_map` |
| `dataPALEO_LDC.mat` | **Not recoverable** — not in the delivery. That section is skipped |
| `Stat_data`, `Latsite`, `Longsite` | Never defined in the .m file. `Stat_data` rebuilt from the 17 AWS stations; ice-core coordinates stand in for `Latsite`/`Longsite` |

### The Victoria Land boundary

The original file is a (latitude, longitude-limit) table used by `interp1` to
split the East Antarctic Plateau from Victoria Land / Ross Sea between 145°E
and 190°E.

It is reconstructed by tracing the **east edge of the contiguous 2000 m
surface** across that longitude band. This is the right construction, not a
guess: inside 145–190 the plateau rule in the MATLAB carries *no* elevation
test of its own, so `vlonlim` is what stands in for the 2000 m contour there.
Taking the first drop below 2000 m — rather than the last high sample —
discards the detached Transantarctic Mountain peaks, which is exactly the
"high mountains we want to exclude" of the source comment.

At 1 km resolution three refinements matter, all in
`scripts/make_victoria_limit.py`:

* the drop below 2000 m must persist for ≥2° of longitude (`MIN_DROP_DEG`),
  otherwise sub-kilometre dips and nunatak saddles truncate the scan hundreds
  of km too early;
* latitudes where the plateau clears 2000 m by less than 200 m
  (`TAPER_MARGIN`) are rejected as noise-dominated;
* the northern tail is trimmed (`trim_north_taper`) where the plateau
  fragments around 71°S and the scan lurches back toward 145°E. Those
  latitudes are then filled by extrapolating the last robust value, **159.8°E**
  — which is where the Wilkes coast gives way to the Victoria Land coast, and
  matches the 160° the MATLAB itself hard-codes on lines 68 and 74.

**It is still an approximation.** Talos Dome (−72.78, 159.07) lands ~1° on the
plateau side of the reconstructed line; in Stenni et al. it is usually grouped
with Victoria Land. If the original file resurfaces, drop it into `data/` and
nothing else changes.

### The elevation backend

`bedmap2_interp` is replaced by **BedMachine Antarctica v3**
(`~/DataFiles/BedMachineAntarctica-v3.nc`): 13333² at 500 m on EPSG:3031, read
at `stride=2` (1 km, ~178 MB) by default. It also supplies a proper land mask
from the BedMachine `mask` variable (ocean / ice-free land / grounded ice /
floating ice), used for the continent masking in the figures.

If BedMachine is absent, `default_elevation()` falls back to the `HGT` field
inside `recon_t2m_1958-2022_ano.final.nc` — a 114×114 WRF grid at 60 km. The
difference is large at small ice caps:

| Site | BedMachine (1 km) | 60 km fallback | Reference |
|---|---|---|---|
| Byrd | 1530 m (−0) | 1520 m (−10) | 1530 m |
| South Pole | 2827 m (−8) | 2832 m (−3) | 2835 m |
| Dome C | 3239 m (+6) | 3218 m (−15) | 3233 m |
| WAIS Divide | 1781 m (+15) | 1749 m (−17) | 1766 m |
| Vostok | 3463 m (−25) | 3456 m (−32) | 3488 m |
| Dome A | 4049 m (−44) | 3972 m (−122) | 4093 m |
| **Law Dome** | **1355 m (−15)** | **742 m (−628)** | 1370 m |

Law Dome is the telling one: a 200 km ice cap that 60 km simply cannot see.

The BAS and NOAA gridded endpoints for Bedmap2/Bedmap3/ETOPO are all dead or
moved as of August 2026. `SurfaceElevation.from_netcdf(path, var="surface")`
accepts any other polar-stereographic grid (defaults to EPSG:3031).

---

## Problems found in the originals

These are real defects, not porting artefacts. Each is reproduced-by-default
and flagged, or fixed and reported at runtime.

### 1. `regions_Ant2K.m` assigns region 0 to 22% of the plateau

Lines 47–50 set `vregion = 0` — not a region in 1..7 — for the plateau where
`lon < 30` or `lon >= 300`. Nothing later overwrites it, because every
subsequent Dronning Maud and Weddell rule requires `elev < 2000`. So the DML
and Weddell **plateau** comes out as 0 rather than 1: **94,251 land cells on
the 0.1° grid, 22% of the plateau**, including the South Pole and Kohnen/EDML.

It is invisible in the original figure only because `caxis([1 7])` clamps 0 to
the plateau colour. `figures/ant2k_regions.png` shows the affected wedge
hatched. Default is `edml_zero=True` (faithful); pass `edml_zero=False` for
what was almost certainly meant.

### 2. The ERA5 file in the delivery is the wrong download — RESOLVED

`Mathieu/Data/1820e70cfe13658fa322f37e6e688cfd.nc` holds variable `t` on
`pressure_level = 1000` (paramId 130), and its latitude axis stops at 55°S.
Over the plateau, 1000 hPa is roughly 3 km *below the ice surface*, so the
values there are a fictitious downward extrapolation:

| Site | File | True annual mean | Error |
|---|---|---|---|
| Dumont d'Urville (coast) | −10.0 °C | ≈ −11 °C | ~1 °C |
| Halley (coast) | −13.4 °C | ≈ −18 °C | ~5 °C |
| South Pole | −24.0 °C | ≈ −49 °C | **25 °C** |
| Vostok | −22.9 °C | ≈ −55 °C | **32 °C** |
| Dome C | −20.7 °C | ≈ −54 °C | **33 °C** |

This file cannot reproduce the published Figure 1: its trends are about half
the observed amplitude and the Ross Ice Shelf comes out the wrong sign. Ten
different trend windows were scanned and none recover the figure, so it is the
input, not the window.

**The correct file is `~/DataFiles/ae3baa6a74f0aa315dc3de6f83298f0e.nc`** —
variable `t2m`, paramId 167, genuine 2 m temperature, 1979–2025, 0.1°,
60–90°S. `config.py` now prefers it automatically and falls back to the
delivery file if it is absent. Switching to it:

| check | 1000 hPa file | t2m file | target |
|---|---|---|---|
| RMS vs published Fig. 1a | 0.216 | **0.126** | — |
| mean field range | −26 … +6 °C | **−53.9 … −1.8 °C** | Antarctic surface |
| PCs to reach 95% variance | 28 | **18** | "20 roughly means 95%" (MATLAB line 502) |
| median r² over land | 0.37 | **0.53** | ~0.5 in published Fig. 1b |

The PC count independently corroborates it: the MATLAB's own comment guesses
20 PCs for 95%, which the t2m file reproduces and the pressure-level file does
not.

**Remaining caveat:** the t2m file starts in 1979, while the MATLAB's PCA and
station-correlation cells run over 1950–2021. Those sections are therefore
truncated to 1979–2021 here (43 years, not 72). The 1980–2020 trend maps are
fully covered. For complete fidelity, re-download `2m_temperature` from
`reanalysis-era5-single-levels-monthly-means` starting in 1940.

### 3. Inconsistent decimal-year conversion

Line 74 uses `datenum(Date)./365`, line 429 uses `./365.25`. `datenum` counts
days from year 0, so `/365` overshoots by ≈ +1.5 years: the "1980–2020" ERA5
trend is really computed over ≈ 1978.5–2018.5. Effect on the trend field is
small (mean |Δ| = 0.022 °C/dec) but it is a genuine inconsistency. The port
uses true calendar years and prints the difference.

### 4. `Trendb` used before it is computed

MATLAB line 241 plots `Trendb`; it is computed at line 389. The cell only works
if the file is run bottom-up. The port computes before plotting.

### 5. The Bromwich trend cell inherits the wrong window

That cell runs after the PALEO cell sets `Year_start=1930`, `Year_stop=2019`,
while its own figure title says 1980–2020. The port uses 1980–2020.

### 6. Ice-core data problems

- Record 8 is an **empty placeholder** — zero-length `Age`, `List_lat` exactly
  0.0. Dropped, leaving 78 usable cores.
- `List_lon` is in −180..180 while `regions_Ant2K` expects 0–360.
- **No core extends past 2012.** The `1950–2020` window is really "to at most
  2012", and the effective end year varies core by core, so the trends are not
  computed over a common period.

### 7. `pvale` / `pvalb` are computed and never used

The gridded p-values are calculated but no map is ever masked with them. The
port returns them so you can.

### 8. Neighbour merging is order-dependent

The greedy 200 km grouping walks records in file order and takes the *seed*
record's coordinates for the whole group. Reordering the input changes the
grouping. Faithful, but worth knowing: 78 cores → 33 groups, largest group 18.

---

## Notes on the PCA

`pca(data2D)` where `data2D` is `(nx*ny) × nz` treats **grid cells as
observations and years as variables** — the transpose of the usual EOF
convention. So `score` reshapes into spatial maps and `coeff` is the temporal
loading. The port reproduces that orientation exactly (`analysis.pca_fields`).

Results: PC1 19.6%, PC2 16.6%, PC3 14.0%; 28 PCs reach 95% cumulative variance
(the MATLAB comment guesses "20 roughly means 95%"). PC2 is a clean annular,
SAM-like mode.

The station-correlation cells (`Nb = 76`, `for j = 1:23`) assume 23 station
records; `dataAWS.mat` has 17. Six are not in the delivery. Those cells also
read `xd`, left over from an earlier loop — the port uses the actual year axis.

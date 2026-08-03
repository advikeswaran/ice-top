"""Surface-elevation lookup: a stand-in for the MATLAB ``bedmap2_interp(lat,lon,'surface')``.

The Bedmap2 MATLAB toolbox is not available here, and the BAS/NOAA gridded
endpoints that used to serve Bedmap2/ETOPO are dead. The default backend
therefore reads the ``HGT`` (terrain height) field that already ships inside
the Bromwich reconstruction netCDF: a 114x114 WRF polar-stereographic grid at
60 km spacing covering the whole continent, peaking at 4010 m over Dome A.

That is genuinely the right *quantity* (ice-surface elevation in metres, 0 over
ocean, same as Bedmap2 'surface'), just coarse. 60 km is fine for the only
thing ``regions_Ant2K`` uses elevation for -- the 2000 m plateau contour, a
broad smooth feature -- but it will round off narrow coastal detail.

If you later obtain a real Bedmap2/Bedmap3/REMA grid, use
``SurfaceElevation.from_netcdf(...)`` or ``.from_arrays(...)`` and everything
downstream is unchanged.
"""

from __future__ import annotations

import numpy as np
import pyproj
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

from .config import BEDMACHINE_NC, BROMWICH_NC


class SurfaceElevation:
    """Callable surface-elevation field on an arbitrary lat/lon query.

    Parameters
    ----------
    x, y : 1-D arrays
        Projected coordinates of the source grid (metres), strictly increasing.
    z : 2-D array, shape (len(y), len(x))
        Surface elevation in metres.
    proj : pyproj.Proj
        Projection mapping (lon, lat) -> (x, y).
    """

    def __init__(self, x, y, z, proj, name="unknown"):
        self.x = np.asarray(x, float)
        self.y = np.asarray(y, float)
        self.z = np.asarray(z, float)
        self.proj = proj
        self.name = name
        self.mask = None
        self._mask_interp = None
        self._interp = RegularGridInterpolator(
            (self.y, self.x), self.z, method="linear",
            bounds_error=False, fill_value=np.nan,
        )

    # -- constructors ----------------------------------------------------
    @classmethod
    def from_bromwich_hgt(cls, path=BROMWICH_NC):
        """Default backend: the WRF terrain height inside the Bromwich file."""
        ds = xr.open_dataset(path, decode_times=False)
        lat = ds["lat"].values.astype(float)
        lon = ds["lon"].values.astype(float)
        hgt = ds["HGT"].values.astype(float)
        a = ds.attrs
        proj = pyproj.Proj(
            proj="stere", lat_0=-90.0,
            lat_ts=float(a.get("TRUELAT1", -71.0)),
            lon_0=float(a.get("STAND_LON", 0.0)),
            R=6370000.0,
        )
        gx, gy = proj(lon, lat)
        # The WRF grid is regular in projected space to within ~4 m of 60 km,
        # so collapsing the 2-D coordinate arrays to 1-D axes is safe.
        x = gx.mean(axis=0)
        y = gy.mean(axis=1)
        if x[0] > x[-1]:
            x, hgt = x[::-1], hgt[:, ::-1]
        if y[0] > y[-1]:
            y, hgt = y[::-1], hgt[::-1, :]
        ds.close()
        return cls(x, y, hgt, proj, name="Bromwich WRF HGT (60 km)")

    @classmethod
    def from_bedmachine(cls, path=BEDMACHINE_NC, stride=2):
        """BedMachine Antarctica v3 surface elevation -- the preferred backend.

        13333x13333 at 500 m on EPSG:3031. ``stride`` coarsens on read:
        stride=2 gives 1 km and ~178 MB, which is still far finer than any grid
        used downstream (0.1 deg of latitude is 11 km) while keeping memory
        sane. stride=1 loads the native 500 m grid at ~711 MB.

        Also loads ``mask`` (0 ocean, 1 ice-free land, 2 grounded ice,
        3 floating ice, 4 Lake Vostok) for a proper land/ice-shelf mask.

        Note BedMachine surface heights are relative to the EIGEN-EC4 geoid,
        whereas Bedmap2 used the ellipsoid; the difference across Antarctica is
        tens of metres, immaterial to the 2000 m plateau threshold.
        """
        ds = xr.open_dataset(path)
        s = slice(None, None, stride)
        z = ds["surface"][s, s].values.astype("float32")
        mask = ds["mask"][s, s].values
        x = ds["x"][s].values.astype(float)
        y = ds["y"][s].values.astype(float)
        ds.close()

        if x[0] > x[-1]:
            x, z, mask = x[::-1], z[:, ::-1], mask[:, ::-1]
        if y[0] > y[-1]:
            y, z, mask = y[::-1], z[::-1, :], mask[::-1, :]

        obj = cls(x, y, z, pyproj.Proj("EPSG:3031"),
                  name=f"BedMachine v3 surface ({500 * stride} m)")
        obj.mask = mask
        obj._mask_interp = RegularGridInterpolator(
            (y, x), mask.astype("float32"), method="nearest",
            bounds_error=False, fill_value=np.nan)
        return obj

    @classmethod
    def from_netcdf(cls, path, var="surface", xname="x", yname="y",
                    proj=None, name=None):
        """Backend for a real Bedmap2/Bedmap3/REMA grid in polar stereographic.

        Defaults to EPSG:3031 (Antarctic Polar Stereographic, 71S), which is
        what Bedmap2 and Bedmap3 use.
        """
        ds = xr.open_dataset(path)
        z = ds[var].values.astype(float)
        x = ds[xname].values.astype(float)
        y = ds[yname].values.astype(float)
        if proj is None:
            proj = pyproj.Proj("EPSG:3031")
        if x[0] > x[-1]:
            x, z = x[::-1], z[:, ::-1]
        if y[0] > y[-1]:
            y, z = y[::-1], z[::-1, :]
        ds.close()
        return cls(x, y, z, proj, name=name or f"{path}:{var}")

    @classmethod
    def from_arrays(cls, x, y, z, proj=None, name="user"):
        if proj is None:
            proj = pyproj.Proj("EPSG:3031")
        return cls(x, y, z, proj, name=name)

    # -- evaluation ------------------------------------------------------
    def __call__(self, lat, lon):
        """Interpolate to (lat, lon). Longitudes may be 0-360 or -180..180.

        Returns NaN outside the source grid, mirroring ``bedmap2_interp``.
        """
        lat = np.asarray(lat, float)
        lon = np.asarray(lon, float)
        lon180 = ((lon + 180.0) % 360.0) - 180.0
        gx, gy = self.proj(lon180, lat)
        pts = np.stack([np.ravel(gy), np.ravel(gx)], axis=-1)
        out = self._interp(pts).reshape(np.shape(lat))
        return out


    def land_mask(self, lat, lon, include_shelves=True):
        """Boolean land mask at (lat, lon).

        Uses the BedMachine ``mask`` when available (1 ice-free land,
        2 grounded ice, 3 floating ice, 4 Lake Vostok), otherwise falls back to
        a simple elevation test.
        """
        if self._mask_interp is None:
            return self(lat, lon) > 10.0
        lat = np.asarray(lat, float)
        lon = np.asarray(lon, float)
        lon180 = ((lon + 180.0) % 360.0) - 180.0
        gx, gy = self.proj(lon180, lat)
        pts = np.stack([np.ravel(gy), np.ravel(gx)], axis=-1)
        m = self._mask_interp(pts).reshape(np.shape(lat))
        keep = (1, 2, 4) + ((3,) if include_shelves else ())
        return np.isin(np.nan_to_num(m, nan=0).astype(int), keep)


_DEFAULT = None


def default_elevation():
    """Lazily-built module-level default elevation field.

    Prefers BedMachine v3 (500 m) and falls back to the 60 km Bromwich HGT.
    """
    global _DEFAULT
    if _DEFAULT is None:
        if BEDMACHINE_NC.exists():
            _DEFAULT = SurfaceElevation.from_bedmachine()
        else:
            _DEFAULT = SurfaceElevation.from_bromwich_hgt()
    return _DEFAULT


def reset_default_elevation():
    """Drop the cached DEM so the next call rebuilds it.

    The cache is a module global, which is precisely the state IPython's
    ``%autoreload`` preserves across reloads. If the backend changes mid-session
    (a new DEM file appears, or ``stride`` is retuned), call this rather than
    restarting the kernel.
    """
    global _DEFAULT
    _DEFAULT = None


def bedmap2_interp(lat, lon, kind="surface"):
    """Drop-in for the MATLAB call used in the original scripts."""
    if kind != "surface":
        raise NotImplementedError(f"only 'surface' is supported, got {kind!r}")
    return default_elevation()(lat, lon)

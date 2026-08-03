"""Loaders for the five data files the MATLAB scripts read."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.io as sio
import xarray as xr

from .config import (AWS_MAT, BROMWICH_NC, ERA5_NC, ICECORE_MAT, PALEO_LDC_MAT)


# ---------------------------------------------------------------- ERA5 -----
def load_era5(path=ERA5_NC):
    """ERA5 monthly temperature, as a (time, lat, lon) DataArray in degC.

    Handles both ERA5 files present on this machine and reports which one it
    got, because they are not interchangeable:

    * ``~/DataFiles/ae3baa6a...nc`` -- variable ``t2m``, paramId 167, genuine
      2 m temperature, 1979-2025, 0.1 deg, 60S-90S. **This is the one that
      reproduces the published figure.**
    * ``Mathieu/Data/1820e70c...nc`` -- variable ``t`` on ``pressure_level``
      1000 hPa, paramId 130. Over the 3 km plateau that level sits below the
      ice surface, so it reads ~33 degC too warm at Dome C and its trends are
      about half the observed amplitude. Kept only for comparison.

    Longitudes come back 0-360 ascending and latitudes ascending.
    """
    ds = xr.open_dataset(path)
    if "t2m" in ds:
        da, kind = ds["t2m"], "ERA5 2 m temperature (t2m, paramId 167)"
    elif "t" in ds:
        da, kind = ds["t"], "ERA5 1000 hPa temperature (t, paramId 130) -- NOT 2 m"
    else:
        raise KeyError(f"no temperature variable in {path}; found {list(ds.data_vars)}")

    if "pressure_level" in da.dims:
        da = da.squeeze("pressure_level", drop=True)
    rename = {k: v for k, v in
              {"valid_time": "time", "latitude": "lat", "longitude": "lon"}.items()
              if k in da.dims or k in da.coords}
    da = da.rename(rename)
    da = da.assign_coords(lon=np.mod(da.lon, 360.0)).sortby("lon").sortby("lat")
    da = da - 273.15
    da.attrs["units"] = "degC"
    da.attrs["source_file"] = str(path)
    da.attrs["note"] = kind
    return da


# ------------------------------------------------------------ Bromwich -----
def load_bromwich(path=BROMWICH_NC):
    """Bromwich et al. reconstructed T2m anomaly, 1958-2022.

    The grid is curvilinear: ``lat`` and ``lon`` are 2-D (114x114) on a WRF
    polar-stereographic mesh at 60 km, with dims ``south_north``/``west_east``.
    Time is 'days since 1951-01-01'.

    Returns
    -------
    xr.Dataset with RECON (time, south_north, west_east), 2-D lat/lon coords,
    HGT and LANDMASK, and decoded times.
    """
    ds = xr.open_dataset(path)
    ds["RECON"].attrs.setdefault("units", "K")
    return ds


# ------------------------------------------------------------ ice cores ----
@dataclass
class IceCores:
    age: list = field(default_factory=list)     # list of 1-D arrays, years CE
    d18O: list = field(default_factory=list)    # list of 1-D arrays, per mil
    lat: np.ndarray = None
    lon: np.ndarray = None                      # -180..180 as stored
    valid: np.ndarray = None                    # bool mask of usable records

    def __len__(self):
        return len(self.age)


def load_ice_cores(path=ICECORE_MAT, drop_invalid=True):
    """Ice-core age / d18O records with site coordinates.

    Two data problems are handled here:

    * record 8 (0-based index 7) is empty -- zero-length ``Age`` and
      ``List_lat`` of exactly 0.0, i.e. a placeholder, not a real site;
    * ``List_lon`` is in the -180..180 convention while ``regions_Ant2K``
      expects 0-360. The raw values are kept; convert at the call site.

    Note that no record extends past 2012, so any 'to 2020' trend window is
    really 'to at most 2012', and the effective end year varies per core.
    """
    m = sio.loadmat(path)
    age = [np.asarray(a, float).ravel() for a in m["Age"][0]]
    d18O = [np.asarray(a, float).ravel() for a in m["d18O"][0]]
    lat = np.asarray(m["List_lat"][0], float)
    lon = np.asarray(m["List_lon"][0], float)

    valid = np.array([
        a.size > 0 and d.size == a.size and np.isfinite(la) and la != 0.0
        for a, d, la in zip(age, d18O, lat)
    ])

    if drop_invalid:
        keep = np.flatnonzero(valid)
        return IceCores([age[i] for i in keep], [d18O[i] for i in keep],
                        lat[keep], lon[keep], np.ones(keep.size, bool))
    return IceCores(age, d18O, lat, lon, valid)


# ------------------------------------------------------------------ AWS ----
@dataclass
class Station:
    name: str
    year: np.ndarray
    temp: np.ndarray
    lat: float
    lon: float


def load_aws(path=AWS_MAT):
    """Automatic weather station / manned station annual temperatures.

    Returns a list of :class:`Station`, in the field order of the struct
    (which is the order the MATLAB loops over).
    """
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    s = m["dataAWS"]
    out = []
    for name in s._fieldnames:
        rec = getattr(s, name)
        out.append(Station(
            name=name,
            year=np.asarray(rec.year, float).ravel(),
            temp=np.asarray(rec.T_AWS, float).ravel(),
            lat=float(rec.lat),
            lon=float(rec.lon),
        ))
    return out


# ------------------------------------------------------- PALEO / LDC -------
def load_paleo_ldc(path=PALEO_LDC_MAT):
    """New team ice cores (``dataPALEO``, ``dataLDC``).

    This file is **not** part of the delivery -- ``Data/`` contains only four
    files. Raises FileNotFoundError with that context rather than a bare
    traceback so the calling script can skip the section cleanly.
    """
    from pathlib import Path
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} is missing. It is not in the Mathieu delivery (Data/ has "
            "only the ERA5, Bromwich, ice-core and AWS files). The 'Add New "
            "ice cores from the team' section cannot be reproduced without it."
        )
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    return m["dataPALEO"], m["dataLDC"]

"""Paths and shared constants for the Ant-2K port."""

from pathlib import Path
import os

# Root of the original MATLAB delivery. Override with ANT2K_DATA if you move it.
DATA_DIR = Path(os.environ.get("ANT2K_DATA", Path.home() / "Downloads" / "Mathieu" / "Data"))

PKG_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PKG_DIR.parent
AUX_DIR = PROJECT_DIR / "data"
FIG_DIR = PROJECT_DIR / "figures"

# Two candidate ERA5 files exist. The one shipped in the Mathieu delivery is a
# *pressure-level* download (variable `t` at 1000 hPa, paramId 130) and is
# ~33 degC too warm over the plateau -- it cannot reproduce the published
# figure. The one in ~/DataFiles is genuine 2 m temperature (`t2m`, paramId
# 167) and does. Prefer the latter; keep the former for comparison.
ERA5_T2M_NC = Path.home() / "DataFiles" / "ae3baa6a74f0aa315dc3de6f83298f0e.nc"
ERA5_1000HPA_NC = DATA_DIR / "1820e70cfe13658fa322f37e6e688cfd.nc"
ERA5_NC = ERA5_T2M_NC if ERA5_T2M_NC.exists() else ERA5_1000HPA_NC
BROMWICH_NC = DATA_DIR / "recon_t2m_1958-2022_ano.final.nc"
ICECORE_MAT = DATA_DIR / "Age_d18O_WNanddiffNullHyp.mat"
AWS_MAT = DATA_DIR / "dataAWS.mat"
PALEO_LDC_MAT = DATA_DIR / "dataPALEO_LDC.mat"  # not shipped with the delivery

VICTORIA_LIMIT_TXT = AUX_DIR / "LatitudeLimofVictoriaLand.txt"

# Preferred DEM: BedMachine Antarctica v3, 500 m, EPSG:3031. Falls back to the
# 60 km WRF HGT field inside the Bromwich netCDF when absent.
BEDMACHINE_NC = Path.home() / "DataFiles" / "BedMachineAntarctica-v3.nc"

# PAGES Ant-2K regions (Stenni et al. 2017)
REGION_NAMES = {
    1: "East Antarctic Plateau",
    2: "Wilkes Land Coast",
    3: "Weddell Sea Coast",
    4: "Antarctic Peninsula",
    5: "WAIS",
    6: "Victoria Land / Ross Sea",
    7: "Dronning Maud Land Coast",
}

# RGB colours hard-coded in Mapper_region_colour.m
REGION_COLOURS = {
    1: (29, 143, 255),
    2: (0, 255, 255),
    3: (143, 237, 143),
    4: (255, 214, 0),
    5: (255, 164, 0),
    6: (255, 0, 0),
    7: (204, 132, 62),
}

"""
barra_extension.py
Extend the BARRA-R evaluation period from 1990-2009 to 1990-2014.

Steps:
  1. Regrid raw BARRA-R variables (ua, va, ta, zg) for 2010-2014.
  2. For each turbulence index, compute the index for the new years in the new barra data.
  3. Compute monthly freq > BARRA-p99 and concatenate onto existing files (with same 1990-2009 threshold)
     TI3 has no existing file — creates 1990-2014 fresh if old BARRA-R TMP files still exist on scratch, otherwise 2010-2014 only (with warning).

submit with the barra_extension.pbs file --> using this command: qsub barra_extension.pbs
"""

import os
import sys
import glob
import inspect
import warnings
import numpy as np
import xarray as xr
import xesmf as xe
import dask
from dask.distributed import Client
from dask.utils import SerializableLock

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from cat_indices import calc_turbulence_indices, windspeed, VWS, TI1, Ri, TI2, TI3

import logging
logging.getLogger("flox").setLevel(logging.WARNING)


# set-up / configuration

P = 250
run = "evaluation_BARRA-R_r1i1p1f1"
source_id = "BARRA-R"
experiment_id = "evaluation"
member_id = "r1i1p1f1"

NEW_YEARS  = np.arange(2010, 2015)          # years to add
OLD_YEARS  = np.arange(1990, 2010)          # only needed for TI3 full-period rebuild
FULL_YEARS = np.arange(1990, 2015)

BARRA_TMP_DIR = "/scratch/v46/ls7238/CAT_turbulence/BARRA-R"
EXT_TMP_DIR   = "/scratch/v46/ls7238/CAT_turbulence/TMP_BARRA_EXT"  # separate from main pipeline

barra_raw_vars = {"wnd_ucmp": "ua", "wnd_vcmp": "va", "air_temp": "ta", "geop_ht": "zg"}

# None = read p99 from existing file's attrs; value provided for TI3 (no existing file)
barra_p99_dict = {
    "windspeed": None,
    "VWS":       0.01237245,
    "Ri":        None,
    "TI1":       None,
    "TI2":       None,
    "TI3":       2.94360592e-07,
}


# Helper functions for the path to relevant vars
def barra_tmp_path(var_short, year):
    return (f"{BARRA_TMP_DIR}/TMP_{var_short}_AUS-15_BARRA-R_evaluation_r1i1p1f1_BOM_BARPA-R_v1-r1_6hr_{year}.nc")

def ti_tmp_path(ti, year):
    return (f"{EXT_TMP_DIR}/TMP_{ti}-{P}hPa_AUS-15_BARRA-R_evaluation_r1i1p1f1_BOM_BARPA-R_v1-r1_6hr_{year}.nc")

def freq_file_path(ti):
    return (f"/scratch/v46/ls7238/CAT_turbulence/{ti}/{P}hPa/BARRA-p99/freq-above-p99/"
            f"{ti}-{P}hPa-monthly-freq-above-p99_AUS-15_{run}_BOM_BARPA-R_v1-r1_6hr.nc")


# STEP 1: Regrid the new years of BARRA data from cj37 (2010-2014)

# Find a barpa target file to copy grid from
def get_ds_target():
    pattern = ("/g/data/py18/BARPA/output-CMIP6/DD/AUS-15/BOM/ACCESS-CM2/historical/r4i1p1f1/BARPA-R/v1-r1/6hr/ua250/v*/"
                "ua250_AUS-15_ACCESS-CM2_historical_r4i1p1f1_BOM_BARPA-R_v1-r1_6hr_197901-197912.nc")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"Cannot find a BARPA target-grid file:\n  {pattern}")
    print(f"Target grid: {matches[0]}")
    return xr.open_dataset(matches[0]).isel(time=0)[["lat", "lon"]]

# Regrid relevant barra variables
def regrid_barra_new_years(ds_target, n_workers, years=NEW_YEARS):
    lat_min = ds_target["lat"].values[0]
    lat_max = ds_target["lat"].values[-1]
    lon_min = ds_target["lon"].values[0]
    lon_max = ds_target["lon"].values[-1]
    lock = SerializableLock()

    # Drop un-needed vars for memory
    def _preprocess(ds):
        ds = (ds.drop_vars(["latitude_longitude"], errors="ignore")
                .sel(latitude=slice(lat_min - 0.2, lat_max + 0.2), longitude=slice(lon_min - 0.2, lon_max + 0.2))
                .sel(pressure=[100, 150, 200, 250, 300], method="nearest"))
        try:
            ds = ds.drop_vars(["forecast_period", "forecast_reference_time"])
        except Exception:
            pass
        return ds

    print("\nRegridding BARRA-R variables")
    with Client(threads_per_worker=1, n_workers=n_workers) as client:
        for raw_var, short in barra_raw_vars.items():
            regridder = None
            for year in years:
                out = barra_tmp_path(short, year)
                if os.path.exists(out):
                    print(f"exists: {out}")
                    continue

                filelist = sorted(glob.glob(f"/g/data/cj37/BARRA/BARRA_R/v1/analysis/prs/{raw_var}/{year}/*/"
                                            f"{raw_var}-an-prs-PT0H-BARRA_R-v1-*T*00Z.sub.nc"))
                if not filelist:
                    print(f"WARNING: no source files found for {raw_var} {year}, skipping")
                    continue

                print(f"opening {raw_var} {year} ({len(filelist)} files)...")
                ds = (xr.open_mfdataset(filelist, preprocess=_preprocess,
                                         combine="nested", concat_dim="time",
                                         parallel=True, decode_timedelta=False, lock=lock)
                        .rename({"latitude": "lat", "longitude": "lon", raw_var: short})
                        .chunk({"pressure": 1, "time": 120, "lat": -1, "lon": -1}))

                if regridder is None:
                    print("fitting regridder...")
                    regridder = xe.Regridder(ds, ds_target, "bilinear")

                print(f"regridding and saving {year}...")
                ds_r = regridder(ds)
                ds_r.to_netcdf(out, engine="h5netcdf",
                               encoding={short: {"zlib": True, "complevel": 6}})

                n_nulls = int(xr.open_dataset(out).isel(pressure=0, lat=0, lon=0)[short]
                              .isnull().sum().values)
                #if n_nulls > 0:
                #    os.remove(out)
                #    raise RuntimeError(f"Nulls in {out} — removed. Check source files.")
                
                print(f"saved {out}  (nulls={n_nulls})")
                ds.close(); ds_r.close()


# STEP 2+3: Turbulence index calculation for eval + Frequency over BARRA p99 + append to existing file
def _delayed_ti_year(turbulence_index, year, outfile):
    """Compute one year of a turbulence index from regridded BARRA-R TMP files."""

    dict_variables = {"t": "ta", "u": "ua", "v": "va", "z": "zg",
                      "p": P, "p0": P - 50, "p1": P + 50}
    inverted = {v: k for k, v in dict_variables.items() if isinstance(v, str)}   # only string values (variable names)

    # Helpers
    def _filename(var, pressure):
        return barra_tmp_path(dict_variables[var], year)

    def _open_it(file):
        ds = (xr.open_dataset(file, decode_times=True, engine="h5netcdf",
                              chunks={"time": 100, "lat": -1, "lon": -1})
                              .drop_vars(["crs"], errors="ignore")
                              .astype("float32")
                              .convert_calendar("standard"))
        VAR = list(ds.data_vars)[0]
        ds = ds.rename({VAR: inverted[VAR]})
        try:
            ds = ds.expand_dims("pressure")
        except Exception:
            pass
        ds["pressure"] = ds["pressure"].astype("int")
        return ds

    sig_vars = set(inspect.signature(globals()[turbulence_index]).parameters.keys())
    params = sig_vars & {"t", "u", "v", "z"}
    plvls = sig_vars & {"p0", "p", "p1"} or {"p"}

    ds = xr.merge([
        xr.concat([
            _open_it(_filename(var, pr)).sel({"pressure": dict_variables[pr]}, method="nearest")
            for pr in plvls
        ], dim="pressure")
        for var in params
    ], join="outer")

    ds = calc_turbulence_indices(ds, which=turbulence_index, p=P, u="u", v="v", t="t", z="z")
    ds_out = ds[[turbulence_index]].sel(time=str(year))
    try:
        ds_out = ds_out.sel(pressure=P, method="nearest")
    except Exception:
        pass

    return ds_out.to_netcdf(outfile, mode="a", compute=False)


def _make_freq_preprocess(p99_threshold):
    """Factory so p99 value is properly captured per turbulence index."""
    def _inner(ds):
        return (ds > p99_threshold).resample({"time": "ME"}).mean("time", skipna=True)
    return _inner


def compute_and_append(turbulence_index, p99_val, years_to_process):
    os.makedirs(EXT_TMP_DIR, exist_ok=True)
    existing = freq_file_path(turbulence_index)
    os.makedirs(os.path.dirname(existing), exist_ok=True)

    # Skip if freq file already covers the target years
    if os.path.exists(existing):
        with xr.open_dataset(existing) as _ds_check:
            if int(_ds_check.time.dt.year.max().values) >= max(years_to_process):
                print(f"\n---{turbulence_index}: freq already extended to {int(_ds_check.time.dt.year.max().values)}, skipping.---")
                return

    print(f"\n---{turbulence_index}: {years_to_process[0]}-{years_to_process[-1]}---")

    if p99_val is None:
        if not os.path.exists(existing):
            raise ValueError(f"{turbulence_index}: no existing file and no p99 value provided.")
        with xr.open_dataset(existing) as _ds:
            p99_val = float(_ds.attrs["p99"])
        print(f"p99 read from existing file: {p99_val}")

    # Compute TI TMP files for any missing years
    delayed_list = []
    for year in years_to_process:
        tmp = ti_tmp_path(turbulence_index, year)
        if os.path.exists(tmp):
            print(f"TI tmp exists: {tmp}")
        else:
            print(f"queuing: {tmp}")
            delayed_list.append(_delayed_ti_year(turbulence_index, year, tmp))

    if delayed_list:
        batch_size = 1
        for i in range(0, len(delayed_list), batch_size):
            batch = delayed_list[i:i + batch_size]
            print(f"computing batch {i // batch_size + 1}/{-(-len(delayed_list) // batch_size)}...")
            dask.compute(*batch)

    # Check all TMP files are present
    tmp_files = [ti_tmp_path(turbulence_index, y) for y in years_to_process]
    missing = [f for f in tmp_files if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(f"Missing TMP files after compute: {missing}")

    # Compute monthly freq for these years
    print("computing frequency...")
    new_freq = xr.open_mfdataset(
        tmp_files,
        preprocess=_make_freq_preprocess(p99_val),
        combine="nested", concat_dim="time",
    ).assign_coords(run=run)

    # Concatenate onto existing file (or create fresh for TI3)
    if os.path.exists(existing) and turbulence_index != "TI3":
        print(f"  concatenating onto existing file...")
        with xr.open_dataset(existing) as ds_old:
            combined = xr.concat([ds_old, new_freq], dim="time")
    else:
        combined = new_freq

    combined = combined.assign_attrs({
        "turbulence_index": turbulence_index,
        "pressure_level":   P,
        "p99":              p99_val,
        "Description": (
            f"Frequency of {turbulence_index} at {P}hPa above the BARRA-R 99th percentile. "
            f"Evaluation 1990-2014, 6-hourly source data, monthly frequency."
        ),
    })

    # Safe write: tmp then replace
    tmp_out = existing + ".writing.nc"
    combined.to_netcdf(tmp_out, compute=True)
    os.replace(tmp_out, existing)
    print(f"  saved → {existing}")

    # Clean up TI TMP files
    for f in tmp_files:
        if os.path.exists(f):
            os.remove(f)
            print(f"  removed TMP: {f}")


# MAIN
if __name__ == "__main__":
    ncpus = int(os.environ.get("PBS_NCPUS", 40))

    # Step 1
    ds_target = get_ds_target()
    regrid_barra_new_years(ds_target, n_workers=min(4, ncpus), years=FULL_YEARS)

    # Step 2 and 3
    with dask.config.set(scheduler="synchronous"):
        for ti, p99_val in barra_p99_dict.items():
            if ti == "TI3":
                compute_and_append(ti, p99_val, FULL_YEARS) # needs 1990-2014
            else:
                compute_and_append(ti, p99_val, NEW_YEARS)


    print(f"\nAll done.")

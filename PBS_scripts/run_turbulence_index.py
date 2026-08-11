"""
Compute monthly frequency-above-p99 (and baseline percentiles) for a given
CAT turbulence index, across BARRA-R evaluation and BARPA-R historical / future runs.

Required for script to work:
    - assumes BARRA-R variables have been re-gridded already and are stored in v46 scratch
    - cat_indices.py in same directory
    - run_turbulence.pbs in same directory (is is the shell that runs this script)

Usage:
    qsub -v TURB_INDEX -N XXXX run_turbulence.pbs
- full run eg:
    qsub -v TURB_INDEX=TI2 -N TI2_job run_turbulence.pbs 
- test eg:
    qsub -v TURB_INDEX=TI2 -N TI2_test run_turbulence_test.pbs

"""

# IMPORTS
import os
import sys
import inspect
import argparse
import warnings
import numpy as np
import xarray as xr
import dask
from dask.distributed import Client

from cat_indices import calc_turbulence_indices, windspeed, VWS, TI1, AbsVort, Ri, TI2, TI3

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import logging
logging.getLogger("flox").setLevel(logging.WARNING)


#                                                                                       CONFIG
parser = argparse.ArgumentParser()
parser.add_argument("turbulence_index", help="One of windspeed, VWS, Ri, TI1, TI2, TI3")
parser.add_argument("--ncpus", type=int, default=int(os.environ.get("PBS_NCPUS", 20)))
parser.add_argument("--test", action="store_true",
                     help="Restrict to 1 model run and 2 years per scenario, "
                          "and write to a separate _TEST output folder.")
args = parser.parse_args()

turbulence_index = args.turbulence_index
ncpus = args.ncpus
TEST_MODE = args.test

# tag used for scratch TMP directories / output folder so test runs can
# never collide with, or be mistaken for, a real completed result
run_tag = turbulence_index + ("_TEST" if TEST_MODE else "")

#                                                                                  PRE-DEFINED VARIABLES

# Pressure
P = 250
p = P
p0 = P - 50
p1 = P + 50

# lat and lon slices of interest
mid_lat_slice = slice(-50, -25)
lon_slice = slice(90, 195)

# Time periods of interest
baseline_time_range = np.arange(1990, 2009 + 1)

dict_years = {"evaluation": (1990, 2009),
              "historical": (1979, 2014),
              "future": (2015, 2100)}

# which p99 directory to use
p99_dir = "mmm-p99"

# Model runs and names for all scenarios
list_evaluation = ["evaluation_BARRA-R_r1i1p1f1"]

list_historical = ["historical_ACCESS-CM2_r4i1p1f1", "historical_ACCESS-ESM1-5_r6i1p1f1", "historical_CESM2_r11i1p1f1", "historical_CMCC-ESM2_r1i1p1f1",
                   "historical_EC-Earth3_r1i1p1f1", "historical_MPI-ESM1-2-HR_r1i1p1f1", "historical_NorESM2-MM_r1i1p1f1"]

list_ssp126 = ["ssp126_ACCESS-CM2_r4i1p1f1", "ssp126_ACCESS-ESM1-5_r6i1p1f1", "ssp126_CESM2_r11i1p1f1", "ssp126_CMCC-ESM2_r1i1p1f1",
               "ssp126_EC-Earth3_r1i1p1f1", "ssp126_MPI-ESM1-2-HR_r1i1p1f1", "ssp126_NorESM2-MM_r1i1p1f1"]

list_ssp370 = ["ssp370_ACCESS-CM2_r4i1p1f1", "ssp370_ACCESS-ESM1-5_r6i1p1f1", "ssp370_CESM2_r11i1p1f1", "ssp370_CMCC-ESM2_r1i1p1f1",
               "ssp370_EC-Earth3_r1i1p1f1", "ssp370_MPI-ESM1-2-HR_r1i1p1f1", "ssp370_NorESM2-MM_r1i1p1f1"]

list_ssp585 = ["ssp585_ACCESS-CM2_r4i1p1f1", "ssp585_EC-Earth3_r1i1p1f1"]

list_future = list_ssp126 + list_ssp370 + list_ssp585

dict_model_index = {"evaluation": list_evaluation,
                    "historical": list_historical,
                    "future": list_future,
                    }

# Pre-computed mmm-p99 values from the interactive jupyter notebook (/home/563/ls7238/code/CAT_turbulence/gen_work/turbulence_AUSCAT/CAT_indices-caluclation.ipynb)
mmm_p99_dict = {"windspeed": 76.7527,
                "VWS": 0.012295220450099018,
                "Ri": 1.152088,
                "TI1": 2.1185820196367124e-07,
                "TI2": 2.267155e-07,
                "TI3": 2.3883844807661087e-07
                }

if turbulence_index not in mmm_p99_dict:
    sys.exit(f"Unknown turbulence_index '{turbulence_index}'. "
              f"Must be one of {list(mmm_p99_dict.keys())}")

p99 = mmm_p99_dict[turbulence_index]

# for testing out script on a small section of years across the models - make sure the pipeline has been converted to a pbs script fine
if TEST_MODE:
    # one model run per scenario, 2 years each, so a full pipeline pass
    # (delayed compute -> freq-above-p99 -> percentiles -> cleanup)
    
    # initialise variables on sample reduced 2 year period
    dict_model_index = {k: v[:1] for k, v in dict_model_index.items()}
    dict_years = {"evaluation": (1990, 1991), "historical": (1990, 1991), "future": (2026, 2027)}
    baseline_time_range = np.arange(1990, 1991 + 1)

    # for printing out just to make sure all is okay as it runs.
    print(f"[TEST MODE] model runs: {dict_model_index}")
    print(f"[TEST MODE] years: {dict_years}, baseline: {baseline_time_range}")



#                                                                               FUNCTIONS
# (unchanged from jupyter notebooks minus bits that are not needed)

def delayed_turbulence_index(turbulence_index=None,
                             year=None,
                             path="/g/data/py18/BARPA/output-CMIP6/DD/AUS-15/BOM",
                             source_id=None,
                             experiment_id=None,
                             member_id=None,
                             table_id="6hr",
                             version="v20231001",
                             P=250,
                             outfile=None,
                             chunks={"time": -1, "lat": -1, "lon": -1},
                             ):
    """Use this function to compute the turbulence indices in a delayed routine"""

    dict_variables = {"t": "ta", "u": "ua", "v": "va", "z": "zg",
                      "p": int(P), "p0": int(P - 50), "p1": int(P + 50),}
    
    inverted_dict = {value: key for key, value in dict_variables.items()}

    # helper funcs for this func:
    def _filename(var, pressure, source_id, experiment_id):
        '''For collecting the file paths and names names for the relevant variables: 
                - The BARPA-R files named like their original directory and file path on PY18
                - The BARRA-R variables have been regridded to match BARPA-R variables, function reads pre-saved files from my scratch.'''

        VAR = f"{dict_variables[var]}{dict_variables[pressure]}"

        # Barra files from scratch
        if source_id == "BARRA-R" and experiment_id == "evaluation":
            filename = (f"/scratch/v46/ls7238/CAT_turbulence/BARRA-R/"
                        f"TMP_{dict_variables[var]}_AUS-15_BARRA-R_evaluation_r1i1p1f1_"
                        f"BOM_BARPA-R_v1-r1_6hr_{year}.nc")
        # Barpa files from PY18
        else:
            filename = (f"{path}/{source_id}/{experiment_id}/{member_id}/BARPA-R/v1-r1/{table_id}/{VAR}/{version}/"
                        f"{VAR}_AUS-15_{source_id}_{experiment_id}_{member_id}_BOM_BARPA-R_v1-r1_6hr_{year}01-{year}12.nc")

        return filename

    def _open_it(file):
        '''Now function for opening the files that the function above has identified'''
        ds = (xr.open_dataset(file, decode_times=True, chunks=chunks)
              .drop_vars(["crs"], errors="ignore")          # get rid of un-needed variables
              .astype("float32")
              .convert_calendar("standard")
              )
        VAR = list(ds.variables)[0]
        ds = ds.rename({VAR: inverted_dict["".join(c for c in VAR if c.isalpha())]})
        try:
            ds = ds.expand_dims("pressure")
        except Exception:
            pass
        ds["pressure"] = ds["pressure"].astype("int")
        return ds

    
    turbulence_index_vars = set()
    turbulence_index_vars.update(list(inspect.signature(globals()[turbulence_index]).parameters.keys()))

    params = turbulence_index_vars.intersection(["t", "u", "v", "z"])
    plvls = turbulence_index_vars.intersection(["p0", "p", "p1"])
    if len(plvls) == 0:
        plvls = set(["p"])

    # open files and combine together
    ds = xr.merge([xr.concat([_open_it(_filename(var, pressure, source_id, experiment_id)).sel({"pressure": dict_variables[pressure]}, method="nearest") for pressure in list(plvls)],
                             dim="pressure",) for var in list(params)], join="outer", )

    # run turbulence index calculation from /home/563/ls7238/code/CAT_turbulence/gen_work/turbulence_AUSCAT/PBS_scripts/cat_indices.py for relevant turbulence index
    ds = calc_turbulence_indices(ds, which=turbulence_index, p=P, u="u", v="v", t="t", z="z")

    # temp path for year by year saves
    if outfile is None:
        outfile = (f"/scratch/v46/ls7238/CAT_turbulence/TMP_{turbulence_index}/"
                   f"TMP_{turbulence_index}-{P}hPa_AUS-15_{source_id}_{experiment_id}_{member_id}_BOM_BARPA-R_v1-r1_6hr_{year}.nc")

    ds_to_write = ds[[turbulence_index]].sel({"time": str(year)})

    try:
        ds_to_write = ds_to_write.sel({"pressure": P}, method="nearest")
    except Exception:
        pass

    # turn to netcdf temps
    delayed_write = ds_to_write.to_netcdf(outfile, mode="a", compute=False)

    return delayed_write


def p99freq_preprocess(ds):
    """Calculate frequency of exceeding p99 threshold"""
    return (ds > p99).resample({"time": "ME"}).mean(["time"], skipna=True)


def quantiles_preprocess(ds):
    """Monthly values of 1st-99th quantiles within the mid-lat box"""
    ds = ds.convert_calendar("standard")
    ds = ds.chunk({"time": 4 * 365, "lat": -1, "lon": -1})
    ds = (
        ds.sel(lat=mid_lat_slice, lon=lon_slice)
        .resample(time="ME")
        .apply(lambda ds: ds.quantile(np.arange(0, 1, 0.01), dim=["lat", "lon", "time"]))
    )
    return ds


def run_p99freq_and_quantiles(scenario, chunks={"time": -1, "lat": -1, "lon": -1}):
    """
    Calls all functions together to compute the p99 exceedance frequency and the quantiles, 
    then saves one file per model / scenario and removes all of the temp files created in process.

    scenario has to be one of ['historical', 'future', 'evaluation']
    """

    for run in dict_model_index[scenario]:
        experiment_id, source_id, member_id = run.split("_")

        # initialising file paths and names for percentiles
        base_dir = f"/scratch/v46/ls7238/CAT_turbulence/{run_tag}/{P}hPa/{p99_dir}"
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(f"{base_dir}/freq-above-p99/", exist_ok=True)
        p99_filename = (f"{base_dir}/freq-above-p99/"
                        f"{turbulence_index}-{P}hPa-monthly-freq-above-p99_AUS-15_{run}_BOM_BARPA-R_v1-r1_6hr.nc")

        # and percentile paths and names for historical and eval comparison later
        if scenario in ("historical", "evaluation"):
            frequency = "monthly"
            perc_filename = (f"{base_dir}/percentiles/"
                             f"{turbulence_index}-{P}hPa-{frequency}-percentiles_AUS-15_{run}_BOM_BARPA-R_v1-r1_6hr.nc")
            perc_filename_exists = os.path.exists(perc_filename)
        else:
            perc_filename_exists = True

        # Check if file has already been made, don't re-make
        if os.path.exists(p99_filename) and perc_filename_exists:
            print(f"{p99_filename} already exists, skipping {run}")
            continue

        # Begin temp runs using delayed func
        delayed_list = []
        start_year, end_year = dict_years[scenario]
        os.makedirs(f"/scratch/v46/ls7238/CAT_turbulence/TMP_{run_tag}", exist_ok=True)

        for year in np.arange(start_year, end_year + 1):
            tmp_file = (f"/scratch/v46/ls7238/CAT_turbulence/TMP_{run_tag}/"
                        f"TMP_{run_tag}-{P}hPa_AUS-15_{source_id}_{experiment_id}_{member_id}_BOM_BARPA-R_v1-r1_6hr_{year}.nc")

            if os.path.exists(tmp_file):
                print(f"File '{tmp_file}' already exists.")

            else:
                print(f"Making {tmp_file}")
                delayed_list.append(
                    delayed_turbulence_index(turbulence_index=turbulence_index,
                                             year=year,
                                             source_id=source_id,
                                             experiment_id=experiment_id,
                                             member_id=member_id,
                                             P=P,
                                             outfile=tmp_file,
                                             chunks=chunks,)
                                             )

        if len(delayed_list) > 0:
            print("compute ... ")
            batch_size = 5
            for i in range(0, len(delayed_list), batch_size):
                batch = delayed_list[i:i + batch_size]
                print(f"Computing batch {i // batch_size + 1} of "
                      f"{(len(delayed_list) - 1) // batch_size + 1}...")
                dask.compute(*batch)

        filelist = [f"/scratch/v46/ls7238/CAT_turbulence/TMP_{run_tag}/"
                    f"TMP_{run_tag}-{P}hPa_AUS-15_{source_id}_{experiment_id}_{member_id}_BOM_BARPA-R_v1-r1_6hr_{year}.nc"
                    for year in np.arange(start_year, end_year + 1)]

        # Calculate frequency over temp files and save just frequency in final files
        if os.path.exists(p99_filename):
            print(f"File '{p99_filename}' already exists.")
        else:
            xr.open_mfdataset(filelist, preprocess=p99freq_preprocess, combine="nested", concat_dim="time").assign_coords({"run": run}).assign_attrs({
                "turbulence_index": turbulence_index,
                "pressure_level": P,
                "Description": (f"Frequency of {turbulence_index} at {P}hPa above the 99th percentile for BARPA-R experiments based on BARPA-R MMM percentile from 1990 to "
                                f"2009 within latitudes (-50 to -25) and longitudes (90 to 195) using 6-hourly data and calculating frequencies per calendar month"),
                "p99": p99,
            }).to_netcdf(p99_filename, compute=True)

            print(f"made {p99_filename}")

        if scenario in ("historical", "evaluation"):
            frequency = "monthly"
            os.makedirs(f"{base_dir}/percentiles", exist_ok=True)
            perc_filename = f"{base_dir}/percentiles/{turbulence_index}-{P}hPa-{frequency}-percentiles_AUS-15_{run}_BOM_BARPA-R_v1-r1_6hr.nc"

            if os.path.exists(perc_filename):
                print(f"File '{perc_filename}' already exists.")

            else:
                filelist_baseline = [f"/scratch/v46/ls7238/CAT_turbulence/TMP_{run_tag}/"
                                     f"TMP_{run_tag}-{P}hPa_AUS-15_{source_id}_{experiment_id}_{member_id}_BOM_BARPA-R_v1-r1_6hr_{year}.nc"
                                     for year in baseline_time_range]
                ds = xr.open_mfdataset(filelist_baseline, combine="nested", concat_dim="time", preprocess=quantiles_preprocess,
                                       ).assign_coords({"run": run}).assign_attrs({
                                           "turbulence_index": turbulence_index,
                                           "pressure_level": P,
                                           "Description": (f"Percentiles of {turbulence_index} at {P}hPa for BARPA-R "
                                                           f"experiments using 6-hourly data and calculating frequencies "
                                                           f"per calendar month for years 1990 to 2009"),
                                                           })

                try:
                    ds.to_netcdf(perc_filename, compute=True)
                    print(f"Made '{perc_filename}'")
                except Exception as e:
                    print(f"Error in {run}: {e}")

        for file in filelist:
            if os.path.exists(file):
                os.remove(file)
                print(f"File removed: {file}")
            else:
                print(f"File does not exist: {file}")




#                                                                              RUN

if __name__ == "__main__":
    print(f"Running turbulence_index={turbulence_index}, P={P}, ncpus={ncpus}")

    with Client(threads_per_worker=ncpus, n_workers=1) as client:
        print(client.dashboard_link)
        for scenario in ["historical", "future"]:
            run_p99freq_and_quantiles(scenario)

    with Client(threads_per_worker=ncpus, n_workers=1) as client:
        print(client.dashboard_link)
        for scenario in ["evaluation"]:
            run_p99freq_and_quantiles(scenario)

    print(f"Done: {turbulence_index}")

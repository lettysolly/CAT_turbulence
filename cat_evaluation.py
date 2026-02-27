#!/usr/bin/env python
# coding: utf-8

# after you have run "streamlined_turbulence_code.ipynb", run this to perform the model evaluation
# you should have these files: glob.glob(f"/scratch/v46/ls7238/CAT_turbulence/{turbulence_index}/{P}hPa/freq-above-p99/{turbulence_index}-{P}hPa-monthly-freq-above-p99_AUS-15_*_BOM_BARPA-R_v1-r1_6hr.nc") 
# and glob.glob(f"/scratch/v46/ls7238/CAT_turbulence/{turbulence_index}/{P}hPa/percentiles/{turbulence_index}-{P}hPa-monthly-percentiles_AUS-15_*_BOM_BARPA-R_v1-r1_6hr.nc")

import xarray as xr
import glob
import numpy as np
import seaborn as sns
from scipy import stats
import calendar
import datetime
import pandas as pd
from turbulence_AUSCAT.cat_indices import calc_turbulence_indices, windspeed, VWS
from xarray.groupers import SeasonResampler
import matplotlib.pyplot as plt

# Letty edit path:
path= "/scratch/v46/ls7238/CAT_turbulence"

mid_lat_slice = slice(-50,-25)
lon_slice = slice(90,195)

baseline_time_range = np.arange(1990,2009+1)
baseline_time_slice = slice("1990", "2009")

P0 = 1000
step_size=  0.1545


list_evaluation = ['evaluation_BARRA-R_r1i1p1f1',
                   # 'evaluation_ERA5_r1i1p1f1',
                  ]

list_historical = ['historical_ACCESS-CM2_r4i1p1f1', 
                   'historical_ACCESS-ESM1-5_r6i1p1f1',
                   'historical_CESM2_r11i1p1f1', 
                   'historical_CMCC-ESM2_r1i1p1f1',
                   'historical_EC-Earth3_r1i1p1f1',
                   'historical_MPI-ESM1-2-HR_r1i1p1f1',
                   'historical_NorESM2-MM_r1i1p1f1',
                  ]

list_ssp126 = [
                 'ssp126_ACCESS-CM2_r4i1p1f1', 
                 'ssp126_ACCESS-ESM1-5_r6i1p1f1',
                 'ssp126_CESM2_r11i1p1f1',
                 'ssp126_CMCC-ESM2_r1i1p1f1',
                 'ssp126_EC-Earth3_r1i1p1f1',
                 'ssp126_MPI-ESM1-2-HR_r1i1p1f1',
                 'ssp126_NorESM2-MM_r1i1p1f1',
              ]

list_ssp370 = ['ssp370_ACCESS-CM2_r4i1p1f1',
                 'ssp370_ACCESS-ESM1-5_r6i1p1f1',
                 'ssp370_CESM2_r11i1p1f1',
                 'ssp370_CMCC-ESM2_r1i1p1f1',
                 'ssp370_EC-Earth3_r1i1p1f1',
                 'ssp370_MPI-ESM1-2-HR_r1i1p1f1',
                 'ssp370_NorESM2-MM_r1i1p1f1',
              ]

list_ssp585 = ['ssp585_ACCESS-CM2_r4i1p1f1',
                 'ssp585_EC-Earth3_r1i1p1f1']

list_future = list_ssp126 + list_ssp370 + list_ssp585

time_sel_dict = {"annual":"year",
                 "MJJASO" : "6M",
                 "NDJFMA" : "6M",
                 "DJF":"season",
                 "MAM":"season",
                 "JJA":"season",
                 "SON":"season",
                 "January": "month",
                 "February": "month",
                 "March": "month",
                 "April": "month",
                 "May": "month",
                 "June": "month",
                 "July": "month",
                 "August": "month",
                 "September": "month",
                 "October": "month",
                 "November": "month",
                 "December": "month",
                }

# # Make tables for evaluation:

def _trend_table_rows(da, run, start_year, end_year, time_selection):
    """Calculates the data to put into the trend table.
    Columns are  [["run", "time_slice", "time_selection", "mon/seas/yr", "slope", "intercept", "pvalue", "significance"]].
    da is the preselected xarray data array. eg ds_ann[turbulence_index].sel({"run":run})
    start_year and end_year define the years over which to calculate the linear regression
    """
    try:
        slope, intercept, _, pvalue, _ = stats.linregress(x=np.arange(start_year, end_year+1),
                                                          y=da.sel({"time":slice(str(start_year), str(end_year))}),)
    except:
        slope, intercept, _, pvalue, _ = stats.linregress(x=np.arange(start_year, end_year),
                                                          y=da.sel({"time":slice(str(start_year), str(end_year))}),)

    # set symbol for significance
    significance = "**" if pvalue < 0.01 else "*" if pvalue < 0.05 else ""

    # row to update the trend table
    return[run, f"{start_year}-{end_year}", time_selection, time_sel_dict[time_selection], f"{slope:.5f}", f"{intercept:.5f}", f"{pvalue:.3f}", significance]

def select_resample_time(ds, time_selection):
    if time_selection == "annual":
        ds= ds.resample({"time":"YE"}).mean()
    elif time_selection in ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']:
        ds= ds.sel({"time": (ds["time"].dt.month== datetime.datetime.strptime(time_selection, '%B').month)},)
    elif time_selection in ["MJJASO", "NDJFMA"]:
        ds= ds.resample({"time":SeasonResampler([time_selection], drop_incomplete=True)}).mean()\
              .resample({"time":"YE-OCT"}).mean()
    elif time_selection in ["DJF", "MAM", "JJA", "SON"]:
        ds= ds.resample({"time":SeasonResampler([time_selection], drop_incomplete=True)}).mean()
    else:
        print("time_selection  must be one of [annual, MJJASO, NDJFMA, DJF, MAM, JJA, SON, January, February, March, April, May, June, July, August, September, October, November, December]")
    return ds

def _trend_table_per_run(ds_ts, trend_table, run, start_year, end_year, turbulence_index):
    """Takes the dataset time series, then for each run and specified start year and end year, 
    add to the exisitng trend_table.
    Iterates through annual, seasons and months"""
    ds_run = ds_ts.sel({"run": run,  "time":slice(str(start_year), str(end_year))})

    for time_selection in list(time_sel_dict.keys()):
        da=select_resample_time(ds_run, time_selection)[turbulence_index]
        trend_table.append(_trend_table_rows(da, run, start_year=start_year, end_year=end_year, 
                                              time_selection=time_selection))

    return trend_table

def calc_trend_table(ds_ts, turbulence_index, P, outfile=None,):
    """Makes a table for all runs and time selections which reports the trends' slope, intercept and pvalue.
    Marks significance at 0.01 and 0.05.
    """
    trend_table = [["run", "time_slice", "time_selection", "mon/seas/yr", "slope", "intercept", "pvalue", "significance"]]

    start_year, end_year = (1990, 2009)
    for run in list_evaluation:
        trend_table = _trend_table_per_run(ds_ts, trend_table, run, start_year, end_year, turbulence_index)
        
    start_year, end_year = (1979, 2014)
    for run in list_historical:
        trend_table = _trend_table_per_run(ds_ts, trend_table, run, start_year, end_year, turbulence_index)
    
    start_year, end_year = (2015, 2100)
    for run in list_future:
        trend_table = _trend_table_per_run(ds_ts, trend_table, run, start_year, end_year, turbulence_index)
    
    trend_df = pd.DataFrame(columns = trend_table[0], data = trend_table[1:])
    if outfile is None:
        outfile = f"{path}/{turbulence_index}/{P}hPa/trend_table_{turbulence_index}-{P}hPa.csv"
    trend_df.to_csv(outfile)
    return trend_df
    



# ## Kovmogolov-Smirnov two sample test for model evaluation

# evaluation
evaluation = ['evaluation_BARRA-R_r1i1p1f1']

def plot_kstest(ds_eval, ds_hist, turbulence_index, time_selection, evaluation, run, pvalue, significance, ax=None, show_years=False):
    if ax is None:
        fig, ax = plt.subplots()
    ax.set_title(f"{run} in {time_selection}")        

    if show_years:
        # show each single year
        for year in baseline_time_range:
            sns.lineplot(data = ds_eval.sel({"time": (ds_eval.time.dt.year==year)}).mean("time"),
                         x=turbulence_index,
                         y="quantile",
                         color="r",
                         alpha=0.1,
                         ax=ax)
        for year in baseline_time_range:
            sns.lineplot(data = ds_hist.sel({"time": (ds_hist.time.dt.year==year)}).mean("time"),
                         x=turbulence_index,
                         y="quantile",
                         color="k",
                         alpha=0.1,
                         ax=ax)
            
    # empirical CDF for ks test        
    sns.ecdfplot(data = ds_eval.mean("time")[turbulence_index], color="k", label=evaluation, ax=ax,)
    sns.ecdfplot(data = ds_hist.mean("time")[turbulence_index], color="r", label=run, ax=ax,)
    
    # add pvalue to plot
    ax.text(x=0.98, y=0.02, s= f"pvalue = {pvalue:.3f}{significance}", transform = ax.transAxes, va="bottom", ha="right",)
    ax.set_xlabel(turbulence_index)
    ax.legend(loc="upper left", fontsize=10)
    return 

def select_resample_time(ds, time_selection):
    if time_selection == "annual":
        ds= ds.resample({"time":"YE"}).mean()
    elif time_selection in ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']:
        ds= ds.sel({"time": (ds["time"].dt.month== datetime.datetime.strptime(time_selection, '%B').month)},)
    elif time_selection in ["MJJASO", "NDJFMA"]:
        ds= ds.resample({"time":SeasonResampler([time_selection], drop_incomplete=True)}).mean()\
              .resample({"time":"YE-OCT"}).mean()
    elif time_selection in ["DJF", "MAM", "JJA", "SON"]:
        ds= ds.resample({"time":SeasonResampler([time_selection], drop_incomplete=True)}).mean()
    else:
        print("time_selection  must be one of [annual, MJJASO, NDJFMA, DJF, MAM, JJA, SON, January, February, March, April, May, June, July, August, September, October, November, December]")
    return ds

def calc_kstest_table(ds = None, 
                      turbulence_index=None,
                      P=None,
                      evaluation = ['evaluation_BARRA-R_r1i1p1f1'], 
                      list_historical=list_historical, 
                      outfile=None,
                      show_plot = False):
    """"""
    kstest_table = [["sample1", "sample2", "time_selection", "mon/seas/yr", "pvalue", "significance"]]
    for time_selection in list(time_sel_dict.keys()):
        ds_tmp = select_resample_time(ds, time_selection)[turbulence_index]
        data_eval = ds_tmp.sel({"run":evaluation}).stack(z=[...])
        for run in list_historical:
            data_hist = ds_tmp.sel({"run":run}).stack(z=[...])
            # calculate pvalue
            ks_statistic, pvalue = stats.ks_2samp(data_eval, data_hist,)
            significance = "**" if pvalue < 0.01 else "*" if pvalue < 0.05 else ""
            
            # update ks test stats
            kstest_table.append([evaluation, run, time_selection, time_sel_dict[time_selection], f"{pvalue:.3f}", significance])
            
            if show_plot:
                plot_kstest(ds_eval, ds_hist, time_selection, evaluation, run, pvalue)

    kstest_df = pd.DataFrame(columns = kstest_table[0], data = kstest_table[1:]).sort_values(by='sample2', ascending=False)
    if outfile is None:
        outfile = f"{path}/{turbulence_index}/{P}hPa/evaluation_kstest_table_{turbulence_index}-{P}hPa.csv"
    kstest_df.to_csv(outfile)
    return kstest_df

def calc_ttest_table(ds=None, q=0.33, turbulence_index=None, P = None, evaluation = evaluation, path=path,  outfile=None,):
    """"""
    # q ttest value in the mid lat box to evaluation spatial conherence
    ttest_table = [["sample1", "sample2", "time_selection", "mon/seas/yr", f"p{int(q*100)}_pvalue", "significance"]]
    for time_selection in list(time_sel_dict.keys()):
        ds_tmp = select_resample_time(ds, time_selection)[turbulence_index]
        ds_eval = ds_tmp.sel({"run":evaluation})
        for run in list_historical:
            ds_hist = ds_tmp.sel({"run":run})
        
            # calculate pvalue
            stat, pvals = stats.ttest_ind(ds_eval, ds_hist)
        
            pvalue = xr.Dataset(data_vars={"pval" : (["lat", "lon"],  pvals),}, 
                                coords= {"lat": ds_hist.lat, "lon": ds_hist.lon},
                               )["pval"].quantile(q, dim=["lat", "lon"])
            # mark significance
            significance = "**" if pvalue < 0.01 else "*" if pvalue < 0.05 else ""
    
            # update t test stats
            ttest_table.append([evaluation, run, time_selection, time_sel_dict[time_selection], f"{pvalue:.3f}", significance])
        
            if False:
                plot_kde_test(ds_eval, ds_hist, time_selection, evaluation, run, pvalue)
    
    ttest_df = pd.DataFrame(columns = ttest_table[0], data = ttest_table[1:]).sort_values(by='sample2', ascending=False)
    if outfile is None:
        outfile = f"{path}/{turbulence_index}/{P}hPa/evaluation_ttest_{turbulence_index}-{P}hPa_p{int(q*100)}_table.csv"
    ttest_df.to_csv(outfile)
    return ttest_df


def combined_significance_table(turbulence_index, P, path=path):
    ds = xr.open_mfdataset(glob.glob(f"{path}/{turbulence_index}/{P}hPa/percentiles/{turbulence_index}-{P}hPa-monthly-percentiles_AUS-15_*_BOM_BARPA-R_v1-r1_6hr.nc"),
                      combine="nested",
                      concat_dim="run",
                          compat='no_conflicts')
    ktest_df = calc_kstest_table(ds = ds, 
                                  turbulence_index=turbulence_index,
                                  P=P,
                                  evaluation = evaluation[0], 
                                  list_historical=list_historical, 
                                  outfile=None,
                                )
    
    ds = xr.open_mfdataset(glob.glob(f"{path}/{turbulence_index}/{P}hPa/freq-above-p99/{turbulence_index}-{P}hPa-monthly-freq-above-p99_AUS-15_*_BOM_BARPA-R_v1-r1_6hr.nc"),
                           combine="nested",
                           concat_dim="run",
                           coords="different",
                           join="outer",
                           compat='no_conflicts').sel({"time":baseline_time_slice, "lon":lon_slice })
    q=0.33
    ttest_df = calc_ttest_table(ds = ds,
                                 q=q, 
                                 turbulence_index=turbulence_index,
                                 P = P, 
                                 evaluation = evaluation[0], )

    # read the test p values from the saved files
    evaluation_kstest = pd.read_csv(f"{path}/{turbulence_index}/{P}hPa/evaluation_kstest_table_{turbulence_index}-{P}hPa.csv", index_col=0,).replace(np.nan, "")
    evaluation_ttest = pd.read_csv(f"{path}/{turbulence_index}/{P}hPa/evaluation_ttest_{turbulence_index}-{P}hPa_p{int(q*100)}_table.csv", index_col=0,)[["p33_pvalue", "significance"]].replace(np.nan, "")

    evaluation_combined = pd.concat([evaluation_kstest, evaluation_ttest], axis=1, join="inner")
    evaluation_combined["combined_significance"] = evaluation_kstest["significance"] + evaluation_ttest["significance"]
    evaluation_combined.to_csv(f"{path}/{turbulence_index}/{P}hPa/evaluation_combined_tests_table_{turbulence_index}-{P}hPa.csv")
    return evaluation_combined




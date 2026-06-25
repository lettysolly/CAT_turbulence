"""originally from https://github.com/AusClimateService/plotting_maps
Standardising Australia Hazard Maps
This module plots maps to consistently present climate hazards for Australia.
It is code is designed to work with hh5 analysis3-24.04 venv"""


import seaborn as sns
import pandas as pd
from turbulence_AUSCAT.auscat_plots import *
import calendar
from scipy import stats
from xarray.groupers import SeasonResampler, SeasonGrouper
from matplotlib.lines import Line2D
from IPython.display import display

import datetime
import os

# import packages used in this workflow
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image, cm, colors
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import math

from glob import glob

# import colormap packages
import cmaps
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap

from shapely.geometry import box
import shapely
# set tolerance for simplifying geometries
# this tolerance is approx 110 m resolution for lat lon data
tolerance=0.001

import matplotlib as mpl
mpl.rcParams['hatch.linewidth'] = 0.3 
plt.rcParams['savefig.facecolor']='white'

# define some standard imput for the maps
projection = ccrs.LambertConformal(
    central_latitude=-24.75,
    central_longitude=134.0,
    cutoff=30,
    standard_parallels=(-10, -40),
)

from pathlib import Path

# Set title sizes according to ACS comms guidance (in pt)
fontsize_title = 14
fontsize_subtitle = 12
fontsize_cbar = 10
fontsize_footnote = 8
padding = 5

# # Suggested colormaps and scales
# Using suggested colormaps and scales will improve the consistency across teams
# producing similar variables. This will support comparison across different plots.
# - see many colormaps here:
# https://www.ncl.ucar.edu/Document/Graphics/color_table_gallery.shtml
# This suggested colormaps are matched with possible variables to plot.
# This includes color maps for the total amount and anomalies

#ipcc colormaps from github.com/IPCC-WG1/colormaps/

cmap_mustard = LinearSegmentedColormap.from_list(
    "mustard",
    ["#5a4c10", "#977f1b","#d3b125", "#e2c85e",  "#eddd9b",]
)
cmap_mustard.set_bad(color="lightgrey")

cmap_BuGnPi = LinearSegmentedColormap.from_list('BuGnPi', np.vstack((cm.BrBG(np.linspace(0.55, 1, 30)[::-1]),[0.9,0.9,0.9,0.9], cm.pink_r(np.linspace(0.25, 1, 30)) )))
cmap_BuGnPi_r = LinearSegmentedColormap.from_list('BuGnPi_r',cmap_BuGnPi(np.linspace(0,1,256))[::-1])

cmap_dir = f"{Path(__file__).parent}/continuous_colormaps_rgb_0-1"

cmap_dict = {
    "sst": cmaps.cmocean_tempo,
    "sst_anom": cmaps.cmocean_balance_r,
    "mhw_days": cm.YlOrRd,
    "mhw_intensity": cm.hot_r,
    "hot_r": cm.hot_r,
    "surface_pH": cm.YlGnBu,
    "surface_aragonite_sat": cmaps.cmocean_delta,
    "tas": cm.Spectral_r,
    "tas_anom": cm.RdBu_r,
    "tas_anom_1": cm.seismic,
    "tas_deciles_bwr": cm.bwr,
    "EHF_days": cm.YlOrRd,
    "EHF_days_1": cm.YlOrBr,
    "EHF_duration": cm.hot_r,
    "AFDRS_category": ListedColormap(["white", "green", "orange", "red", "darkred"]),
    "ffdi_category": ListedColormap(
        ["green", "blue", "yellow", "orange", "red", "darkred"]
    ),
    "fire_climate": ListedColormap(
        [ "#84a19b", "#e0d7c6", "#486136", "#737932", "#a18a6e", ]
    ),
    "fire_climate_alternative": ListedColormap(
        [ "#355834", "#F1F5F2", "#14281D", "#6E633D", "#C2A878", ]
    ),
    'tasmax_bom': ListedColormap(
        [ '#E3F4FB','#C8DEE8','#91C4EA','#56B6DC','#00A2AC','#30996C',
         '#7FC69A','#B9DA88','#DCE799', '#FCE850','#EACD44','#FED98E',
         '#F89E64','#E67754','#D24241', '#AD283B','#832D57','#A2667A','#AB9487']
    ), #not colourblind safe
    'tasmax' : ListedColormap(
        ['#014636','#016c59','#02818a','#3690c0','#67a9cf','#a6bddb',
         '#d0d1e6','#ece2f0','#fff7fb','#ffffcc','#ffeda0','#fed976',
         '#feb24c','#fd8d3c','#fc4e2a','#e31a1c','#bd0026','#800026',
         '#510019','#2E000E']
    ),
    "pr": cm.YlGnBu,
    "pr_1": cmaps.cmocean_deep,
    "pr_days": cm.Blues,
    "pr_GMT_drywet": cmaps.GMT_drywet,
    "pr_anom": cm.BrBG,
    "pr_anom_1": cmaps.cmocean_curl,
    "pr_anom_12lev": cmaps.precip_diff_12lev,
    "pr_chance_extremes": cmaps.cmocean_solar_r,
    "tc_days": cm.RdPu,
    "tc_intensity": cm.PuRd,
    "tc_days_anom": cm.PuOr,
    "tc_intensity_anom": cm.PiYG_r,
    "xts_freq": cmaps.cmocean_dense,
    "xts_intensity": cmaps.cmocean_matter,
    "xts_freq_anom": cmaps.cmocean_balance_r,
    "xts_intensity_anom": cmaps.cmocean_curl_r,
    "drought_severity": cm.RdYlGn, # not colorblind friendly
    "drought_severity_r": cm.RdYlGn_r, # not colorblind friendly
    "drought_duration": cmaps.hotres,
    "drought_duration_r": cmaps.hotres_r,
    "aridity": cmap_mustard,
    # "aridity_anom": cmaps.NEO_div_vegetation_a, # not colorblind friendly
    # "aridity_anom_r": cmaps.NEO_div_vegetation_a_r, # not colorblind friendly
    "aridity_anom": cmap_BuGnPi_r,
    "aridity_anom_r": cmap_BuGnPi,
    "BrBu": LinearSegmentedColormap.from_list("BrBu", ["#3c320a", "#d3b125", "lightgrey", "royalblue", "navy"]),
    "BuBr": LinearSegmentedColormap.from_list("BuBr", ["navy", "royalblue", "lightgrey", "#d3b125", "#3c320a"]),
    "anom_BlueYellowRed": cmaps.BlueYellowRed,
    "anom_BlueYellowRed_r": cmaps.BlueYellowRed_r,
    "anom": cmaps.BlueWhiteOrangeRed,
    "anom_r": cmaps.BlueWhiteOrangeRed_r,
    "anom_b2r": cmaps.cmp_b2r,
    "anom_b2r_r": cmaps.cmp_b2r_r,
    "anom_coolwarm": cmaps.MPL_coolwarm,
    "anom_coolwarm_r": cmaps.MPL_coolwarm_r,
    "anom_deciles": cm.bwr,
    "anom_deciles_r": cm.bwr_r,
    "anom_veg_1": cmaps.NEO_div_vegetation_a, # not colorblind friendly
    "anom_veg_1_r": cmaps.NEO_div_vegetation_a_r, # not colorblind friendly
    "BuGnRd": cmaps.temp1,
    "rh_19lev": cmaps.rh_19lev,
    "sunshine_9lev": cmaps.sunshine_9lev,
    "sunshine_diff_12lev": cmaps.sunshine_diff_12lev,
    "inferno": cm.inferno,
    "Oranges": cm.Oranges,
    "Oranges_r": cm.Oranges_r,
    "OrRd": cm.OrRd,
    "Greens": cm.Greens,
    "topo": cmaps.OceanLakeLandSnow,
    "gmt_relief": cmaps.GMT_relief,
    "ipcc_chem_div": LinearSegmentedColormap.from_list('colormap', np.loadtxt(f"{cmap_dir}/chem_div.txt")),
    "ipcc_chem_seq": LinearSegmentedColormap.from_list('colormap', np.loadtxt(f"{cmap_dir}/chem_seq.txt")),
    "ipcc_cryo_div": LinearSegmentedColormap.from_list('colormap', np.loadtxt(f"{cmap_dir}/cryo_div.txt")),
    "ipcc_cryo_seq": LinearSegmentedColormap.from_list('colormap', np.loadtxt(f"{cmap_dir}/cryo_seq.txt")),
    "ipcc_misc_div": LinearSegmentedColormap.from_list('colormap', np.loadtxt(f"{cmap_dir}/misc_div.txt")),
    "ipcc_misc_seq_1": LinearSegmentedColormap.from_list('colormap', np.loadtxt(f"{cmap_dir}/misc_seq_1.txt")),
    "ipcc_misc_seq_2": LinearSegmentedColormap.from_list('colormap', np.loadtxt(f"{cmap_dir}/misc_seq_2.txt")),
    "ipcc_misc_seq_3": LinearSegmentedColormap.from_list('colormap', np.loadtxt(f"{cmap_dir}/misc_seq_3.txt")),
    "ipcc_prec_div": LinearSegmentedColormap.from_list('colormap', np.loadtxt(f"{cmap_dir}/prec_div.txt")),
    "ipcc_prec_seq": LinearSegmentedColormap.from_list('colormap', np.loadtxt(f"{cmap_dir}/prec_seq.txt")),
    "ipcc_slev_div": LinearSegmentedColormap.from_list('colormap', np.loadtxt(f"{cmap_dir}/slev_div.txt")),
    "ipcc_slev_seq": LinearSegmentedColormap.from_list('colormap', np.loadtxt(f"{cmap_dir}/slev_seq.txt")),
    "ipcc_temp_div": LinearSegmentedColormap.from_list('colormap', np.loadtxt(f"{cmap_dir}/temp_div.txt")),
    "ipcc_temp_seq": LinearSegmentedColormap.from_list('colormap', np.loadtxt(f"{cmap_dir}/temp_seq.txt")),
    "ipcc_wind_div": LinearSegmentedColormap.from_list('colormap', np.loadtxt(f"{cmap_dir}/wind_div.txt")),
    "ipcc_wind_seq": LinearSegmentedColormap.from_list('colormap', np.loadtxt(f"{cmap_dir}/wind_seq.txt")),
    "acs_geophysical_biochemical_div1": cm.PRGn,
    "acs_geophysical_biochemical_div2": cm.PiYG,
    "acs_geophysical_biochemical_seq1": cm.BuGn, 
    "acs_geophysical_biochemical_seq2": cm.RdPu, 
    "acs_geophysical_biochemical_seq3": cm.Greys, 
    "acs_precipitation_div1": cm.RdBu,
    "acs_precipitation_div2": cm.BrBG, 
    "acs_precipitation_seq1": cm.Blues,
    "acs_precipitation_seq2": cm.GnBu, 
    "acs_precipitation_seq3": cm.PuBuGn,
    "acs_temperature_div1": cm.RdYlBu_r,
    "acs_temperature_div2": cm.PuOr_r,
    "acs_temperature_seq1": cm.YlOrRd, 
    "acs_temperature_seq2": cm.Reds,
    "acs_temperature_seq3": cm.YlGnBu, 
}


# Here are some suggestions for the ticks/ scale for some variables.
# Some scales are taken from climate maps on bom.gov.au/climate
tick_dict = {
    "pr_annual": [0, 50, 100, 200, 300, 400, 600, 1000, 1500, 2000, 3000, 6000],
    "pr_6mon": [0, 50, 100, 200, 300, 400, 600, 900, 1200, 1800, 2400, 6000],
    "pr_3mon": [0, 10, 25, 50, 100, 200, 300, 400, 600, 800, 1200, 2500],
    "pr_mon": [0, 1, 5, 10, 25, 50, 100, 200, 300, 400, 600, 1200],
    "pr_hour": [0, 1, 2, 5, 10, 15, 20, 30, 50, 75, 100, 200,],
    "pr_days": [0, 2, 3, 5, 10, 20, 30, 40, 50, 75, 100, 125, 150, 175],
    "pr_anom_mon": [ -1000, -400, -200, -100, -50, -25, -10, 0, 10, 25, 50, 100, 200, 400, 1000,],
    "pr_anom_3mon": [ -2000, -600, -400, -200, -100, -50, -25, 0, 25, 50, 100, 200, 400, 600, 2000,],
    "pr_anom_6mon": [ -3000, -1200, -800, -400, -200, -100, -50, 0, 50, 100, 200, 400, 800, 1200, 3000,],
    "pr_anom_ann": [ -4000, -1800, -1200, -800, -400, -200, -100, 0, 100, 200, 400, 800, 1200, 1800, 4000,],
    "pr_diff_mon": [ -1000, -400, -200, -100, -50, -25, -10, 10, 25, 50, 100, 200, 400, 1000,],
    "pr_diff_ann": [ -3000, -1800, -1200, -800, -400, -200, -100, 100, 200, 400, 800, 1200, 1800, 3000,],
    "frost_days": [0, 10, 20, 30, 40, 50, 75, 100, 150, 300],
    "frost_days_mon": [0, 2, 5, 10, 15, 20, 25, 31],
    "tas": np.arange(-9, 52, 3),
    "tas_anom_day": np.arange(-14, 14.1, 2),
    "tas_anom_mon": np.arange(-7, 7.1, 1),
    "tas_anom_ann": np.arange(-3.5, 3.6, 0.5),
    "apparent_tas": np.arange(-6, 42, 3),
    "percent": np.arange(0, 101, 10),
    "xts_freq": [0.00, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15],
    "fire_climate_ticks": [ 100, 101, 102, 103, 104, ],
    "fire_climate_labels": [
        "Tropical Savanna",
        "Arid grass \nand woodland",
        "Wet Forest",
        "Dry Forest",
        "Grassland",
    ],
    "aridity_index_ticks": [0.0, 0.05, 0.2, 0.5, 0.65],
    "aridity_index_labels": ["Hyper-arid", "Arid", "Semi-arid", "Dry sub-humid"],
}

# # Load the State and Region shape files
class RegionShapefiles:
    """Load and return a shapefile based on its name."""

    def __init__(self, path, shapefiles):
        """Create an instance of the RegionShapefiles class.
        Parameters
        ----------
        path : str
            The path to the shapefiles directory.
        shapefiles : list
            A list of shapefile names to load.
        """
        self.path = path
        self.shapefiles = shapefiles
        self._regions_dict = {}

    def __call__(self, name):
        """Retrieve the shapefile for the given region name.
        Parameters
        ----------
        name : str
            The name of the region to retrieve.
        Returns
        -------
        GeoDataFrame or GeoSeries
            The shapefile data for the specified region.
        """
        if name not in self._regions_dict:
            if name in self.shapefiles:
                regions = gpd.read_file(glob(f"{self.path}/{name}/*.shp")[0]).to_crs(crs = "GDA2020")
                regions[["geometry"]] =shapely.simplify(regions[["geometry"]], tolerance)
                self._regions_dict[name] = regions
                
            elif name == "not_australia":
                # Define a white mask for the area outside of Australian land
                # We will use this to hide data outside the Australian land borders.
                # note that this is not a data mask,
                # the data under the masked area is still loaded and computed, but not visualised
                base_name = name[4:]  # Remove 'not_' prefix
                base_region = self(base_name).copy().to_crs(crs = "GDA2020")
                base_region[["geometry"]] =shapely.simplify(base_region[["geometry"]], tolerance)
                
                # This mask is a rectangular box around the maximum land extent of Australia
                # with a buffer of 20 degrees on every side,
                # with the Australian land area cut out, only the ocean is hidden.
                not_region = gpd.GeoSeries(
                    data=[
                        box(*box(*base_region.total_bounds).buffer(20).bounds
                        ).difference(base_region["geometry"].values[0])
                    ],
                    crs=ccrs.PlateCarree(),
                )
                self._regions_dict[name] = not_region
            else:
                raise ValueError(f"Shapefile for region '{name}' not found.")
        return self._regions_dict[name]

    def __getitem__(self, name):
        if name not in self._regions_dict:
            self(name)
        return self._regions_dict[name]

    def __setitem__(self, name, value):
        self._regions_dict[name] = value

    def keys(self):
        return self._regions_dict.keys()

    def __len__(self):
        return len(self._regions_dict)

    def __repr__(self):
        return repr(self._regions_dict)

    def update(self, *args, **kwargs):
        self._regions_dict.update(*args, **kwargs)


# Define the path and shapefiles
# These will be used for state boundaries, LGAs, NRM, etc
PATH = "/g/data/ia39/aus-ref-clim-data-nci/shapefiles/data"
shapefile_list = ["aus_local_gov",
                  "aus_states_territories",
                  "australia", 
                  "broadacre_regions", 
                  "NCRA_Marine_region",
                  "ncra_regions", 
                  "NCRA_regions_coastal_waters_GDA94", 
                  "nrm_regions",
                  "plantations"]

# Create an instance of the RegionShapefiles class
regions_dict = RegionShapefiles(PATH, shapefile_list)

# define a white mask for the area outside of Australian land
# We will use this to hide data outside of the Australian land borders.
# note that this is not a data mask,
# the data under the masked area is still loaded and computed, but not visualised
australia = regions_dict["australia"]

# This mask is a rectangular box around the maximum land extent of Australia
# with a buffer of 10 degrees on every side,
# with the Australian land area cut out so only the ocean is hidden.
not_australia =  regions_dict["not_australia"]

def crop_cmap_center(cmap, ticks, mid, extend=None):
    """
    This function is used to align the centre of a colormap (cmap)
    to a specified midpoint (mid). Allows the cmap to be normalised on 
    the specified ticks with cbar_extend arrows taken into account.
    Intended for divergent colormaps that show anomalies that are mostly
    all positive (or negative), for example, temperature anomalies in 
    future climate projections.
    The shorter side of the colormap is cropped and not stretched - unlike 
    matplotlib.colors.TwoSlopeNorm.

    Parameters
    -------
    cmap: matplotlib colormap
        original colormap to crop

    ticks: list or np.array
        list or array of numerical ticks

    mid: float
        Set the midpoint value of the colormap. 
        For example, 0.0

    extend: {'neither', 'both', 'min', 'max'}
        Make pointed end(s) for out-of-range values (unless 'neither').
        These are set for a given colormap using the colormap set_under
        and set_over methods.
    
    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
    """
    ticks=np.array(ticks)
    # number of color segments:
    below = (ticks<mid).sum()
    above = (ticks>mid).sum()
    if extend =="both" or extend == "max":
        above=above+1
    if extend =="both" or extend =="min":
        below=below+1
    
    #total segments
    N = below+above
    
    #porportion below mid point 
    prop_below_mid = below/(max(below,above)+1)
    # propotion above mid point
    prop_above_mid = above/(max(below,above)+1)

    bounds = np.linspace(0.5*(1-prop_below_mid),
                     0.5*(1+prop_above_mid),
                     N+1)
    between_bounds = [(n0+n1)/2 for n0, n1 in zip(bounds[:-1], bounds[1:])]
    
    new_cmap_list = cmap(between_bounds)
    new_cmap = LinearSegmentedColormap.from_list("new_cmap",new_cmap_list, N)
    return new_cmap


# Define subfunctions for different parts of the plotting 
# so that they can be reused for single panel and multi panel plots
def plot_data(regions=None,
              data=None, 
              station_df = None,
              markersize=None,
              stippling=None,
              shading=None,
              xlim=(114, 162),
              ylim=(-43.5, -7.5),
              cmap=cm.Greens,
              cbar_extend="both",
              ticks=None,
              tick_interval=1,
              tick_labels=None,
              contourf=False,
              contour=False,
              ax=None,
              subtitle = "",
              subtitle_xy = None,
              facecolor="none",
              edgecolor="k",
              area_linewidth=0.3,
              coastlines=False,
              mask_not_australia = True,
              mask_australia=False,
              agcd_mask=False,
              select_area = None,
              vcentre=None,
              latlongrid=True,
              panel = None,
             ):
    """This function takes one axis and plots the hazard data to one map of Australia. 
    This function takes gridded "data" and/or a "station_df" dataframe and "regions" shapefiles 
    to visualise hazard data from a 2D Xarray data array and plots the data on a map
    of Australia with the regions outlines.

    Parameters
    ----------
    regions: geopandas.GeoDataFrame
        region geometries for regions/states/catchments etc

    data: xr.DataArray
        a 2D xarray DataArray which has already computed the 
        average, sum, anomaly, metric or index you wish to visualise.
        This function is resolution agnostic.

    station_df: pd.DataFrame, optional
        a pandas.DataFrame with columns ["lon", "lat", variable]. 
        If station_df is given, then variable values are represented as dots on 
        the mapaccordingg to the lat lon coordinates and coloured according to
        cmap colors and ticks.

    markersize: int, optional
        default None. If None then the markersize will adjust to the size of the
        figure and the number of markers in the plot such that when there are
        many markers and the figure is small, the markersize is smaller.

    stippling: xr.DataArray, optional
        a True/False mask to define regions of stippling hatching. 
        Intended to show information such as "model agreement for direction of change".

    shading: xr.DataArray, optional
        A list of True/False masks to define regions of dotted stippling  
        for each subplot. ["....."] 
        Intended to obscure data outside the area of interest.

    xlim: tuple, optional
        longitude min and max of the plot area.
        default is cropped to Australian continent xlim=(114, 162)

    ylim: tuple, optional
        latitude min and max of the plot area.
        default is cropped to Australian continent ylim=(-43.5, -7.5),

    cmap:
        defines the colormap used for the data.
        See cmap_dict for suggested colormaps.
        Default cmap set to cm.Greens.
        Please choose appropriate colormaps for your data.

    cbar_extend: one of {'neither', 'both', 'min', 'max'}.
        eg "both" changes the ends of the colorbar to arrows to indicate that
        values are possible outside the scale shown.
        If contour or contourf is True, then cbar_extend will be overridden to "none".

    ticks: list or arraylike
        Define the ticks on the colorbar. Define any number of intervals. 
        This will make the color for each interval one discrete color, 
        instead of a smooth color gradient.
        If None, linear ticks will be auto-generated to fit the provided data.

    tick_interval: int
        Default 1
        For showing every second tick label, set tick_interval=2

    tick_labels: list
        Labels for categorical data. 
        If tick_labels is used, then pcolormesh is used to plot data 
        and does not allow contour or contourf to be used.
        Tick labels will correspond to the ticks.

    contourf: bool
        if True then the gridded data is visualised as smoothed filled contours. 
        Default is False.
        Use with caution when plotting data with negative and positive values;
        Check output for NaNs and misaligned values.  
        High resolution data is slow to compute.

    contour: bool
        if True then the gridded data is visualised as smoothed unfilled grey contours.
        Default is False.
        High resolution data is slow to compute.
        Using both contourf and contour results in smooth filled contours
        with grey outlines between the color levels.

    ax: matplotlib.axes.Axes
        Axes object of existing figure to put the plotting.

    subtitle: str
        default ""
        Intended to label global warming levels for subplots eg. "GWL 1.2"
        
    subtitle_xy: tuple, optional
        (x, y) location of subtitle relative to transAxes.
        defines the top left location for the subtitle. 

    facecolor: color
        color of land when plotting the regions without climate data and select_area is None. 
        facecolor recommendations include "white", "lightgrey", "none".
        default is "none"

    edgecolor: color
        defines the color of the state/region borders. 
        edgecolor recommendations include "black" and "white".

    mask_not_australia: boolean
        decides whether or not the area outside of Australian land is hidden 
        under white shape.
        Default is True.

    mask_australia: boolean
        decides whether or not Australian land is hidden under white shape.
        Eg, use when plotting ocean only.
        Default is False.

    agcd_mask: boolean
        If True, applies a ~5km mask for data-sparse inland areas of Australia.
        Default is False.

    area_linewidth: float, optional
        the width of state/region borders only. All other linewidths are hardcoded.

    coastlines: boolean
        If True, add cartopy coastlines for all coasts (not just Australia). 
        Default is False.

    select_area: list
        If None, then don't add region borders geometries.
        
    vcentre: float, eg 0
        default is None.
        Align centre of colormap to this value. 
        Intended for using a divergent colormap with uneven number of ticks 
        around the centre, eg for future temperature anomalies with a larger
        positive range compared to the negative range.

    Returns
    -------
    ax, norm, cont, middle_ticks
    ax, the matplotlib.axes.Axes with the plot
    norm, the normalisation for the colormap and plotted contours according to ticks
    cont, the QuadContourSet or QuadMesh of the plotted gridded data
    middle_ticks, the location to label categorical tick labels
    """

    if vcentre is not None:
        cmap = crop_cmap_center(cmap, ticks, vcentre, extend=cbar_extend)

    ax.set_extent([xlim[0], xlim[1], ylim[0], ylim[1]])
    
    middle_ticks=[]

    # set norm from ticks and cbar_extend
    if ticks is None:
        norm = None
    else:
        # if ticks are labelled or if there is one more tick than tick labels,
        # do the usual normalisation
        if tick_labels is None or (len(tick_labels) == len(ticks) - 1):
            norm = BoundaryNorm(ticks, cmap.N+1, extend = cbar_extend)
            if tick_labels is not None:
                middle_ticks = [
                    (ticks[i + 1] + ticks[i]) / 2 for i in range(len(ticks) - 1)
                ]
            else:
                middle_ticks = []
        else:
            middle_ticks = [
                (ticks[i + 1] + ticks[i]) / 2 for i in range(len(ticks) - 1)
            ]
            outside_bound_first = [ticks[0] - (ticks[1] - ticks[0]) / 2]
            outside_bound_last = [ticks[-1] + (ticks[-1] - ticks[-2]) / 2]
            bounds = outside_bound_first + middle_ticks + outside_bound_last
            norm = BoundaryNorm(bounds, cmap.N, extend = "neither")

    # define cont
    if data is None:
        cont=None
    else:
        data = data.squeeze()

        if agcd_mask:
            # mask data where observations are sparse
            directory = "/g/data/ia39/aus-ref-clim-data-nci/shapefiles/masks/AGCD-05i"
            agcd = xr.open_dataset(f"{directory}/mask-fraction_agcd_v1-0-2_precip_weight_1960_2022.nc").fraction
            data = data.where(agcd>=0.8)
        

        # plot the hazard data
        if contourf and tick_labels is None:
            if data.max()>=0 and data.min()<=0: 
                print("Using contourf to plot data. Use with caution and check output for data crossing zero")
            cont = ax.contourf(
                data.lon,
                data.lat,
                data,
                cmap=cmap,
                norm=norm,
                levels=ticks,
                extend=cbar_extend,
                zorder=2,
                transform=ccrs.PlateCarree(),
            )
        else:
            cont = ax.pcolormesh(
                data.lon,
                data.lat,
                data,
                cmap=cmap,
                norm=norm,
                zorder=2,
                transform=ccrs.PlateCarree(),
            )
       
        if contour and tick_labels is None:
            ax.contour(
                data.lon,
                data.lat,
                data,
                colors="grey",
                norm=norm,
                levels=ticks,
                extend=cbar_extend,
                linewidths=0.2,
                zorder=3,
                transform=ccrs.PlateCarree(),
            )

    # for station data
    if station_df is not None:
        # assuming columns are named "lon", "lat", variable,
        gdf = gpd.GeoDataFrame(
            station_df, geometry=gpd.points_from_xy(station_df.lon, station_df.lat), crs=ccrs.PlateCarree()
            )
        var = gdf.columns[[2]][0]
        # norm = BoundaryNorm(ticks, cmap.N, extend=cbar_extend)
        cont = ax.scatter(x=station_df.lon,
                          y=station_df.lat,
                          s=markersize, 
                          c=station_df[var],
                          # edgecolors="k", 
                          alpha = 0.8,
                          zorder=7,
                          transform=ccrs.PlateCarree(), 
                          cmap= cmap,
                          norm = norm)
    
    if stippling is not None:
        ax.contourf(stippling.lon,
                    stippling.lat,
                    stippling,
                    alpha=0,
                    hatches = ["","xxxxxx"],
                    zorder=4,
                    transform=ccrs.PlateCarree(),
                   )
        
    if shading is not None:
        ax.contourf(shading.lon,
                    shading.lat,
                    shading,
                    alpha=0,
                    hatches = ["","....."],
                    zorder=4,
                    transform=ccrs.PlateCarree(),
                   )

    # cover area outside australia land area eg mask ocean
    if mask_not_australia:
        # inside the shape, fill white
        ax.add_geometries(
            not_australia,
            crs=ccrs.PlateCarree(),
            facecolor="white",
            linewidth=0,
            zorder=5,
        )

    # cover australia land area eg for ocean data
    if mask_australia:
        # inside the shape, fill white
        ax.add_geometries(
            australia["geometry"],
            crs=ccrs.PlateCarree(),
            facecolor="lightgrey",
            linewidth=0.3,
            edgecolor="k",
            zorder=4,
        )

    if select_area is None:
        # add region borders unless you have selected area
        ax.add_geometries(
            regions["geometry"],
            crs=ccrs.PlateCarree(),
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=area_linewidth,
            zorder=6,
        )

    if coastlines:
        try:
            ax.add_feature(cfeature.BORDERS, zorder=5, linewidth=area_linewidth*0.8,)
        except:
            print("could not download borders")
        ax.coastlines(resolution = "10m", zorder=5, linewidth=area_linewidth*0.8,)


    '''
    if latlongrid:
        
        ax.tick_params(
            bottom=False, top=False, left=False, right=False,
            labelbottom=False, labeltop=False, labelleft=False, labelright=False
        )

        xticks = np.arange(-180, 181, 20)
        yticks = np.arange(-75, 91, 25)

        gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=0.5, color='black', alpha=0.5, linestyle='--', draw_labels=False)

        gl.top_labels = False; gl.bottom_labels = True
        gl.left_labels = False; gl.right_labels = True

        gl.ylocator = mticker.FixedLocator(yticks)
        gl.xlocator = mticker.FixedLocator(xticks)

        # (Optional) Use degree formatters for nicer labels
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        
        gl.xlabel_style = {'fontsize': 8, 'rotation': 0}
        gl.ylabel_style = {'fontsize': 8}
        '''
    

    if latlongrid:

        xticks = [100, 120, 140, 160, 180]
        yticks = [-50, -40, -30, -20, -10, 0, 10]

        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            linewidth=0.4,
            color="black",
            alpha=0.25,
            linestyle="--",
            draw_labels=True,
        )
        gl.rotate_labels = False

        gl.xlocator = mticker.FixedLocator(xticks)
        gl.ylocator = mticker.FixedLocator(yticks)

        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

        gl.left_labels = False
        gl.right_labels = True

        if panel == 3:
            gl.top_labels = False
            gl.bottom_labels = True
        else:
            gl.top_labels = False
            gl.bottom_labels = False

        gl.xlabel_style = {
            "fontsize": 8,
            "rotation": 0,
            "ha": "center",
        }
        gl.ylabel_style = {
            "fontsize": 8,
            "rotation": 0,
            "va": "center",
            "ha": "left",
        }

        ax.tick_params(
            bottom=False, top=False, left=False, right=False,
            labelbottom=False, labeltop=False, labelleft=False, labelright=False
        )
    

    # subtitle
    if subtitle_xy is None:
        subtitle_xy = (0.02, 0.98)
        
    # subtitle above each panel
    if subtitle is not None and subtitle != "":
        ax.set_title(
            subtitle,
            loc="left",
            fontsize=fontsize_subtitle,
            fontweight="normal",
            pad=2)

    return ax, norm, cont, middle_ticks

def plot_cbar(cont=None,
              norm=None,
              ax=None,
              cbar_extend=None, 
              cbar_label=None,
              ticks=None, 
              tick_interval=1,
              tick_labels=None,
              middle_ticks=[], 
              cax_bounds =None,
              contour=False,
              location=None,
              rotation=None,
             ):
    """This function defines and plots the colorbar. 
    It takes crucial information from the plot_data function.
    
    Parameters
    ----------
    cont:
        output from matplotlib plotting
        
    norm:
        normalisation
        
    ax: matplotlib.axes.Axes
        Axes to put the colorbar
        
    cbar_extend: one of {'neither', 'both', 'min', 'max'}.
        eg "both" changes the ends of the colorbar to arrows to indicate that
        values are possible outside the scale shown.
         
    cbar_label: str
        Title for colorbar. Include name of metric and [units]
        
    ticks: list or array
        numerical location of ticks

    tick_interval: int
        Default 1
        For showing every second tick label, set tick_interval=2
        
    tick_labels: list
        If categorical data, these labels go inbetween the numerical bounds set by ticks
        
    middle_ticks: list
        If categorical data, this specifies the location of the tick labels.
        Typically in the middle of the bounds set by ticks
        
    cax_bounds: [left, bottom, width, height]
        Colorbar axes bounds relative to ax
        
    contour: bool
        draw lines on colorbar if True
        Default is False
        
    location: {"top", "bottom", "left", "right"}
        location of the colorbar. Defaults to right.
        
    rotation: [-360,360]
        rotation of tick labels in degrees. Set to 0 for horizontal.

    Returns
    -------
    matplotlib.colorbar    
    
    """

    if cax_bounds is not None:
        cax = ax.inset_axes(cax_bounds)
    else:
        cax=None
    
    cbar = None
    
    if norm is None:
        return cbar
    
    if tick_labels is None:
        cbar = plt.colorbar(
            cont,
            ax=ax,
            extend=cbar_extend,
            cax=cax,
            ticks=ticks,
            norm=norm,
            location=location,
            fraction=0.046, 
            pad=0.1
        )
        # only label ticks at specified tick intervals
        [l.set_visible(False) for (i,l) in enumerate(cbar.ax.xaxis.get_ticklabels()) if i % tick_interval != 0]
        [l.set_visible(False) for (i,l) in enumerate(cbar.ax.yaxis.get_ticklabels()) if i % tick_interval != 0]

    else:
        # for categorical data
        cbar = plt.colorbar(
            cont,
            ax=ax,
            extend='neither',
            cax=cax,
            ticks=ticks,
            norm=norm,
            drawedges=True,
            location=location,
            fraction=0.046, 
            pad=0.1
        )
        if location=="bottom":
            if len(ticks) == len(tick_labels):
                cbar.ax.set_xticks(ticks, tick_labels, wrap=True, verticalalignment="top")
            elif len(middle_ticks) == len(tick_labels):
                cbar.ax.set_xticks(middle_ticks, tick_labels, wrap=True, verticalalignment="top")

        else:
            if len(ticks) == len(tick_labels):
                cbar.ax.set_yticks(ticks, tick_labels, wrap=True)
            elif len(middle_ticks) == len(tick_labels):
                cbar.ax.set_yticks(middle_ticks, tick_labels, wrap=True)

    cbar.ax.tick_params(labelsize=8)
    if contour and tick_labels is None:
        cbar.add_lines(cont)
    
    # Label colorbar
    if cbar is not None:
        if location == "bottom":
            cbar.set_label(cbar_label, fontsize=fontsize_cbar, labelpad=6)
        else:
            cbar.ax.set_title(
                cbar_label,
                zorder=10,
                loc="center",
                fontsize=fontsize_cbar,
                fontstretch="normal",
                verticalalignment="bottom",
                pad=8,
                wrap=True,
            )
    cbar.ax.tick_params(rotation=rotation) 
    return cbar

def plot_select_area(select_area=None,
                     ax=None, 
                     xlim=None,
                     ylim=None,
                     regions=None,
                     land_shadow=False,
                     area_linewidth=0.3,
                    ):
    """This function takes a list of named areas to plot and adjusts 
    the limits of the axis.   
    
    Parameters
    ----------
    select_area: list
        list of selected areas to plot. Must be name of area in regions.

    ax: matplotlib.axes.Axes
        axis 

    xlim:
        longitude limits to use if select_area is None

    ylim:
        latitude limits to use if select_area is None  

    regions: geopandas.GeoDataFrame
        region border data, must contain a column name with "NAME" in it
        to select those areas.

    land_shadow: bool
        whether or not to shade in the non-selected areas. Can help 
        visualise land-sea borders.

    area_linewidth: float
        default 0.3
        linewidth of area edges. Larger values have thicker borders.

    Returns
    -------
    matplotlib.axes.Axes
    """

    if select_area is None:
        ax.set_extent([xlim[0], xlim[1], ylim[0], ylim[1]], crs=ccrs.PlateCarree())
        pass
    else:
        assert isinstance(select_area, list), "select_area must be a list"
        # select state
        name_column = [name for name in regions.columns if "NAME" in name.upper()][0]
        area = regions.loc[regions[name_column].isin(select_area)]
        area= area.to_crs(crs = "GDA2020")
        area[["geometry"]] =shapely.simplify(area[["geometry"]], tolerance)
        map_total_bounds = area.total_bounds
        minx, miny, maxx, maxy = map_total_bounds
        mid_x = (minx + maxx) / 2
        mid_y = (miny + maxy) / 2
        max_range = np.max([(maxy - miny), (maxx - minx)])
        buffer = 0.1 * max_range        
    
        not_area = gpd.GeoSeries(
            data=[
                box(*box(*map_total_bounds).buffer(10 * buffer).bounds).difference(
                    area.dissolve()["geometry"].values[0]
                )
            ],
            crs=ccrs.PlateCarree(),
        )
    
        # mask outside selected area
        if land_shadow:
            # show land as light grey
            ax.add_geometries(not_area,
                              crs=ccrs.PlateCarree(),
                              facecolor="lightgrey",
                              linewidth=area_linewidth,
                              edgecolor="k", 
                              zorder=4)
        else:
            # mask white
            ax.add_geometries(not_area,
                              crs=ccrs.PlateCarree(),
                              facecolor="white",
                              linewidth=area_linewidth, 
                              edgecolor="k", 
                              zorder=4)
    
        ax.set_extent([mid_x - 0.6 * max_range,
                       mid_x + 0.8 * max_range,
                       mid_y - 0.7 * max_range,
                       mid_y + 0.7 * max_range],
                      crs=ccrs.PlateCarree())
    return ax

def plot_titles(title="title",
                date_range = "DD Mon YYYY to DD Mon YYYY", 
                baseline = None, 
                dataset_name= None,
                issued_date=None,
                watermark= None, 
                watermark_color="r",
                ax=None,
                text_xy = None,
                title_ha = "left",
                show_copyright = True,):
    """
    Set the plot title and axis labels
    
    Parameters
    ----------
    title: str
        Main text. Size 14, bold. Location set by text_xy["title"]. 
        Horizontal alignment set by title_ha.
        
    date_range: str
        Text under title. Size 12. Horizontal alignment set by title_ha.
        Intended for data date range. Also can be used for any subtitle.
        Default "DD Mon YYYY to DD Mon YYYY" to indicate prefered date format.
        
    baseline: str
        Text in bottom left corner. Size 8. 
        Intended to describe the baseline period of the data.
        If None, then no text is printed.
        Default is None. 
        
    dataset_name: str
        Text inside bottom right. Size 8.
        Intended to describe data source.
        If None, then no text is printed.
        Default is None.
        
    issued_date: str
        Text on bottom right under the border.
        The date that the plot is current/valid/produced.
        If None (default), then today's date is used.
        To suppress any text, use issued_date="".
        Default is None.
        
    watermark: str
        Large text over plot. Use to indicate draft figures etc.
        Default is None. 
        
    watermark_color: color
        Option to change watermark text colour from red,
        eg if figure colours are red and then you can't read the watermark.
        default is "r" for red text.
        
    ax: matplotlib.axes.Axes
        axes
        
    text_xy: dictionary 
        Expects a dictionary with "title", "date_range" and "watermark" keys
        dictionary items with tuples describing text locations.
        Can omit "watermark" if watermark is None.

    title_ha: {"left", "center", "right"}
        Title horizontal alignment.
        Default "left"

    show_copyright: bool
        Default True

    Returns
    -------
    matplotlib.axes.Axes with text for titles etc.
    
    """

    figsize = plt.gcf().get_size_inches()
    xpt = 1/72/figsize[0]
    ypt = 1/72/figsize[1]
    
    ax.text(
        x=text_xy["title"][0],
        y=text_xy["title"][1],
        s=f"{title}",
        fontsize=fontsize_title,
        weight="normal",
        horizontalalignment=title_ha,
        verticalalignment="bottom",
        transform=ax.transAxes,
        zorder=10,
        wrap=True,
    )

    ax.text(
        x=text_xy["date_range"][0],
        y=text_xy["date_range"][1],
        s=f"{date_range}",
        fontsize=fontsize_subtitle,
        horizontalalignment=title_ha,
        verticalalignment="top",
        transform=ax.transAxes,
        zorder=10,
        wrap=True,
    )
    
    if baseline is not None:
        # print base period inside bottom left corner
        ax.text(
            x=padding*xpt,
            y=(padding+2*fontsize_footnote)*ypt,
            s=f"Base period: {baseline}",
            fontsize=fontsize_footnote,
            verticalalignment="bottom",
            transform=ax.transAxes,
            zorder=10,
        )
    if show_copyright:
        # print copyright outside bottom left corner
        if figsize[0]<5.3:
            string = f"\u00A9 {datetime.datetime.now().year}"
        else:
            string = f"\u00A9 {datetime.datetime.now().year}"
        ax.text(
            x=padding*xpt,
            y=(padding+2*fontsize_footnote)*ypt,
            s=string,
            fontsize=fontsize_footnote,
            transform=ax.transAxes,
            verticalalignment="top",
            zorder=10,
            wrap=True,
        )
    if dataset_name is not None:
        # print data source inside bottom right
        ax.text(
            x=1- padding*xpt,
            y=(padding+2*fontsize_footnote)*ypt,
            s=f"Dataset: {dataset_name}",
            fontsize=fontsize_footnote,
            transform=ax.transAxes,
            horizontalalignment="right",
            verticalalignment="bottom",
            zorder=10,
        )
    # print issued date on bottom right under the border.
    # Set to today's date if None supplied
    # Suppress this by issued_date=""
    if issued_date is None:
        issued_date = datetime.datetime.today().date().strftime("%d %B %Y")
    if len(issued_date)>=1:
        ax.text(
            x=1- padding*xpt,
            y=(padding+2*fontsize_footnote)*ypt,
            s=f"Issued: {issued_date}",
            fontsize=fontsize_footnote,
            transform=ax.transAxes,
            horizontalalignment="right",
            verticalalignment="top",
            zorder=10,
        )
    
    if watermark is not None:
        ax.text(
            x=text_xy["watermark"][0],
            y=text_xy["watermark"][1],
            s=watermark.upper(),
            fontsize=36,
            transform=ax.transAxes,
            horizontalalignment="center",
            verticalalignment="center",
            zorder=10,
            wrap=True,
            alpha=0.5,
            color=watermark_color,
        )
    ax.axis('off')
    return ax


def _iterate_list(_list, i):
    """
    Helper function to iterate over list of objects, eg datasets
    Returns
    -------
    the ith instance in the list
    """
    if isinstance(_list, list):
        list_i = _list[i]
    else:
        list_i = _list
    return list_i


def plot_acs_hazard_multi(
                nrows=None,
                ncols=None,
                regions=None,
                ds_list=None,
                station_dfs=None,                    
                stippling_list=None,
                shading_list=None,
                mask_not_australia=True,
                mask_australia=False,
                agcd_mask=False,
                facecolor="none",
                edgecolor="black",
                figsize=None,
                markersize=None,
                title=None,
                date_range="",
                subplot_titles=None,
                subtitle_xy=None,
                projection=None,
                area_linewidth=0.3,
                coastlines=False,
                xlim=(113, 154),
                ylim=(-43.5, -9.5),
                cmap=cm.Greens,
                cmap_bad="lightgrey",
                cbar_extend="both",
                ticks=None,
                tick_interval=1,
                tick_labels=None,
                cbar_label="",
                cbar_location="bottom",
                share_cbar = True,
                baseline=None,
                dataset_name=None,
                issued_date=None,
                contourf=False,
                contour=False,
                select_area=None,
                land_shadow=False,
                watermark="EXPERIMENTAL\nIMAGE ONLY",
                watermark_color = "r",
                infile=None,
                outfile=None,
                savefig=True,
                tick_rotation=None,
                vcentre=None,
                show_copyright=True,
            ):
    """
    m-by-n panel plot with shared projection and titles etc. 
    Change share_cbar to False and give lists of cmaps/ ticks/cbar_label etc
    for individual colorbars for each subplot. If only one value is given
    then that value is shared for all subplots.
    As with plot_acs_hazard, but takes a list of xarray data arrays.

    Parameters
    ----------     
    regions: geopandas.GeoDataFrame or list
        if None, then will try to read from regions_dict['ncra_regions'].

    ds_list: list of xr.DataArray, optional
        The list of DataArrays to plot.
        Expects a list of 2D xarray DataArray that has already computed the 
        average, sum, anomaly, metric or index you wish to visualise.
        This function is resolution agnostic.

    station_dfs: list of pd.DataFrame, optional
        The list of pandas.DataFrame with columns ["lon", "lat", variable]. 
        If station_df_list is given, then variable values are represented as dots on 
        the map according to the lat lon coordinates and coloured according to
        cmap colours and ticks. Use markersize to change dot size.
        
    stippling_list: list of xr.DataArray, optional
        A list of True/False masks to define regions of stippling hatching 
        for each subplot. ["xxxxxx"]
        Intended to show model agreement, eg for the direction of change.

    shading_list: list of xr.DataArray, optional
        A list of True/False masks to define regions of dotted stippling  
        for each subplot. ["....."] 
        Intended to obscure data outside area of interest.

    mask_not_australia: boolean or list
        decides whether or not the area outside of Australian land is hidden 
        under white shape.
        Default is True.

    mask_australia: boolean or list
        decides whether or not Australian land is hidden under white shape.
        Eg, use when plotting ocean only.
        Default is False.

    agcd_mask: boolean or list
        If True, applies a ~5km mask for data-sparse inland areas of Australia.
        Default is False.

    facecolor: color or list
        color of land when plotting the regions without climate data. 
        facecolor recommendations include "white", "lightgrey", "none".
        Default is "none"

    edgecolor: color
        defines the color of the state/region borders. 
        edgecolor recommendations include "black" and "white".
        Default is "black"

    figsize: tuple
        defines the width and height of the figure in inches.
        ACS recommends a maximum width of 6.7" (17cm) and 
        maximum height of ~7.5" (19cm)

    markersize: optional 
        Markersize for station_df dots.
        default None. If None then the markersize will adjust to the size of the
        figure and the number of markers in the plot such that when there are
        many markers and the figure is small, the markersize is smaller.
        
    title: str
        A title should describe what is shown in the map. 
        The title should be written in plain English and 
        centred at the top of the visualization.
        If title is None, then defaults to the name of the shapefile.
        
    date_range: str
        date_range (or subtitle)
        Expected to decribe the start and end date of the data analysed. 
        This is printed under the title. 
        format: dd Month yyyy to dd Month yyy.
        Default=""
        
    subplot_titles: list of strings
        subplot_titles for labeling each subplot title, Default None

    subtitle_xy: tuple, optional
        (x, y) location of subtitle relative to subplot transAxes.
        defines the top left location for the subtitle. 
        
    projection:
        Specify projection of the maps. The default suits Australia.
        All subplots share the same projection.
        Formally "crs".
        If None, defaults to
        ccrs.LambertConformal(central_latitude=-24.75,
                              central_longitude=134.0,
                              cutoff=30,
                              standard_parallels=(-10, -40),
        unless select_area is not None, then defaults to
        ccrs.PlateCarree()
                            
    area_linewidth: float
        linewidth of state/region borders.
        Default =0.3

    coastlines: boolean or list
        If True, add cartopy coastlines for all coasts (not just Australia). 
        Default is False.
        
    xlim: tuple of floats, or list
        longitude limits
        Default = (113,154)
        
    ylim: tuple of floats, or list
        latitude limits
        Default = (-43.5, -9.5)
        
    cmap: matplotlib colormap or list
        color map for gridded and/or station data
        See cmap_dict for suggested colormaps.
        Default cmap set to cm.Greens.
        Please choose appropriate colormaps for your data.

    cmap_bad: color or list
        define the color to set for "bad" or missing values
        default "lightgrey"
        
    cbar_extend: one of {'neither', 'both', 'min', 'max'} or list
        eg "both" changes the ends of the colorbar to arrows to indicate that
        values are possible outside the scale show.
        If contour or contourf is True, then cbar_extend will be overridden to "none".
        Default is "both"
        
    ticks: list or arraylike (or list of lists and/or arrays)
        Define the ticks on the colorbar. Define any number of intervals. 
        This will make the color for each interval one discrete color, 
        instead of a smooth color gradient.
        If None, linear ticks will be auto-generated to fit the provided data.

    tick_interval: int or list
        Default 1
        For showing every second tick label, set tick_interval=2

    tick_labels: list or list
        Labels for categorical data. 
        If tick_labels is used, then pcolormesh is used to plot data 
        and does not allow contour or contourf to be used.
        Tick labels will correspond to the ticks.
        
    cbar_label: string or list
        defines the title for the color bar. 
        This should indicate the variable name and the units eg 
        "daily rainfall [mm]",
        "annual rainfall [mm]", 
        "monthly rainfall anomaly [mm]",
        "tas [\N{DEGREE SIGN}C]".
        Default is ""

    cbar_location: ["bottom", "right"]
        location of the cbar/s

    share_cbar: boolean
        If True, use one colorbar for all the subplots in the figure.
        If False, use one colorbar per subplot. 
        Make sure to specify the lists of cmaps and ticks etc for each subplot. 
        
    baseline: string
        the baseline period for anomalies, eg "1961 - 1990".
        
    dataset_name: string
        describes the source of the data eg "AGCD v2" or "BARPA-R ACCESS-CM2"
        
    issued_date: string
        The date of issue. If None is supplied, then today's date is printed.
        To supress, set to ""
        
    contourf: bool or list
        if True then the gridded data is visualised as smoothed filled contours. 
        Default is False.
        Use with caution when plotting data with negative and positive values;
        Check output for NaNs and misaligned values.  

    contour: bool ot list
        if True then the gridded data is visualised as smoothed unfilled grey contours.
        Default is True.
        Using both contourf and contour results in smooth filled contours
        with grey outlines between the color levels.

    select_area: list or list of lists
        A list of areas (eg states) that are in the geopandas.GeoDataFrame.
        Inspect the regions gdf for area names. eg ["Victoria", "New South Wales"]

    land_shadow: bool or list
        Used when select_area is not None. 
        This option controls whether to show Australian land area that is outside 
        the select area in grey for visual context.
        Default False.

    watermark: str
        text over the plot for images not in their final form. 
        If the plot is in final form, set to None. 
        Suggestions include "PRELIMINARY DATA", "DRAFT ONLY", 
        "SAMPLE ONLY (NOT A FORECAST)", "EXPERIMENTAL IMAGE ONLY"
        default "EXPERIMENTAL\nIMAGE ONLY"

    watermark_color: default "r"
        for the watermark, this changes the colour of the text.
        The default is red. Only change color if red is not visible. 

    infile: str
        Not yet tested. 
        The idea is to read in 2D netCDF data and use this as the mappable data.

    outfile: str
        The location to save the figure. 
        If None, then figure is saved here f"figures/{title.replace(' ', '-')}.png"

    savefig: bool
        default is True
        If set to False, then fig is not saved.
 
    tick_rotation: int [-360,360]
        Angle to rotate colorbar tick labels.
        Default is None. Tick labels will be horizontal if colorbar is vertical,
        or vertical if colorbar is horizontal.
        Easiest to read if tick_rotation = 0
        
    vcentre: float, eg 0
        default is None.
        Align centre of colormap to this value. 
        Intended for using a divergent colormap with uneven number of ticks 
        around the centre, eg for future temperature anomalies with a larger
        positive range compared to the negative range.

    show_copyright: bool
        Default True
        
    Returns
    -------
    A multi panel plot saved as a png in a "figures" file in your working directory.
    This function returns fig and ax.
    """
 
    if tick_rotation is None:
        tick_rotation = 0

    if ds_list is None:
        ds_list = [None for i in np.arange(ncols*nrows)]

    if station_dfs is None:
        station_dfs = [None for i in np.arange(ncols*nrows)]

    if stippling_list is None:
        stippling_list = [None for i in np.arange(ncols*nrows)]

    if shading_list is None:
        shading_list = [None for i in np.arange(ncols*nrows)]

    if subplot_titles is None:
        subplot_titles = [None for i in np.arange(ncols*nrows)]

    if figsize is None:
        figsize=(6.7, 5)

    # how big is a point (inch/72) as a proportion of the figsize
    xpt = 1/72/figsize[0]
    ypt = 1/72/figsize[1]
    

    if cbar_location == "right":
        # height might be a bit too much for figures with only one line of titles
        #left bottom width height
        if share_cbar:
            plots_rect = (padding*xpt, 
                          (padding+3*fontsize_footnote)*ypt,
                          1-(2*padding+70)*xpt, 
                          1-(2*padding + 3*fontsize_footnote + 2*fontsize_title + 2*fontsize_subtitle)*ypt) 
        else:
            plots_rect = (padding*xpt, 
                          (padding+3*fontsize_footnote)*ypt,
                          1-(2*padding)*xpt, 
                          1-(2*padding + 3*fontsize_footnote + 2*fontsize_title + 2*fontsize_subtitle)*ypt) 
        cbar_rect = [1-(padding+70)*xpt,
                     0.2,
                     30*xpt,
                     0.5] 
        cax_bounds = [0.3,0,0.5,1]
           
    elif cbar_location == "bottom":
        if share_cbar:
            # reserve area in fig for colorbar
            plots_rect = (padding*xpt, 
                          (2*padding+5*fontsize_footnote+20+fontsize_cbar)*ypt, 
                          1-(2*padding)*xpt,
                          1-(3*padding + 5*fontsize_footnote+20+fontsize_cbar + 2*fontsize_title+ 2*fontsize_subtitle)*ypt)
        else:
            # colorbars steal area from subplots
            plots_rect = (padding*xpt, 
                          (2*padding+5*fontsize_footnote)*ypt, 
                          1-(2*padding)*xpt,
                          1-(3*padding + 5*fontsize_footnote + 2*fontsize_title+ 2*fontsize_subtitle)*ypt)
        cbar_rect = [0.05, (padding+5*fontsize_footnote)*ypt, 0.9, 20*ypt]
        cax_bounds = [0, 0.5, 1, 0.5]

    else:
        pass

    # text annotation xy locations for multi-panel plot
    text_xy = {"title": (0.5, 1-(padding+2*fontsize_title)*ypt),
               "date_range": (0.5, 1-(padding+2*fontsize_title+4)*ypt),
               "watermark": (0.45, 0.41),}
    
    if regions is None:
        regions = regions_dict['aus_states_territories']

    # Set default projection for Australia maps and selection maps
    if projection is None:
        if select_area is None:
            # Default for Australian map
            projection = ccrs.LambertConformal(
                central_latitude=-24.75,
                central_longitude=134.0,
                cutoff=30,
                standard_parallels=(-10, -40),
            )
        else:
            projection = ccrs.PlateCarree()

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                            sharey=True, sharex=True, 
                            figsize=figsize,
                            layout="constrained",
                            subplot_kw={'projection': projection, "frame_on":False},)
    
    

    if any(df is not None for df in station_dfs) and markersize is None:
        markersize=(100 - 80*len(station_dfs[0])/5000)*(figsize[0]*figsize[1])/48/4
        
    for i in np.arange(len(ds_list)):
        # if attribute is a list, then iterate over it
        ds = _iterate_list(ds_list, i)
        station_df = _iterate_list(station_dfs, i)
        stippling = _iterate_list(stippling_list, i)
        subtitle = _iterate_list(subplot_titles, i)
        regions_i = _iterate_list(regions, i)
        xlim_i = _iterate_list(xlim, i)
        ylim_i = _iterate_list(ylim, i)
        contourf_i = _iterate_list(contourf, i)
        contour_i = _iterate_list(contour, i)
        facecolor_i = _iterate_list(facecolor, i)
        mask_not_australia_i = _iterate_list(mask_not_australia, i)
        mask_australia_i = _iterate_list(mask_australia, i)
        agcd_mask_i = _iterate_list(agcd_mask, i)
        coastlines_i = _iterate_list(coastlines, i)
        select_area_i = _iterate_list(select_area, i)
        if ncols*nrows==1:
            ax_i=axs
        else:
            ax_i=axs.flatten()[i]
        
        if not share_cbar:
            # if attribute is a list, then iterate over it
            cmap_i = _iterate_list(cmap, i)
            cmap_bad_i = _iterate_list(cmap_bad, i)
            vcentre_i = _iterate_list(vcentre, i)
            cbar_extend_i = _iterate_list(cbar_extend, i)
            cbar_label_i = _iterate_list(cbar_label, i)
            ticks_i = _iterate_list(ticks, i)
            tick_interval_i = _iterate_list(tick_interval, i)
            tick_rotation_i = _iterate_list(tick_rotation, i)
            tick_labels_i = _iterate_list(tick_labels, i)

            cmap_i.set_bad(cmap_bad_i)
        else:
            # do not iterate, use the one cmap for all subplots
            cmap_i = cmap
            cmap_bad_i = cmap_bad
            vcentre_i = vcentre
            cbar_extend_i = cbar_extend
            cbar_label_i = cbar_label
            ticks_i = ticks
            tick_interval_i = tick_interval
            tick_rotation_i = tick_rotation
            tick_labels_i = tick_labels
            contourf_i = contourf
            contour_i = contour
            facecolor_i = facecolor
            mask_not_australia_i = mask_not_australia
            mask_australia_i = mask_australia
            agcd_mask_i = agcd_mask
            coastlines_i = coastlines
            select_area_i = select_area
    
            cmap_i.set_bad(cmap_bad_i)    
        
        ax, _norm, _cont, _middle_ticks = plot_data(regions=regions_i,
                                                  data=ds,
                                                  station_df=station_df,
                                                  markersize=markersize,
                                                  xlim=xlim_i,
                                                  ylim=ylim_i,
                                                  cmap=cmap_i,
                                                  cbar_extend=cbar_extend_i,
                                                  ticks=ticks_i,
                                                  tick_labels=tick_labels_i,
                                                  contourf=contourf_i,
                                                  contour=contour_i,
                                                  ax=ax_i,
                                                  subtitle=subtitle,
                                                  subtitle_xy=subtitle_xy,
                                                  facecolor=facecolor_i,
                                                  mask_not_australia = mask_not_australia_i,
                                                  mask_australia=mask_australia_i,
                                                  agcd_mask=agcd_mask_i,
                                                  area_linewidth=area_linewidth,
                                                  coastlines=coastlines_i,
                                                  stippling=stippling,
                                                  vcentre=vcentre_i,)
        if _norm is not None:
            norm=_norm
            cont=_cont 
            middle_ticks=_middle_ticks

        # if select a specific area -----------
        ax = plot_select_area(select_area=select_area_i, 
                              ax=ax,
                              xlim=xlim_i,
                              ylim=ylim_i,
                              regions=regions_i,
                              land_shadow=land_shadow,
                              area_linewidth=area_linewidth,
                              )
        # ---------------------------------------------
    
                    
        #ax.axis('off')

        if not share_cbar:  
            if _cont is None:
                continue
            cbar_ax = ax
            cbar = plot_cbar(cont=_cont,
                     norm=_norm,
                     ax=cbar_ax,
                     cbar_extend=cbar_extend_i, 
                     cbar_label=cbar_label_i,
                     location=cbar_location,
                     ticks=ticks_i, 
                     tick_interval=tick_interval_i,
                     tick_labels=tick_labels_i,
                     middle_ticks=_middle_ticks,
                     cax_bounds=None,
                     rotation = tick_rotation_i,
                     )   
    
    
    # colorbar -----------------------------------------------------------
    fig.get_layout_engine().set(rect=plots_rect)
    if share_cbar:
        
        cbar_ax = fig.add_axes(cbar_rect) #left bottom width height
        cbar_ax.axis('off')
        
        cbar = plot_cbar(cont=cont,
                         norm=norm,
                         ax=cbar_ax,
                         cbar_extend=cbar_extend, 
                         cbar_label=cbar_label,
                         location=cbar_location,
                         ticks=ticks, 
                         tick_interval=tick_interval,
                         tick_labels=tick_labels,
                         middle_ticks=middle_ticks,
                         cax_bounds=cax_bounds,
                         rotation = tick_rotation,
                         )    
        #------------------------------------------
        
    
    # plot border and annotations -----------------
    ax111 = fig.add_axes([0.,0.,1.,1.], xticks=[], yticks=[]) #(left, bottom, width, height)
    
    
    ax111 = plot_titles(title=title,
                        date_range = date_range, 
                        baseline = baseline, 
                        dataset_name= dataset_name,
                        issued_date=issued_date,
                        watermark=watermark, 
                        watermark_color=watermark_color,
                        ax=ax111,
                        text_xy = text_xy,
                        title_ha = "center",
                        show_copyright=show_copyright,
                   )
    # draw border
    # ax111.axis(True)
    ax111.axis(False)
    # --------------------------------------------
    
    if outfile is None:
        PATH = os.path.abspath(os.getcwd())
        outfile = f"{PATH}/figures/{title.replace(' ', '-')}.png"
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
    
    if savefig:
        plt.savefig(outfile, dpi=300,)
    return fig, axs

# # Define a function for plotting maps
# This is the function you call to plot all the graphs
def plot_acs_hazard(
    name='aus_states_territories',
    regions=None,
    data=None,
    station_df=None,
    markersize=None,
    stippling=None,
    shading=None,
    mask_not_australia=True,
    mask_australia=False,
    agcd_mask=False,
    facecolor="none",
    edgecolor="black",
    figsize=(6,4.5),
    title="",
    date_range="",
    projection=None,
    area_linewidth=0.3,
    coastlines=False,
    xlim=(114,154),
    ylim=(-43.5, -7.5),
    cmap=cm.Greens,
    cmap_bad="lightgrey",
    cbar_extend="both",
    ticks=None,
    tick_interval=1,
    tick_labels=None,
    cbar_label="",
    baseline=None,
    dataset_name=None,
    issued_date=None,
    contourf=False,
    contour=False,
    select_area=None,
    land_shadow=False,
    watermark="EXPERIMENTAL\nIMAGE ONLY",
    watermark_color = "r",
    infile=None,
    outfile=None,
    savefig=True,
    tick_rotation=None,
    vcentre=None,
    show_copyright=True,
):
    """This function takes gridded data or station_df dataframe and shapefiles 
    to visualise hazard data from a 2D Xarray data array
    and plots the data on a map of Australia with the shape outlines.
    Consider using plot_acs_hazard_1pp for plotting consistent with multi-panel plotting, eg centre top titles 

    Parameters
    ----------
    name: str
        name of a shapefile collection in 
        /g/data/ia39/aus-ref-clim-data-nci/shapefiles/data/
        to get regions from.

    regions: geopandas.GeoDataFrame
        if None, then will try to read from regions_dict[{name}].

    data: xr.DataArray
        a 2D xarray DataArray that has already computed the 
        average, sum, anomaly, metric or index you wish to visualise.
        This function is resolution agnostic.

    station_df: pd.DataFrame, optional
        a pandas.DataFrame with columns ["lon", "lat", variable]. 
        If station_df is given, then variable values are represented as dots on 
        the map according to the lat lon coordinates and coloured according to
        cmap colours and ticks.

    markersize: int, optional
        default None. If None then the markersize will adjust to the size of the
        figure and the number of markers in the plot such that when there are
        many markers and the figure is small, the markersize is smaller.

    stippling: xr.DataArray
        a True/False mask to define regions of stippling hatching. 
        Intended to show model agreement, eg for the direction of change.

    mask_not_australia: boolean
        decides whether or not the area outside of Australian land is hidden 
        under white shape.
        Default is True.

    mask_australia: boolean
        decides whether or not Australian land is hidden under white shape.
        Eg, use when plotting ocean only.
        Default is False.

    agcd_mask: boolean
        If True, applies a ~5km mask for data-sparse inland areas of Australia.
        Default is False.

    facecolor: color
        color of land when plotting the regions without climate data. 
        facecolor recommendations include "white", "lightgrey", "none".

    edgecolor: color
        defines the color of the state/region borders. 
        edgecolor recommendations include "black" and "white".

    figsize: tuple
        defines the width and height of the figure in inches.
        Reccommend (8,6) for Australia-wide plots and (6,6) for individual states

    title: str
        defines the text inside the plot.
        If none is given, then will print the name of the shape file.

    date_range: str
        decribes the start and end date of the data analysed. 
        This is printed under the title. 
        format: dd Month yyyy to dd Month yyy.

    projection: optional
        formally spuriously named "crs". Defines the projection of the plots.
        defaults to a LambertConformal projection unless using "select_area".

    area_linewidth: float, optional
        the width of state/region borders only. All other linewidths are hardcoded.

    coastlines: boolean
        If True, add cartopy coastlines for all coasts (not just Australia). 
        Default is False.

    xlim: tuple, optional
        longitude min and max of the plot area.

    ylim: tuple, optional
        latitude min and max of the plot area.

    cmap:
        defines the colormap used for the data.
        See cmap_dict for suggested colormaps.
        If none, cmap set to cm.Greens.
        Please choose appropriate colormaps for your data.

    cmap_bad: color
        define the color to set for "bad" or missing values
        default "lightgrey"

    cbar_extend: one of {'neither', 'both', 'min', 'max'}.
        eg "both" changes the ends of the colorbar to arrows to indicate that
        values are possible outside the scale show.
        If contour or contourf is True, then cbar_extend will be overridden to "none".

    ticks: list or arraylike
        Define the ticks on the colorbar. Define any number of intervals. 
        This will make the color for each interval one discrete color, 
        instead of a smooth color gradient.
        If None, linear ticks will be auto-generated to fit the provided data.

    tick_interval: int
        Default 1
        For showing every second tick label, set tick_interval=2

    tick_labels: list
        Labels for categorical data. 
        If tick_labels is used, then pcolormesh is used to plot data 
        and does not allow contour or contourf to be used.
        Tick labels will correspond to the ticks.

    cbar_label: string
        defines the title for the color bar. 
        This should indicate the variable name and the units eg 
        "daily rainfall [mm]",
        "annual rainfall [mm]", 
        "monthly rainfall anomaly [mm]",
        "tas [\N{DEGREE SIGN}C]".

    baseline: string
        the baseline period for anomalies, eg "1961 - 1990".

    dataset_name: string
        describes the source of the data eg "AGCD v2" or "BARPA-R ACCESS-CM2"

    issued_date: string
        The date of issue. If None is supplied, then today's date is printed.

    contourf: bool
        if True then the gridded data is visualised as smoothed filled contours. 
        Default is False.
        Use with caution when plotting data with negative and positive values;
        Check output for NaNs and misaligned values.  

    contour: bool
        if True then the gridded data is visualised as smoothed unfilled grey contours.
        Default is True.
        Using both contourf and contour results in smooth filled contours
        with grey outlines between the color levels.

    select_area: list
        A list of areas (eg states) that are in the geopandas.GeoDataFrame.
        Inspect the regions gdf for area names. eg ["Victoria", "New South Wales"]

    land_shadow: bool
        Used when select_area is not None. 
        This option controls whether to show Australian land area that is outside 
        the select area in grey for visual context.
        Default False.

    watermark: str
        text over the plot for images not in their final form. 
        If the plot is in final form, set to None. 
        Suggestions include "PRELIMINARY DATA", "DRAFT ONLY", 
        "SAMPLE ONLY (NOT A FORECAST)", "EXPERIMENTAL IMAGE ONLY"

    watermark_color: default "r"
        for the watermark, this changes the colour of the text.
        The default is red. Only change color if red is not visible. 

    infile: str
        Not yet tested. 
        The idea is to read in 2D netCDF data and use this as the mappable data.

    outfile: str
        The location to save the figure. 
        If None, then figure is saved here f"figures/{title.replace(' ', '-')}.png"

    savefig: bool
        default is True
        If set to False, then fig is not saved.

    tick_rotation: int [-360,360]
        Angle to rotate colorbar tick labels.
        Default is None. Tick labels will be horizontal if colorbar is vertical,
        or vertical if colorbar is horizontal.
        Easiest to read id tick_rotation = 0
        
    vcentre: float, eg 0
        default is None.
        Align centre of colormap to this value. 
        Intended for using a divergent colormap with uneven number of ticks 
        around the centre, eg for future temperature anomalies with a larger
        positive range compared to the negative range.

    show_copyright: bool
        Default True

    Returns
    -------
    The map is saved as a png in a "figures" file in your working directory.
    This function returns fig and ax.
    """
    
    if regions is None:
        try:
            regions = regions_dict[name]
        except:
            print(f"Could not read regions_dict[{name}]")
    
    # Set default crs for Australia maps and selection maps
    if projection is None:
        if select_area is None:
            # Default for Australian map
            projection = ccrs.LambertConformal(
                central_latitude=-24.75,
                central_longitude=134.0,
                cutoff=30,
                standard_parallels=(-10, -40),
            )
        else:
            projection = ccrs.PlateCarree()

    # Set up the plot
    fig = plt.figure(
        figsize=figsize,
        zorder=1,
        layout="constrained",
    )
    ax = plt.axes(
        projection=projection,
        frameon=False,
    )
    ax.set_global()

    if infile is not None:
        data = xr.open_dataset(infile)

    if contourf:
        cbar_extend = "neither"

    # plot hazard data ------------------------
    cmap.set_bad(cmap_bad)
    if station_df is not None and markersize is None:
        markersize=(100 - 80*len(station_df)/5000)*(figsize[0]*figsize[1])/48
    ax, norm, cont, middle_ticks =plot_data(regions=regions,
                                            data=data, 
                                            station_df = station_df,
                                            markersize=markersize,
                                            xlim=xlim,
                                            ylim=ylim,
                                            cmap=cmap,
                                            cbar_extend=cbar_extend,
                                            ticks=ticks,
                                            tick_labels=tick_labels,
                                            contourf=contourf,
                                            contour=contour,
                                            ax=ax,
                                            subtitle="",
                                            facecolor=facecolor,
                                            mask_not_australia = mask_not_australia,
                                            mask_australia=mask_australia,
                                            agcd_mask=agcd_mask,
                                            area_linewidth=area_linewidth,
                                            coastlines=coastlines,
                                            stippling=stippling,
                                            shading=shading,
                                            vcentre=vcentre,
                                            )
                    
    # ---------------------------------

    # if select a specific area -----------
    ax = plot_select_area(select_area=select_area, 
                          ax=ax,
                          xlim=xlim,
                          ylim=ylim,
                          regions=regions,
                          land_shadow=land_shadow,
                          area_linewidth=area_linewidth,
                         )
    # ---------------------------------------------

    # colorbar------------------------
  
    if cont is not None and norm is not None:
        cbar = plot_cbar(cont=cont,
                         norm=norm,
                         ax=ax,
                         cbar_extend=cbar_extend, 
                         cbar_label=cbar_label,
                         location = "right",
                         ticks=ticks, 
                         tick_interval=tick_interval,
                         tick_labels=tick_labels,
                         middle_ticks=middle_ticks,
                         cax_bounds = [1.04,0.08,0.04,0.84],
                         rotation = tick_rotation,
                      )
    # ---------------------------------------

    # set the limits of the plotted data
    plots_rect = (0.02,0.02,0.9,0.96) #left bottom width height
    fig.get_layout_engine().set(rect=plots_rect)

    # Annotations and titles ---------------------

    #plot border and annotations
    ax111 = fig.add_axes([0.,0.,1,1], xticks=[], yticks=[]) #(left, bottom, width, height)

    # text annotation xy locations for 1-panel plot
    xpt = 1/72/figsize[0]
    ypt = 1/72/figsize[1]
    
    text_xy_1pp = {"title": (padding*xpt, (padding + 3*fontsize_footnote+fontsize_subtitle+fontsize_title)*ypt), 
                   "date_range": (padding*xpt, (padding + 3*fontsize_footnote + fontsize_subtitle)*ypt),
                   "watermark": (0.4, 0.5),}
    
    ax111 = plot_titles(title=title,
                        date_range = date_range, 
                        baseline = baseline, 
                        dataset_name= dataset_name,
                        issued_date=issued_date,
                        watermark=watermark, 
                        watermark_color=watermark_color,
                        ax=ax111,
                        text_xy = text_xy_1pp,
                        title_ha = "left",
                        show_copyright=show_copyright,
                        )
    # -----------------------------------------------


    if outfile is None:
        PATH = os.path.abspath(os.getcwd())
        outfile = f"{PATH}/figures/{title.replace(' ', '-')}.png"
        os.makedirs(os.path.dirname(outfile), exist_ok=True)

    if savefig:
        plt.savefig(outfile, dpi=300,)
    return fig, ax


    

def plot_acs_hazard_1plus3(
                regions=None,
                ds_gwl12=None,
                station_df_gwl12=None, 
                stippling_gwl12=None,
                shading_gwl12=None,
                gwl12_cmap=cm.Greens,
                gwl12_cbar_extend="both",
                gwl12_cbar_label=None,
                gwl12_ticks=None,
                gwl12_tick_interval=1,
                gwl12_tick_labels=None,
                gwl12_tick_rotation=None,
                gwl12_vcentre=None,
                ds_gwl15=None,
                ds_gwl20=None,
                ds_gwl30=None,                      
                station_df_gwl15=None,
                station_df_gwl20=None,
                station_df_gwl30=None,
                stippling_gwl15=None,
                stippling_gwl20=None,
                stippling_gwl30=None,
                shading_gwl15=None,
                shading_gwl20=None,
                shading_gwl30=None,
                mask_not_australia=True,
                mask_australia=False,
                agcd_mask=False,
                facecolor="none",
                edgecolor="black",
                figsize=None,
                markersize=None,
                title=None,
                date_range="",
                subplot_titles=None,
                subtitle_xy = None,
                projection=None,
                area_linewidth=0.3,
                coastlines=False,
                xlim=(113, 154),
                ylim=(-43.5, -9.5),
                cmap=cm.Greens,
                cmap_bad="lightgrey",
                cbar_extend="both",
                ticks=None,
                tick_interval=1,
                tick_labels=None,
                cbar_label="",
                baseline=None,
                dataset_name=None,
                issued_date=None,
                contourf=False,
                contour=False,
                select_area=None,
                land_shadow=False,
                watermark="EXPERIMENTAL\nIMAGE ONLY",
                watermark_color = "r",
                infile=None,
                outfile=None,
                savefig=True,
                orientation="vertical",
                tick_rotation=None,
                vcentre=None,
                show_copyright=True,
            ):
    """
    Four panel plot with 1 baseline plot and 3 future scenario 
    anomaly plots. The first plot has its own cmap and normalisation.
    The last three  plots share colormap and normalisation.
    
    As with plot_acs_hazard, but takes four xarray data arrays:
    ds_gwl12, ds_gwl15, ds_gwl20, ds_gwl30. (left to right), 
    (top-left, top-right, bottom-left, bottom-right), or 
    (top to bottom).
    This function is intended for plotting multiple Global Warming 
    Levels, but it will plot any valid data (xr.DataArrays or pd.DataFrames)

    This layout is slightly different from plot_acs_hazard_multi. 
    Here in plot_acs_hazard_1plus3, cbars are given their own axes.
    In plot_acs_hazard_multi, cbars steal space from subplot axes for cbars.
    Here, the layout is optimised for figure sizes on A4 pages, 
    other figure sizes may look weird. In contrast, plot_acs_hazard_multi
    adjusts the layout for different figure sizes.

    Parameters
    ----------  
    regions: geopandas.GeoDataFrame
        if None, then will try to read from regions_dict['ncra_regions'].

    ds_gwl12: xr.DataArray
        The first DataArray to plot.
        Expects a 2D xarray DataArray that has already computed the 
        average, sum, anomaly, metric or index you wish to visualise.
        This function is resolution agnostic.

    station_df_gwl12: pd.DataFrame, optional
        The first pandas.DataFrame with columns ["lon", "lat", variable]. 
        If station_df_gwl12 is given, then variable values are represented as dots on 
        the map according to the lat lon coordinates and coloured according to
        cmap colours and ticks. Use markersize to change dot size.
    
    stippling_gwl12: xr.DataArray
        a True/False mask to define regions of stippling hatching 
        for the first subplot. 
        Intended to show model agreement, eg for the direction of change.

    shading_gwl12: xr.DataArray, optional
        A True/False mask to define regions of dotted stippling  
        for the first subplot. ["....."] 
        Intended to obscure data outside area of interest.

    gwl12_cmap: matplotlib.colors.Colormap
        colormap for baseline plot
        default cm.Greens
        
    gwl12_cbar_extend: {"both", "neither", "min", "max"}
        arrows for colorbar for first (gwl12) plot.
        default "both".
        
    gwl12_cbar_label: str
        title for first (gwl12) colorbar.
        
    gwl12_ticks: list or np.array
        ticks for normalising colorbar for gwl12 data. 
        Used for both gridded (ds_gwl12) and station data (gwl12_station_df)

    gwl12_tick_interval: int
        Default 1
        For showing every second tick label for gwl12 data, set tick_interval=2
        
    gwl12_tick_labels: list
        tick labels for gwl12 data if categorical data
        
    gwl12_tick_rotation: [-360,360]
        degrees to rotate tick labels for gwl12 colorbar
        0 is horizontal
         
    gwl12_vcentre: float, optional
        Align centre of colormap to this value. 
        Intended for using a divergent colormap with uneven number of ticks 
        around the centre, eg for future temperature anomalies with a larger
        positive range compared to the negative range.
        
    ds_gwl15: xr.DataArray
        The second DataArray to plot.

    ds_gwl20: xr.DataArray
        The third DataArray to plot.
        
    ds_gwl30: xr.DataArray
        The fourth DataArray to plot.
        
    station_df_gwl15: pd.DataFrame, optional
        The second pandas.DataFrame with columns ["lon", "lat", variable]. 

    station_df_gwl20: pd.DataFrame, optional
        The third pandas.DataFrame with columns ["lon", "lat", variable].
        
    station_df_gwl30: pd.DataFrame, optional
        The fourth pandas.DataFrame with columns ["lon", "lat", variable]. 
        
    stippling_gwl15: xr.DataArray, optional
        a True/False mask to define regions of stippling hatching 
        for the second subplot. 
        
    stippling_gwl20: xr.DataArray, optional
        a True/False mask to define regions of stippling hatching 
        for the third subplot. 
        
    stippling_gwl30: xr.DataArray, optional
        a True/False mask to define regions of stippling hatching 
        for the fourth subplot. 

    shading_gwl15: xr.DataArray, optional
        A True/False mask to define regions of dotted stippling  
        for the second subplot. ["....."] 
        Intended to obscure data outside area of interest.

    shading_gwl20: xr.DataArray, optional
        A True/False mask to define regions of dotted stippling  
        for the third subplot. ["....."] 
        Intended to obscure data outside area of interest.

    shading_gwl30: xr.DataArray, optional
        A True/False mask to define regions of dotted stippling  
        for the fourth subplot. ["....."] 
        Intended to obscure data outside area of interest.
        
    mask_not_australia: boolean
        decides whether or not the area outside of Australian land is hidden 
        under white shape.
        Default is True.

    mask_australia: boolean
        decides whether or not Australian land is hidden under white shape.
        Eg, use when plotting ocean only.
        Default is False.

    agcd_mask: boolean
        If True, applies a ~5km mask for data-sparse inland areas of Australia.
        Default is False.

    facecolor: color
        color of land when plotting the regions without climate data. 
        facecolor recommendations include "white", "lightgrey", "none".
        Default is "none"

    edgecolor: color
        defines the color of the state/region borders. 
        edgecolor recommendations include "black" and "white".
        Default is "black"

    figsize: tuple
        defines the width and height of the figure in inches.
        ACS recommends a maximum width of 6.7" (17cm) and 
        maximum height of ~7.5" (19cm)
        Defaults depend on "orientation"

    markersize: optional 
        Markersize for station_df dots.
        default None. If None then the markersize will adjust to the size of the
        figure and the number of markers in the plot such that when there are
        many markers and the figure is small, the markersize is smaller.
        
    title: str
        A title should describe what is shown in the map. 
        The title should be written in plain English and 
        centred at the top of the visualization.
        If title is None, then defaults to the name of the shapefile.
        
    date_range: str
        date_range (or subtitle)
        Expected to decribe the start and end date of the data analysed. 
        This is printed under the title. 
        format: dd Month yyyy to dd Month yyy.
        Default=""
        
    subplot_titles: list of strings
        subplot_titles for labeling each subplot title
        if None, then subtitles are ["GWL1.5", "GWL2.0", "GWL3.0"]
        otherwise specify a list of three strings.
        
    projection:
        Specify projection of the maps. The default suits Australia.
        Formally "crs".
        If None, defaults to
        ccrs.LambertConformal(central_latitude=-24.75,
                              central_longitude=134.0,
                              cutoff=30,
                              standard_parallels=(-10, -40),
        unless select_area is not None, then defaults to
        ccrs.PlateCarree()
                            
    area_linewidth: float
        linewidth of state/region borders.
        Default =0.3

    coastlines: boolean
        If True, add cartopy coastlines for all coasts (not just Australia). 
        Default is False.
        
    xlim: tuple of floats
        longitude limits
        Default = (113, 154)
        
    ylim: tuple of floats
        latitude limits
        Default = (-43.5, -9.5)
        
    cmap: matplotlib colormap
        color map for gridded and/or station data
        See cmap_dict for suggested colormaps.
        Default cmap set to cm.Greens.
        Please choose appropriate colormaps for your data.

    cmap_bad: color
        define the color to set for "bad" or missing values
        default "lightgrey"
        
    cbar_extend: one of {'neither', 'both', 'min', 'max'}.
        eg "both" changes the ends of the colorbar to arrows to indicate that
        values are possible outside the scale show.
        If contour or contourf is True, then cbar_extend will be overridden to "none".
        Default is "both"
        
    ticks: list or arraylike
        Define the ticks on the colorbar. Define any number of intervals. 
        This will make the color for each interval one discrete color, 
        instead of a smooth color gradient.
        If None, linear ticks will be auto-generated to fit the provided data.

    tick_interval: int
        Default 1
        For showing every second tick label, set tick_interval=2

    tick_labels: list
        Labels for categorical data. 
        If tick_labels is used, then pcolormesh is used to plot data 
        and does not allow contour or contourf to be used.
        Tick labels will correspond to the ticks.
        
    cbar_label: string
        defines the title for the color bar. 
        This should indicate the variable name and the units eg 
        "daily rainfall [mm]",
        "annual rainfall [mm]", 
        "monthly rainfall anomaly [mm]",
        "tas [\N{DEGREE SIGN}C]".
        Default is ""
        
    baseline: string
        the baseline period for anomalies, eg "1961 - 1990".
        
    dataset_name: string
        describes the source of the data eg "AGCD v2" or "BARPA-R ACCESS-CM2"
        
    issued_date: string
        The date of issue. If None is supplied, then today's date is printed.
        To supress, set to ""
        
    contourf: bool
        if True then the gridded data is visualised as smoothed filled contours. 
        Default is False.
        Use with caution when plotting data with negative and positive values;
        Check output for NaNs and misaligned values.  

    contour: bool
        if True then the gridded data is visualised as smoothed unfilled grey contours.
        Default is True.
        Using both contourf and contour results in smooth filled contours
        with grey outlines between the color levels.

    select_area: list
        A list of areas (eg states) that are in the geopandas.GeoDataFrame.
        Inspect the regions gdf for area names. eg ["Victoria", "New South Wales"]

    land_shadow: bool
        Used when select_area is not None. 
        This option controls whether to show Australian land area that is outside 
        the select area in grey for visual context.
        Default False.

    watermark: str
        text over the plot for images not in their final form. 
        If the plot is in final form, set to None. 
        Suggestions include "PRELIMINARY DATA", "DRAFT ONLY", 
        "SAMPLE ONLY (NOT A FORECAST)", "EXPERIMENTAL IMAGE ONLY"
        default "EXPERIMENTAL\nIMAGE ONLY"

    watermark_color: default "r"
        for the watermark, this changes the colour of the text.
        The default is red. Only change color if red is not visible. 

    infile: str
        Not yet tested. 
        The idea is to read in 2D netCDF data and use this as the mappable data.

    outfile: str
        The location to save the figure. 
        If None, then figure is saved here f"figures/{title.replace(' ', '-')}.png"

    savefig: bool
        default is True
        If set to False, then fig is not saved.

    orientation: {"horizontal", "vertical", "square"}
        whether the four plots are orientatied in a vertical stack,
        horizontally, or in a 2-by-2 grid (left to right).
        Default "horizontal"
 
    tick_rotation: int [-360,360]
        Angle to rotate colorbar tick labels.
        Default is None. Tick labels will be horizontal if colorbar is vertical,
        or vertical if colorbar is horizontal.
        Easiest to read if tick_rotation = 0
        
    vcentre: float, eg 0
        default is None.
        Align centre of colormap to this value. 
        Intended for using a divergent colormap with uneven number of ticks 
        around the centre, eg for future temperature anomalies with a larger
        positive range compared to the negative range.

    show_copyright: bool
        Default True

    Returns
    -------
    A four panel plot (one baseline and three anomalies) saved as a png 
    in a "figures" file in your working directory.
    This function returns fig and ax.
    """

    
    if orientation=="horizontal":
        cax_bounds = [1.05,0,0.1,1]
        if tick_rotation is None:
            tick_rotation = 0
        nrows = 1
        ncols = 4
        cbar_location = "right"
        plots_rect = (0.01,0.02,0.98,0.88) #left bottom width height
        # text annotation xy locations
        text_xy = {"title": (0.5, 0.93),
                   "date_range": (0.5, 0.9),
                   "watermark": (0.45, 0.41),}
        subtitle_xy = None
        if figsize is None:
            figsize=(10, 3)
        
    elif orientation == "vertical":
        cax_bounds = [0.08, -0.22, 0.84, 0.07]
        if tick_rotation is None:
            tick_rotation = 0
        nrows = 4
        ncols = 1
        cbar_location = "bottom"
        # use more of the figure width
        plots_rect = (0.06, 0.09, 0.88, 0.84)
        text_xy = {
            "title": (0.5, 0.965),
            "date_range": (0.5, 0.945),
            "watermark": (0.45, 0.41),
        }
        if subtitle_xy is None:
            subtitle_xy = (-0.5, 0.2)
        if figsize is None:
            figsize = (6, 10)
        
    elif orientation=="square":
        cax_bounds = [1.05,0,0.1,1]
        if tick_rotation is None:
            tick_rotation = 0
        nrows = 2
        ncols = 2
        cbar_location = "right"
        plots_rect = (0.01,0.05,0.98,0.85) #left bottom width height
        # text annotation xy locations
        text_xy = {"title": (0.5, 0.96),
                   "date_range": (0.5, 0.95),
                   "watermark": (0.45, 0.41),}
        subtitle_xy = None
        if figsize is None:
            figsize=(6,4.5)
    else:
        print('orientation must be one of ["horizontal", "vertical", "square"]')

    if gwl12_tick_rotation is None:
        gwl12_tick_rotation = tick_rotation
    
    if regions is None:
        regions = regions_dict['aus_states_territories']

    # Set default projection for Australia maps and selection maps
    if projection is None:
        if select_area is None:
            # Default for Australian map
            projection = ccrs.LambertConformal(
                central_latitude=-24.75,
                central_longitude=134.0,
                cutoff=30,
                standard_parallels=(-10, -40),
            )
        else:
            projection = ccrs.PlateCarree()

    fig, axs = plt.subplots(nrows=nrows,
                            ncols=ncols,  
                            sharey=False,
                            sharex=True,
                            figsize=figsize, 
                            #layout="constrained",
                            subplot_kw={'projection': projection, "frame_on":False},)

    gwl12_cmap.set_bad(cmap_bad)
    cmap.set_bad(cmap_bad)

    station_dfs = [station_df_gwl12, station_df_gwl15, station_df_gwl20, station_df_gwl30]
    if any(df is not None for df in station_dfs) and markersize is None:
        markersize=(100 - 80*len(station_dfs[0])/5000)*(figsize[0]*figsize[1])/48/4

    if subplot_titles is None:
        subplot_titles = [f"GWL{[1.2, 1.5, 2.0, 3.0][i]}" for i in range(4)]

    # -------- plot baseline plot and its colorbar ---------------------
    ax, norm, cont, middle_ticks = plot_data(regions=regions,
                                             data=ds_gwl12, 
                                             station_df = station_df_gwl12,
                                             markersize=markersize,
                                             xlim=xlim,
                                             ylim=ylim,
                                             cmap=gwl12_cmap,
                                             cbar_extend=gwl12_cbar_extend,
                                             ticks=gwl12_ticks,
                                             tick_labels=gwl12_tick_labels,
                                             contourf=contourf,
                                             contour=contour,
                                             ax=axs.flatten()[0],
                                             subtitle=subplot_titles[0],
                                             subtitle_xy=subtitle_xy,
                                             facecolor=facecolor,
                                             mask_not_australia = mask_not_australia,
                                             mask_australia=mask_australia,
                                             agcd_mask=agcd_mask,
                                             area_linewidth=area_linewidth,
                                             coastlines=coastlines,
                                             stippling=stippling_gwl12,
                                             shading=shading_gwl12,
                                             vcentre=gwl12_vcentre,
                                             panel = 0
                                            )
    cbar = plot_cbar(cont=cont,
                  norm=norm,
                  ax=axs.flatten()[0],
                  cbar_extend=gwl12_cbar_extend, 
                  cbar_label=gwl12_cbar_label,
                  location=cbar_location,
                  ticks=gwl12_ticks, 
                  tick_interval=gwl12_tick_interval,
                  tick_labels=gwl12_tick_labels,
                  middle_ticks=middle_ticks,
                  cax_bounds=cax_bounds,
                  rotation=gwl12_tick_rotation,
                  )
    # ------- end plot baseline plot and its colorbar ---------------------

    # ------- plot three scenarios as anomalies from baseline--------------
    for i, ds in enumerate([ds_gwl15, ds_gwl20, ds_gwl30]):
        station_df = station_dfs[i+1]
        stippling = [stippling_gwl15, stippling_gwl20, stippling_gwl30][i]
        shading = [shading_gwl15, shading_gwl20, shading_gwl30][i]
        subtitle = subplot_titles[i+1]
        ax, norm, cont, middle_ticks = plot_data(regions=regions,
                                                 data=ds, 
                                                 station_df = station_df,
                                                 markersize=markersize,
                                                 xlim=xlim,
                                                 ylim=ylim,
                                                 cmap=cmap,
                                                 cbar_extend=cbar_extend,
                                                 ticks=ticks,
                                                 tick_labels=tick_labels,
                                                 contourf=contourf,
                                                 contour=contour,
                                                 ax=axs.flatten()[i+1],
                                                 subtitle=subtitle,
                                                 subtitle_xy=subtitle_xy,
                                                 facecolor=facecolor,
                                                 mask_not_australia = mask_not_australia,
                                                 mask_australia=mask_australia,
                                                 agcd_mask=agcd_mask,
                                                 area_linewidth=area_linewidth,
                                                 coastlines=coastlines,
                                                 stippling=stippling,
                                                 shading=shading,
                                                vcentre=vcentre,
                                                panel = i+1)
        
        # if select a specific area -----------
        ax = plot_select_area(select_area=select_area, 
                              ax=ax,
                              xlim=xlim,
                              ylim=ylim,
                              regions=regions,
                              land_shadow=land_shadow,
                              area_linewidth=area_linewidth,
                              )
        # ---------------------------------------------                    
        #ax.axis('off')
    
    # colorbar -----------------------------------------------------------
    cbar = plot_cbar(cont=cont,
                  norm=norm,
                  ax=axs.flatten()[-1],
                  cbar_extend=cbar_extend, 
                  cbar_label=cbar_label,
                  ticks=ticks, 
                  tick_interval=tick_interval,
                  tick_labels=tick_labels,
                  middle_ticks=middle_ticks,
                  cax_bounds =cax_bounds,
                  location=cbar_location,
                    rotation=tick_rotation,
                    )
    
    #------------------------------------------
    
    
    # plot border and annotations -----------------
    #fig.get_layout_engine().set(rect=plots_rect)
    fig.subplots_adjust(left=0.12, right=0.88, top=0.91, bottom=0.11, hspace=0.42)
    
    ax111 = fig.add_axes([0.01,0.01,0.98,0.98], xticks=[], yticks=[]) #(left, bottom, width, height)
        
    ax111 = plot_titles(title=title,
                        date_range = date_range, 
                        baseline = baseline, 
                        dataset_name= dataset_name,
                        issued_date="",
                        watermark="", 
                        watermark_color=watermark_color,
                        ax=ax111,
                        text_xy = text_xy,
                        title_ha = "center",
                        show_copyright=show_copyright,
                   )
    # draw border
    # ax111.axis(True)
    ax111.axis(False)
    # --------------------------------------------
    
    if outfile is None:
        PATH = os.path.abspath(os.getcwd())
        outfile = f"{PATH}/figures/{title.replace(' ', '-')}.png"
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
    
    if savefig:
        plt.savefig(outfile, dpi=300,)
    return fig, axs

# -------------------------------
# -------------------------------AUSCAT specific plots:
from turbulence_AUSCAT.cat_evaluation import combined_significance_table, time_sel_dict, calc_trend_table, select_resample_time

mid_lat_slice = slice(-50,-25)
lon_slice = slice(90,195)
baseline_time_range = np.arange(1990,2009+1)
baseline_time_slice = slice("1990", "2009")

ssp_colors = {"evaluation":"k",
              "historical":"grey",
              "ssp585":(149/255,27/255,30/255),
              "ssp370":(231/255,29/255,37/255),
              "ssp245":(247/255,148/255,32/255),
              "ssp126":(23/255,60/255,102/255),
              "ssp119":(0/255,173/255,207/255),}


list_evaluation = ['evaluation_BARRA-R_r1i1p1f1',
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

def plot_timeseries(ds_resampled,
                    turbulence_index=None,
                    title=None, 
                    ymax=None, 
                    window_size=None, 
                    ax=None,
                    time_selection=None,
                    significance_tested=False,
                   legend=True,
                   pi=100,
                   outfile=None,):
    """ds_resampled should be an xarray dataset that has been sub-sampled and resampled to annual.
    Window_size, therefore, should be the number of years for the rolling average.
    """
    if significance_tested:
        evaluation_combined = pd.read_csv(f"/scratch/v46/ls7238/CAT_turbulence/{turbulence_index}/{P}hPa/evaluation_combined_tests_table_{turbulence_index}-{P}hPa.csv")

        significance_list = list(evaluation_combined[(evaluation_combined["time_selection"]==time_selection) 
                         & (evaluation_combined["combined_significance"].isin(["", "*", "**", "***"]))]["sample2"])
        assert len(significance_list)>0, f"significance_list is empty. Check that time_selection is one of \
        {list(time_sel_dict.keys())}"
        
        print(f"using {significance_list}")
        run_list = ['evaluation_BARRA-R_r1i1p1f1'] + [experiment + run[10:]
                                                      for experiment in ["historical", "ssp126", "ssp370", "ssp585"] 
                                                      for run in significance_list]
        ds_resampled = ds_resampled.sel(run = ds_ts["run"].isin(run_list))  
    else:
        print(f"using {list_historical}")

    
    y=turbulence_index
    if window_size is not None:
        ds_resampled[f"rolling_{window_size}y"] = ds_resampled[turbulence_index].rolling(time = window_size).mean()
        y=f"rolling_{window_size}y"
    
    df = ds_resampled.to_dataframe().reset_index()
    df["experiment"] = [x.split("_")[0] for x in df["run"]]

    if ax is None:
        plt.figure()
        ax=plt.gca()
    else:
        ax = ax
    sns.lineplot(df, x="time", y=y, hue="experiment", 
                 errorbar=('pi', pi), palette=ssp_colors, ax=ax, legend=legend)
    if legend:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    ax.set_title(title)
    plt.ylim((0, ymax))
    ax.set_ylabel(f"Frequency p99 [per 6h]")
    plt.xlim((df["time"][0], df["time"][len(df["time"])-1]))
    ax.grid()
    if outfile is not None:
        plt.savefig(outfile, bbox_inches="tight")
        print(f"Saved {outfile}")
    return ax


def plot_futures(time_selection="annual",
                 turbulence_index=None,
                 P=None,
                 time_slices = [("2015", "2035")], 
                 experiment="ssp370", 
                 all_agree_except=0,
                 save_fig=True,
                 outfile=None,
                 significance_tested=False,
                 zonal_plots = True,
                 ticks_max=0.020):
    
    """
    Plot future change maps and zonal means for the p99 frequency metric.
    Change is calculated from the historical baseline period per model. 
    BARRA-R evaluation is not used in these calculations

    Parameters
    ----------
    time_selection : str
        e.g., 	time_selection: [annual, MJJASO, NDJFMA, DJF, MAM, JJA, SON, January, February, March,
        April, May, June, July, August, September, October, November, December]
    time_slices : list[tuple[str,str]]
        List of (start_year, end_year) pairs.
    experiment : str
        e.g., 'ssp370'
    all_agree_except : int
        Number of runs allowed to disagree with the majority sign.
    save_fig : bool
        If True, saves the map figure.
    significance_tested : bool
        If True, restrict runs to those which pass at least one significance test.

    Returns
    -------
    fig, axs
    """
    if significance_tested:  
        combined_significance_table_file= (f"/scratch/v46/ls7238/CAT_turbulence/{turbulence_index}/{P}hPa/evaluation_combined_tests_table_{turbulence_index}-{P}hPa.csv")
        evaluation_combined = pd.read_csv(combined_significance_table_file)
        # runs that pass at least one test
        run_list = list(evaluation_combined[(evaluation_combined["time_selection"]==time_selection) 
                                 & (evaluation_combined["combined_significance"].isin(["", "*", "**", "***",]))]["sample2"])
    else:
        # use all available
        run_list = list_historical
        
    # get the list of files we need
    glob_list = glob(f"/scratch/v46/ls7238/CAT_turbulence/{turbulence_index}/{P}hPa/freq-above-p99/\
{turbulence_index}-{P}hPa-monthly-freq-above-p99_AUS-15_*_BOM_BARPA-R_v1-r1_6hr.nc")

    # historical
    desired_list = [f"/scratch/v46/ls7238/CAT_turbulence/{turbulence_index}/{P}hPa/freq-above-p99/\
{turbulence_index}-{P}hPa-monthly-freq-above-p99_AUS-15_{run}_BOM_BARPA-R_v1-r1_6hr.nc" 
                                for run in run_list]
    filelist = [f for f in desired_list if f in glob_list]

    def _preprocess(ds):
        ds = ds.sel({"lon":lon_slice,})
        ds = select_resample_time(ds, time_selection)
        return ds
 
    ds_hist = xr.open_mfdataset(filelist,
                          concat_dim="run",
                          combine="nested",
                          preprocess= _preprocess,
                          chunks={"lat":160,"lon":-1},
                          )
    ds_hist["run"] = [run[run.index("_")+1:] for run in ds_hist["run"].values] 
    
    #future
    future_list =  [experiment + run[10:] for run in run_list]
    desired_list = [f"/scratch/v46/ls7238/CAT_turbulence/{turbulence_index}/{P}hPa/freq-above-p99/\
{turbulence_index}-{P}hPa-monthly-freq-above-p99_AUS-15_{run}_BOM_BARPA-R_v1-r1_6hr.nc" 
                                for run in future_list]
    filelist = [f for f in desired_list if f in glob_list]

    ds_future = xr.open_mfdataset(filelist,
                          concat_dim="run",
                          combine="nested",
                          preprocess= _preprocess,
                                  chunks={"lat":160,"lon":-1},
                          )
    ds_future["run"] = [run[run.index("_")+1:] for run in ds_future["run"].values] 

    # combine
    common_runs = [f for f in list(set(ds_hist["run"].values)) if f in set(ds_future["run"].values)]
    print(f" These are the common runs used for {experiment}: {common_runs}")

    # create a continuous time series by combining historical and future experiements
    ds = xr.concat((ds_hist.sel({"run":common_runs,}),
                    ds_future.sel({"run":common_runs,})), 
                   coords="minimal", dim="time", join="outer",
                   compat="override",
                  ).chunk({"run":1, "time":-1,})
    
    # baseline relative to the same model's historical period 
    baseline = ds[turbulence_index].sel({"time":baseline_time_slice}).mean("time")
    
    # stippling for agreement
    agreement_threshold = len(common_runs) - all_agree_except

    # compute per-slice means and agreement masks
    mean_futures = [ds[turbulence_index].sel(time=slice(start, end)).mean("time") 
                    for (start, end) in time_slices]

    # sign agreement: either majority positive or majority negative relative to baseline
    agreement_list = []    
    # median across runs of the change
    ds_list = []
    for mf in mean_futures:
        # calculate the change from baseline period per model (relative to own model's historical baseline)
        mf_delta = (mf - baseline)
        # median value to map
        ds_list.append(mf_delta.median("run"))

        # for stippling: check if all models agree on sign of change (or all minus "all_agree_except")
        # mapped bool for if all models (or nearly all set by "all_agree_except") show positive changes
        # OR
        # mapped bool for if all models (or nearly all set by "all_agree_except") show negative changes
        # IE all models agree on the sign
        agree_mask = ((mf_delta > 0).sum("run") >= agreement_threshold) + ((mf_delta < 0).sum("run") >= agreement_threshold)
        # coarsen for stippling density
        agree_mask = (agree_mask.coarsen(lat=16, lon=16, boundary="pad").mean() > 0.4)
        agreement_list.append(agree_mask)

    fig, axs = plot_acs_hazard_multi(nrows=len(time_slices), 
                          ncols=1, 
                          ds_list=ds_list,
                          stippling_list= agreement_list,
                          mask_not_australia=False,
                          ticks=np.arange(-1*ticks_max, ticks_max*1.0001, 0.002),
                          vcentre=0,
                          tick_interval=2,
                          cbar_extend="both",
                          cbar_label=f"change in frequency\n[per 6h]",
                          cbar_location="right",
                          figsize=(4,6),
                          title=f"Change in {time_selection} frequency of exceeding p99 {turbulence_index} {P}hPa",
                          date_range=f"{experiment}",
                          subplot_titles=[f"{start_year} to {end_year}" for start_year, end_year in time_slices],
                          xlim=(90, 195),
                          ylim=(-53.58 , 13.63),
                          coastlines=True,
                          projection=ccrs.PlateCarree(130),
                          cmap=cmap_dict["anom"],
                          watermark="",
                          show_copyright=False,
                          baseline= "1990 to 2009",
                          );

    if save_fig:
        if outfile is None:
            outfile = f"Change in {time_selection} frequency of exceeding p99 {turbulence_index} {P}hPa {experiment}.png".replace(" ", "_")
            fig.savefig(outfile, bbox_inches="tight")
            print(f"Saved {outfile}")
    display(fig)

    if zonal_plots:
        fig1, axs1 = plt.subplots(len(time_slices), 1, figsize=(3,7), sharex=True)

        hue_order = common_runs
        palette = {run: sns.color_palette("colorblind", 8)[i] for i, run in enumerate(common_runs)}

        for i, (start_year, end_year) in enumerate(time_slices):
            df_zonal = mean_futures[i].mean(["lon"]).to_dataframe()
            sns.lineplot(
                data=df_zonal, x=turbulence_index, y="lat",
                hue="run", hue_order=hue_order, palette=palette,
                orient="y", ax=axs1[i],
            )
            axs1[i].legend_.remove()
            axs1[i].grid(True)

        plt.xlim((0, None))
        plt.tight_layout()
        axs1[i].legend(loc="upper center", bbox_to_anchor=(0.5, -0.25))

        if save_fig:
            path, filetype = outfile.rsplit(".", 1)
            zonal_outfile = f"{path}_zonal-means.{filetype}"
            fig1.savefig(zonal_outfile, bbox_inches="tight")
            print(f"Saved {zonal_outfile}")
        display(fig1)

    # if True:
    #     # zonal mean values
    #     fig2, axs2 = plt.subplots(len(time_slices),1, figsize=(3,7), sharex=True)
    #     df_baseline = baseline.mean(["run", "lon"])\
    #                           .expand_dims({"run":["baseline"]})\
    #                           .to_dataframe()
    #     i=0
    #     for start_year, end_year in time_slices:
    #         df = ds_ann.sel({"time":slice(start_year, end_year)}).mean(["time","lon"]).to_dataframe()
    #         df_zonal = pd.concat([df, df_baseline])
    
    #         hue_order = common_runs + ["baseline"]
    #         palette = {run: "grey" for i, run in enumerate(common_runs)}
    #         palette["baseline"] = "red"
            
    #         sns.lineplot(data = df_zonal, 
    #                      x = turbulence_index, y ="lat", 
    #                      hue = "run", orient="y", ax=axs2[i],
    #                      palette=palette,
    #                     )
    #         axs2[i].legend("")
    #         axs2[i].grid()
    #         i+=1
        
    #     # plt.xticks(np.arange(0,0.05, 0.01))
    #     plt.xlim((0,None))
    #     plt.tight_layout()
    #     axs2[i-1].legend(loc="upper center",bbox_to_anchor=(0.5, -0.25)
    return fig, axs
    
def evaluate_model_map(ds_baseline_mapped, turbulence_index, P, ticks_max=0.08):
    """Plot the baseline mean for each model"""
    experiment = "historical"
    time_selection = "annual"
    outfile = f"/scratch/v46/ls7238/CAT_turbulence/{turbulence_index}/{P}hPa/Evaluation_{turbulence_index}_{P}hPa_frequency_of_exceeding_p99_{experiment}_{time_selection}_1990-2009.png"
        
    fig, axs = plot_acs_hazard_multi(nrows=4, 
                          ncols=2, 
                          ds_list=[ds_baseline_mapped.sel({"run":run}).mean("time")[turbulence_index]
                                   for run in list_evaluation+ list_historical],
                          mask_not_australia=False,
                          ticks=np.arange(0, ticks_max, 0.005),
                          tick_interval=2,
                          cbar_extend="max",
                          cbar_label=f"frequency of exceeding p99 {turbulence_index} [6h frequency]",
                          figsize=(8,11),
                          title=f"Frequency of exceeding p99 {turbulence_index} {P}hPa",
                          subplot_titles=[run for run in list_evaluation+ list_historical],
                          xlim=(90, 195),
                          ylim=(-53.58 , 13.63),
                          coastlines=True,
                          projection=ccrs.PlateCarree(130),
                          cmap=cmap_dict["ipcc_wind_seq"],
                          watermark="",
                          show_copyright=False,
                          outfile=outfile,
                         );
    print(f"Made {outfile}")
    return

# ttest

def evaluate_model_map_anom(ds_baseline_mapped, turbulence_index, P, ticks_max=0.03):
    """Plot the baseline mean anomaly from BARRA-R for each model"""
    experiment = "historical"
    time_selection = "annual"
    outfile = f"/scratch/v46/ls7238/CAT_turbulence/{turbulence_index}/{P}hPa/Evaluation_{turbulence_index}_{P}hPa_Difference_in_frequency_of_exceeding_p99_from_BARRA-R_{experiment}_{time_selection}_1990-2009.png"
    
    pval_list = []
    ds_eval = ds_baseline_mapped.sel({"run":list_evaluation[0], })
    for run in list_evaluation+ list_historical:
        # for one historical run
        ds_hist = ds_baseline_mapped.sel({"run":run, })
        stat, pval = stats.ttest_ind(ds_eval[turbulence_index], ds_hist[turbulence_index])
        pval_list.append(xr.Dataset(data_vars={"pval" : (["lat", "lon"],  pval),}, 
                                    coords= {"lat": ds_hist.lat, "lon": ds_hist.lon},))
    pvals = xr.concat(pval_list,dim="run")
    pvals_stippling=[pvals.sel(run=run)["pval"] <0.01
                                          for run in list_evaluation+ list_historical]
    baseline_barra = ds_baseline_mapped.sel({"run":'evaluation_BARRA-R_r1i1p1f1'}).mean("time")
    plot_acs_hazard_multi(nrows=4, 
                          ncols=2, 
                          ds_list=[(ds_baseline_mapped.sel({"run":run}).mean("time")-baseline_barra)[turbulence_index]
                                   for run in list_evaluation+ list_historical],
                          stippling_list=pvals_stippling,
                          mask_not_australia=False,
                          ticks=np.arange(-1*ticks_max, 1.001*ticks_max, 0.005),
                          vcentre=0,
                          tick_interval=1,
                          cbar_extend="both",
                          title=f"Difference in frequency of exceeding p99 {turbulence_index} {P}hPa from BARRA-R",
                          baseline= "BARRA-R 1990 to 2009",
                          figsize=(8,11),
                          cbar_label=f"Difference of exceeding p99 {turbulence_index} {P}hPa [6h frequency]",
                          subplot_titles= list_evaluation+ list_historical,
                          subtitle_xy=(0.1, 0.85),
                          xlim=(90 , 195),
                          ylim=(-53.58 , 13.63),
                          coastlines=True,
                          projection=ccrs.PlateCarree(130),
                          # cmap=cmap_dict["ipcc_wind_div"],
                          cmap=cmap_dict["anom"],
                          watermark="",
                          show_copyright=False,
                          outfile=outfile,
                         );
    print(f"Made {outfile}")
    return

def plot_timeseries_annual(ds_ts, turbulence_index, P, window_size, ymax=None, outfile = None):
    if outfile is None:
        outfile = f"/scratch/v46/ls7238/CAT_turbulence/{turbulence_index}/{P}hPa/Timeseries_{turbulence_index}_{P}hPa_Frequency_of_exceeding_p99_over_time_annual_rolling{window_size}y.png"
    
    time_selection = "annual"
    ds_resampled = select_resample_time(ds_ts, time_selection)
    
    plot_timeseries(ds_resampled, 
                    turbulence_index=turbulence_index,
                    title=f"Annual {turbulence_index} at {P}hPa",
                    ymax=ymax, 
                    window_size=window_size,
                    time_selection=time_selection, 
                    significance_tested=False, 
                    pi=100,
                    outfile=outfile,
                   )
    return
    
def plot_timeseries_coolwarmseason(ds_ts, turbulence_index, P, window_size, ymax=None, outfile = None):
    # cool season v warm season
    if outfile is None:
        outfile = f"/scratch/v46/ls7238/CAT_turbulence/{turbulence_index}/{P}hPa/Timeseries_{turbulence_index}_{P}hPa_Frequency_of_exceeding_p99_over_time_6Mseason_rolling{window_size}y.png"
    
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False, figsize=(7,6))
    for i, time_selection in enumerate(["MJJASO", "NDJFMA"]):
        ds_resampled = select_resample_time(ds_ts, time_selection)
    
        if i==1:
            legend=True
        else:
            legend=False
        
        ymax_i = None
        if ymax is not None:
            ymax_i = ymax[i]


        plot_timeseries(ds_resampled, 
                        turbulence_index=turbulence_index,
                        title=time_selection,
                        ymax=ymax_i, 
                        window_size=window_size,
                        time_selection=time_selection, 
                        significance_tested=False,
                        ax=axs[i],
                        legend = False)
    
    # Manually set figure and axis lables
    plt.suptitle(f"{turbulence_index} {P}hPa {window_size}-year rolling mean")
    fig.subplots_adjust(hspace=0.25)
    for ax in axs.flat:
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.grid(alpha=0.3)
        ax.tick_params(axis="both", labelsize=10)
        ax.tick_params(axis="y", pad=2)
    axs[0].set_title("Cool season (MJJASO)")
    axs[1].set_title("Warm season (NDJFMA)")
    fig.supylabel("Frequency p99 [per 6h]")
    fig.supxlabel("Year")

    # Legend below figure:
    handles, labels = axs[0].get_legend_handles_labels()
    legend_handles = [
        Line2D([0], [0], color="black", lw=1.5, label="Evaluation"),
        Line2D([0], [0], color="grey", lw=1.5, label="Historical"),
        Line2D([0], [0], color=ssp_colors["ssp126"], lw=1.5, label="SSP126"),
        Line2D([0], [0], color=ssp_colors["ssp370"], lw=1.5, label="SSP370"),
        Line2D([0], [0], color=ssp_colors["ssp585"], lw=1.5, label="SSP585"),]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.045))

    # plt.tight_layout()
    plt.savefig(outfile, bbox_inches="tight")
    print(f"Saved {outfile}")
    return


def plot_timeseries_season(ds_ts, turbulence_index, P, window_size, ymax=None, outfile = None):
    if outfile is None:
        outfile = f"/scratch/v46/ls7238/CAT_turbulence/{turbulence_index}/{P}hPa/Timeseries_{turbulence_index}_{P}hPa_Frequency_of_exceeding_p99_over_time_season_rolling{window_size}y.png"

    # for season:
    ncols=2
    nrows=2
    
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=False, figsize=(9,9))
    fig.subplots_adjust(bottom=0.07)

    for i, time_selection in enumerate(["DJF", "MAM", "JJA", "SON"]):
        print(time_selection)    
        # get the relevant runs and resample to annual
        ds_resampled = select_resample_time(ds_ts, time_selection)
        if i==ncols-1:
            legend=True
        else:
            legend=False

        # Choose y limit for each subplot:
        ymax_i = None
        if ymax is not None:
            ymax_i = ymax[i]
            
        ax = plot_timeseries(ds_resampled, 
                             turbulence_index=turbulence_index,
                                title=time_selection,
                                ymax=ymax_i, 
                                window_size=window_size,
                                time_selection=time_selection, 
                                significance_tested=False,
                               ax=axs[(i//ncols, i%ncols)],
                               legend=False
                               )
        
    # Manually set figure and axis titles:
    plt.suptitle(f"{turbulence_index} {P}hPa {window_size}-year rolling mean")
    for ax in axs.flat:
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.grid(alpha=0.3)
        ax.tick_params(axis="both", labelsize=10)
        ax.tick_params(axis="y", pad=2)
    fig.supylabel("Frequency p99 [per 6h]")
    fig.supxlabel("Year")
    
    # Legend below figure:
    handles, labels = axs[0, 0].get_legend_handles_labels()
    legend_handles = [
        Line2D([0], [0], color="black", lw=1.5, label="Evaluation"),
        Line2D([0], [0], color="grey", lw=1.5, label="Historical"),
        Line2D([0], [0], color=ssp_colors["ssp126"], lw=1.5, label="SSP126"),
        Line2D([0], [0], color=ssp_colors["ssp370"], lw=1.5, label="SSP370"),
        Line2D([0], [0], color=ssp_colors["ssp585"], lw=1.5, label="SSP585"),]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.045))
    
    plt.savefig(outfile, bbox_inches="tight")
    print(f"Saved {outfile}")
    return

    
def plot_timeseries_month(ds_ts, turbulence_index, P, window_size, ymax=None, outfile = None):
    """for month"""
    if outfile is None:
        outfile = f"/scratch/v46/ls7238/CAT_turbulence/{turbulence_index}/{P}hPa/\
            Timeseries_{turbulence_index}_{P}hPa_Frequency_of_exceeding_p99_over_time_month_rolling{window_size}y.png"
    
    ncols = 3
    nrows = 4
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=False, figsize=(12,12.5))
    
    fig.subplots_adjust(left=0.10, right=0.98, top=0.93, bottom=0.05,wspace=0.19, hspace=0.20)

    for i, month_number in enumerate(np.arange(1,12+1)):
        time_selection = calendar.month_name[month_number]
        # get the relevant runs and resample to annual
        ds_resampled = select_resample_time(ds_ts, time_selection)
    
        if i==ncols-1:
            legend=True
        else:
            legend=False
        
        # Choose y limit for each subplot:
        ymax_i = None
        if ymax is not None:
            ymax_i = ymax[i]
    
        plot_timeseries(ds_resampled, 
                        turbulence_index=turbulence_index,
                        title=time_selection,
                        ymax=ymax_i, 
                        window_size=window_size,
                        time_selection=time_selection, 
                        significance_tested=False,
                       ax=axs[(i//ncols, i%ncols)],
                       legend=False)
    
    plt.suptitle(f"{turbulence_index} {P}hPa {window_size}-year rolling mean")
    # remove per-axis labels
    for ax in axs.flat:
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.3)
        ax.tick_params(axis="both", labelsize=9)
        ax.tick_params(axis="y", pad=2)
    fig.supylabel("Frequency p99 [per 6h]")
    fig.supxlabel("Year")

    handles, labels = axs[0, 0].get_legend_handles_labels()
    legend_handles = [
        Line2D([0], [0], color="black", lw=1.5, label="Evaluation"),
        Line2D([0], [0], color="grey", lw=1.5, label="Historical"),
        Line2D([0], [0], color=ssp_colors["ssp126"], lw=1.5, label="SSP126"),
        Line2D([0], [0], color=ssp_colors["ssp370"], lw=1.5, label="SSP370"),
        Line2D([0], [0], color=ssp_colors["ssp585"], lw=1.5, label="SSP585"),]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.045))
    
    plt.savefig(outfile, bbox_inches="tight")
    print(f"Saved {outfile}")
    return


def plot_baseline_ann(ds_eval, turbulence_index, P, ticks_max=None):
    """annual barra climatology plot"""
    experiment = "evaluation"
    time_selection = "annual"
    outfile = f"/scratch/v46/ls7238/CAT_turbulence/{turbulence_index}/{P}hPa/Evaluation_{turbulence_index}_{P}hPa_Frequency_of_exceeding_p99_{experiment}_{time_selection}_1990-2009.png"
    ds_list = [select_resample_time(ds_eval, time_selection).mean(["time",])[turbulence_index]]
    fig, axs = plot_acs_hazard_multi(nrows=1, 
                                     ncols=1, 
                                     ds_list=ds_list,
                                     mask_not_australia=False,
                                     ticks=np.arange(0.00, ticks_max, 0.005),
                                     tick_interval=4,
                                     cbar_extend="max",
                                     cbar_label=f"frequency of exceeding p99 {turbulence_index} [6h frequency]",
                                     figsize=(6, 6),
                                     title=f"Frequency of exceeding p99 {turbulence_index} {P}hPa",
                                     date_range="1990 to 2009",
                                     subplot_titles="annual",
                                     xlim=(90, 195),
                                     ylim=(-53.58 , 13.63),
                                     coastlines=True,
                                     projection=ccrs.PlateCarree(130),
                                     cmap=cmap_dict["ipcc_wind_seq"],
                                     watermark="",
                                     show_copyright=False,
                                     outfile= outfile
                                     );
    print(f"Saved {outfile}")
    return fig, axs

def plot_baseline_coolwarmseason(ds_eval, turbulence_index, P, ticks_max=None):
    """Six-month cool/warm season barra climatology plot"""
    experiment="historical"
    outfile = f"/scratch/v46/ls7238/CAT_turbulence/{turbulence_index}/{P}hPa/Evaluation_{turbulence_index}_{P}hPa_Frequency_of_exceeding_p99_{experiment}_6Mseason_1990-2009.png"
    # cool season v warm season
    cool_season = [5, 6, 7, 8, 9, 10,]
    warm_season = [11, 12, 1, 2, 3, 4]
    
    # cool season v warm season
    
    ds_list = [select_resample_time(ds_eval, time_selection)[turbulence_index].mean("time") 
               for time_selection in ["MJJASO", "NDJFMA"]]
    ds_6mon = xr.concat(ds_list, dim="time")
        
    fig, axs = plot_acs_hazard_multi(nrows=1, 
                                      ncols=2, 
                                      ds_list=ds_list,
                                      mask_not_australia=False,
                                      ticks=np.arange(0.0, ticks_max, 0.005),
                                      tick_interval=2,
                                      cbar_extend="max",
                                      cbar_label=f"frequency of exceeding p99 {turbulence_index} [6h frequency]",
                                      figsize=(6, 4),
                                      title=f"Frequency of exceeding p99 {turbulence_index} {P}hPa",
                                      date_range="1990 to 2009",
                                      subplot_titles=["MJJASO", "NDJFMA"],
                                      xlim=(90, 195),
                                      ylim=(-53.58 , 13.63),
                                      coastlines=True,
                                      projection=ccrs.PlateCarree(130),
                                      cmap=cmap_dict["ipcc_wind_seq"],
                                      watermark="",
                                      show_copyright=False,
                                     outfile=outfile,
                         );
    print(f"Saved {outfile}")
    return fig, axs

def plot_baseline_season(ds_eval, turbulence_index, P, ticks_max=None):
    """Calendar season barra climatology plot"""
    experiment="historical"
    outfile = f"/scratch/v46/ls7238/CAT_turbulence/{turbulence_index}/{P}hPa/Evaluation_{turbulence_index}_{P}hPa_Frequency_of_exceeding_p99_{experiment}_season_1990-2009.png"

    season_list = ["DJF", "MAM", "JJA", "SON"]
    ds_list = [select_resample_time(ds_eval, time_selection)[turbulence_index].mean("time")
                                              for time_selection in season_list]
    
    fig, axs = plot_acs_hazard_multi(nrows=2, 
                                     ncols=2, 
                                     ds_list=ds_list,
                                     mask_not_australia=False,
                                     ticks=np.arange(0.0, ticks_max, 0.005),
                                     tick_interval=2,
                                     cbar_extend="max",
                                     cbar_label=f"frequency of exceeding p99 {turbulence_index} [6h frequency]",
                                     figsize=(8, 6),
                                     title=f"frequency of exceeding p99 {turbulence_index} {P}hPa",
                                     date_range="1990 to 2009",
                                     subplot_titles=season_list,
                                     xlim=(90 , 195),
                                     ylim=(-53.58 , 13.63),
                                     coastlines=True,
                                     projection=ccrs.PlateCarree(130),
                                     cmap=cmap_dict["ipcc_wind_seq"],
                                     watermark="",
                                     show_copyright=False,
                                     outfile=outfile,     
                                     );
    print(f"Saved {outfile}")

    return fig, axs

def plot_baseline_months(ds_eval, turbulence_index, P, ticks_max=None):
    """Month barra climatology plot"""
    experiment="historical"
    outfile = f"/scratch/v46/ls7238/CAT_turbulence/{turbulence_index}/{P}hPa/Evaluation_{turbulence_index}_{P}hPa_Frequency_of_exceeding_p99_{experiment}_month_1990-2009.png"

    ds_list = [select_resample_time(ds_eval, calendar.month_name[time_selection])[turbulence_index].mean("time")
                                              for time_selection in np.arange(1,12+1)]
    fig, axs = plot_acs_hazard_multi(nrows=4, 
                                  ncols=3, 
                                  ds_list=ds_list,
                                  mask_not_australia=False,
                                  ticks=np.arange(0, ticks_max, 0.005),
                                  tick_interval=2,
                                  cbar_extend="max",
                                  cbar_label=f"frequency of exceeding p99 {turbulence_index} [6h frequency]",
                                  figsize=(8, 8),
                                  title=f"frequency of exceeding p99 {turbulence_index} {P}hPa",
                                  date_range="1990 to 2009",
                                  subplot_titles=[calendar.month_name[time_selection] for time_selection in np.arange(1,12+1)],
                                  xlim=(90, 195),
                                  ylim=(-53.58 , 13.63),
                                  coastlines=True,
                                  projection=ccrs.PlateCarree(130),
                                  cmap=cmap_dict["ipcc_wind_seq"],
                                  watermark="",
                                  show_copyright=False,
                                  outfile=outfile,
                     );
    print(f"Saved {outfile}")
    return fig, axs 


def plot_futures_1plus3(time_selection="annual",
                         turbulence_index=None,
                         P=None,
                         time_slices = None, 
                         experiment="ssp370", 
                         all_agree_except=0,
                         figsize=(6,9),
                         save_fig=True,
                         outfile=None,
                         significance_tested=False,
                         ticks_max=0.020,
                        baseline_time_slice = baseline_time_slice):
    
    """
    Plot future change maps and zonal means for the p99 frequency metric.
    Change is calculated from the historical baseline period per model. 
    BARRA-R evaluation is not used in these calculations

    Parameters
    ----------
    time_selection : str
        e.g., 	time_selection: [annual, MJJASO, NDJFMA, DJF, MAM, JJA, SON, January, February, March,
        April, May, June, July, August, September, October, November, December]
    turbulence_index: str
        turbulence index, one of "windspeed"m "VWS", "TI1"
    P :
        pressure level for index. eg 200 or 250
    time_slices : list[tuple[str,str]]
        List of (start_year, end_year) pairs.
    experiment : str
        e.g., 'ssp370'
    all_agree_except : int
        Number of runs allowed to disagree with the majority sign for hatching. Indicates robust signal across models
    save_fig : bool
        If True, saves the map figure.
    significance_tested : bool
        If True, restrict runs to those which pass at least one significance test. Determined by eg "**" 
    ticks_max: float
        limit for colorscale for anomalies

    Returns
    -------
    fig, axs
    """
    if significance_tested:  
        combined_significance_table_file= (f"/scratch/v46/gt3409/{turbulence_index}/{P}hPa/evaluation_combined_tests_table_{turbulence_index}-{P}hPa.csv")
        evaluation_combined = pd.read_csv(combined_significance_table_file)
        # runs that pass at least one test
        run_list = list(evaluation_combined[(evaluation_combined["time_selection"]==time_selection) 
                                 & (evaluation_combined["combined_significance"].isin(["", "*", "**", "***",]))]["sample2"])
    else:
        # use all available
        run_list = list_historical
        
    # get the list of files we need
    glob_list = glob(f"/scratch/v46/gt3409/{turbulence_index}/{P}hPa/freq-above-p99/{turbulence_index}-{P}hPa-monthly-freq-above-p99_AUS-15_*_BOM_BARPA-R_v1-r1_6hr.nc")

    # historical
    desired_list = [f"/scratch/v46/gt3409/{turbulence_index}/{P}hPa/freq-above-p99/{turbulence_index}-{P}hPa-monthly-freq-above-p99_AUS-15_{run}_BOM_BARPA-R_v1-r1_6hr.nc" for run in run_list]
    filelist = [f for f in desired_list if f in glob_list]
    
    def _preprocess(ds):
        ds = ds.sel({"lon":lon_slice,})
        ds = select_resample_time(ds, time_selection)
        return ds
 
    ds_hist = xr.open_mfdataset(filelist,
                          concat_dim="run",
                          combine="nested",
                          preprocess= _preprocess,
                          chunks={"lat":160,"lon":-1},
                          )
    ds_hist["run"] = [run[run.index("_")+1:] for run in ds_hist["run"].values] 
    
    #future
    future_list =  [experiment + run[10:] for run in run_list]
    desired_list = [f"/scratch/v46/gt3409/{turbulence_index}/{P}hPa/freq-above-p99/\
{turbulence_index}-{P}hPa-monthly-freq-above-p99_AUS-15_{run}_BOM_BARPA-R_v1-r1_6hr.nc" 
                                for run in future_list]
    filelist = [f for f in desired_list if f in glob_list]

    ds_future = xr.open_mfdataset(filelist,
                          concat_dim="run",
                          combine="nested",
                          preprocess= _preprocess,
                                  chunks={"lat":160,"lon":-1},
                          )
    ds_future["run"] = [run[run.index("_")+1:] for run in ds_future["run"].values] 

    # combine
    common_runs = [f for f in list(set(ds_hist["run"].values)) if f in set(ds_future["run"].values)]
    if significance_tested:
        print(f" These are the common runs used for {experiment}: {common_runs}")

    # create a continuous time series by combining historical and future experiments
    ds = xr.concat((ds_hist.sel({"run":common_runs,}),
                    ds_future.sel({"run":common_runs,})), 
                   coords="minimal", dim="time", join="outer",
                   compat="override",
                  ).chunk({"run":1, "time":-1,})

    # baseline relative to the same model's historical period 
    baseline = ds[turbulence_index].sel({"time":baseline_time_slice}).mean("time")
    
    # stippling for agreement
    agreement_threshold = len(common_runs) - all_agree_except

    # compute per-slice means and agreement masks
    mean_futures = [ds[turbulence_index].sel(time=slice(start, end)).mean("time") 
                    for (start, end) in time_slices[1:]]

    # sign agreement: either majority positive or majority negative relative to baseline
    agreement_list = []    
    # median across runs of the change
    ds_list = []
    for mf in mean_futures:
        # calculate the change from baseline period per model (relative to own model's historical baseline)
        mf_delta = (mf - baseline)
        # median value to map
        ds_list.append(mf_delta.mean("run"))

        # for stippling: check if all models agree on sign of change (or all minus "all_agree_except")
        # mapped bool for if all models (or nearly all set by "all_agree_except") show positive changes
        # OR
        # mapped bool for if all models (or nearly all set by "all_agree_except") show negative changes
        # IE all models agree on the sign
        agree_mask = ((mf_delta > 0).sum("run") >= agreement_threshold) + ((mf_delta < 0).sum("run") >= agreement_threshold)
        # coarsen for stippling density
        agree_mask = (agree_mask.coarsen(lat=16, lon=16, boundary="pad").mean() > 0.4)
        agreement_list.append(agree_mask)

    ds_1, ds_2, ds_3 = ds_list
    ds_baseline = baseline.mean("run")
    stippling_1, stippling_2, stippling_3 = agreement_list

    letters = ["(a)", "(b)", "(c)", "(d)"]
    subplot_titles = [f"{letters[i]} {start}-{end}" for i, (start, end) in enumerate(time_slices)]

    plot_acs_hazard_1plus3(
                ds_gwl12=ds_baseline,
                gwl12_cmap=cmap_dict["ipcc_wind_seq"],
                gwl12_cbar_extend="both",
                gwl12_cbar_label="frequency [per 6h]",
                gwl12_ticks=np.arange(0, 0.071, 0.005),
                gwl12_tick_interval=4,
                gwl12_tick_rotation =0,
                ds_gwl15=ds_1,
                ds_gwl20=ds_2,
                ds_gwl30=ds_3,                      
                stippling_gwl15=stippling_1,
                stippling_gwl20=stippling_2,
                stippling_gwl30=stippling_3,
                mask_australia=False,
                mask_not_australia=False,
                ticks=np.arange(-1*ticks_max, ticks_max*1.0001, 0.002),
                tick_rotation =0,
                vcentre=0,
                tick_interval=4,
                cbar_label=f"change in frequency [per 6h]",
                figsize=figsize,
                title=f"{experiment.upper()}: Change in {time_selection} frequency of exceeding p99\n {turbulence_index} - {P}hPa",
                date_range="",
                subplot_titles=subplot_titles,
                subtitle_xy = None,
                area_linewidth=0.3,
                xlim=(90, 195),
                ylim=(-53.58 , 13.63),
                coastlines=True,
                projection=ccrs.PlateCarree(130),
                cmap=cmap_dict["anom"],
                watermark="",
                show_copyright=False,
                baseline= "1990 to 2009",
                cbar_extend="both",
                dataset_name="BARPA-R",
                issued_date=None,
                outfile=None,
                orientation="vertical",
                )
    if save_fig:
        if outfile is None:
            outfile = f"Change in {time_selection} frequency of exceeding p99 {turbulence_index} {P}hPa {experiment}.png".replace(" ", "_")
        plt.savefig(outfile)
        print(f"Saved {outfile}")
    return





from tqdm.notebook import tqdm
import gc


# =====================================================================
# ---------------------------------------------------------------------
# !!!!!!!!!!!! LETTY EDITS AND FUNCTIONS BELOW THIS POINT !!!!!!!!!!!!!
# ---------------------------------------------------------------------
# =====================================================================

def prep_futures_1plus3(time_selection="annual",
                        turbulence_index=None,
                        P=None,
                        time_slices=None,
                        experiment="ssp370",
                        all_agree_except=1,
                        significance_tested=False,
                        baseline_time_slice=baseline_time_slice):

    if significance_tested:
        combined_significance_table_file = (
            f"/scratch/v46/ls7238/CAT_turbulence/{turbulence_index}/{P}hPa/"
            f"evaluation_combined_tests_table_{turbulence_index}-{P}hPa.csv"
        )
        evaluation_combined = pd.read_csv(combined_significance_table_file)
        run_list = list(
            evaluation_combined[
                (evaluation_combined["time_selection"] == time_selection)
                & (evaluation_combined["combined_significance"].isin(["", "*", "**", "***"]))
            ]["sample2"]
        )
    else:
        run_list = list_historical

    glob_list = set(glob(
        f"/scratch/v46/ls7238/CAT_turbulence/{turbulence_index}/{P}hPa/freq-above-p99/"
        f"{turbulence_index}-{P}hPa-monthly-freq-above-p99_AUS-15_*_BOM_BARPA-R_v1-r1_6hr.nc"
    ))

    def _preprocess(ds):
        ds = ds.sel({"lon": lon_slice})
        ds = select_resample_time(ds, time_selection)
        return ds

    def get_file(run):
        return (
            f"/scratch/v46/ls7238/CAT_turbulence/{turbulence_index}/{P}hPa/freq-above-p99/"
            f"{turbulence_index}-{P}hPa-monthly-freq-above-p99_AUS-15_{run}_BOM_BARPA-R_v1-r1_6hr.nc"
        )

    # find common runs that have both hist and future files
    common_runs = []
    for run in run_list:
        hist_run   = run
        future_run = experiment + run[10:]
        if get_file(hist_run) in glob_list and get_file(future_run) in glob_list:
            common_runs.append(run)

    print(f"  {len(common_runs)} common runs found")

    # accumulators
    baseline_accum = None
    delta_accum    = None
    agree_pos      = None
    agree_neg      = None
    coords         = None
    n_runs = 0

    for run in tqdm(common_runs, desc=f"{turbulence_index} {P}hPa {experiment} {time_selection}"):
        hist_file   = get_file(run)
        future_run  = experiment + run[10:]
        future_file = get_file(future_run)

        with xr.open_dataset(hist_file) as ds_h, xr.open_dataset(future_file) as ds_f:
            ds_h = _preprocess(ds_h)
            ds_f = _preprocess(ds_f)
            ds   = xr.concat([ds_h, ds_f], dim="time")

            baseline = ds[turbulence_index].sel(time=baseline_time_slice).mean("time").load()

            run_deltas = []
            for (start, end) in time_slices[1:]:
                delta = (ds[turbulence_index].sel(time=slice(start, end)).mean("time") - baseline).load()
                run_deltas.append(delta)

            del ds, ds_h, ds_f
            gc.collect()

        # extract coords once for reconstructing DataArrays at the end
        if coords is None:
            coords = {"lat": baseline.lat.values, "lon": baseline.lon.values}

        # accumulate as plain numpy — no xarray/dask graph buildup
        if n_runs == 0:
            baseline_accum = baseline.values.copy()
            delta_accum    = [d.values.copy()            for d in run_deltas]
            agree_pos      = [(d.values > 0).astype(int) for d in run_deltas]
            agree_neg      = [(d.values < 0).astype(int) for d in run_deltas]
        else:
            baseline_accum += baseline.values
            for i, d in enumerate(run_deltas):
                delta_accum[i] += d.values
                agree_pos[i]   += (d.values > 0).astype(int)
                agree_neg[i]   += (d.values < 0).astype(int)

        del baseline, run_deltas
        gc.collect()
        n_runs += 1

    if n_runs == 0:
        raise RuntimeError("No valid runs found — check file paths and run lists.")

    agreement_threshold = n_runs - all_agree_except

    # wrap numpy arrays back into xarray for coarsen
    ds_baseline = xr.DataArray(baseline_accum / n_runs, coords=coords, dims=["lat", "lon"])
    ds_list     = [xr.DataArray(acc / n_runs, coords=coords, dims=["lat", "lon"]) for acc in delta_accum]

    agreement_list = []
    for pos, neg in zip(agree_pos, agree_neg):
        agree_mask = xr.DataArray(
            ((pos >= agreement_threshold) | (neg >= agreement_threshold)),
            coords=coords,
            dims=["lat", "lon"]
        )
        agree_mask = (agree_mask.coarsen(lat=16, lon=16, boundary="pad").mean() > 0.4)
        agreement_list.append(agree_mask.load())

    return {
        "ds_baseline":      ds_baseline,
        "ds_list":          ds_list,
        "agreement_list":   agreement_list,
        "time_slices":      time_slices,
        "time_selection":   time_selection,
        "turbulence_index": turbulence_index,
        "P":                P,
        "experiment":       experiment,
        "n_runs":           n_runs,
    }




def draw_futures_1plus3(prepped, figsize=(6, 9), ticks_max=0.02, outfile=None, save_fig=False):
    '''Sister function to prep_futures_1plus3 --> takes information computed in this function as argument "prepped" (exepcts dict)
    and just plots it, no computation done in this function, simply putting pre-computed info on maps.'''
    
    ds_baseline = prepped["ds_baseline"]
    ds_1, ds_2, ds_3 = prepped["ds_list"]
    stippling_1, stippling_2, stippling_3 = prepped["agreement_list"]
    time_slices = prepped["time_slices"]
    time_selection = prepped["time_selection"]
    turbulence_index = prepped["turbulence_index"]
    P = prepped["P"]
    experiment = prepped["experiment"]

    letters = ["(a)", "(b)", "(c)", "(d)"]
    subplot_titles = [f"{letters[i]} {start}-{end}" for i, (start, end) in enumerate(time_slices)]

    fig, axs = plot_acs_hazard_1plus3(
        regions=regions_dict["aus_states_territories"],
        ds_gwl12=ds_baseline,
        gwl12_cmap=cmap_dict["ipcc_wind_seq"],
        gwl12_cbar_extend="both",
        gwl12_cbar_label="Frequency [per 6h]",
        gwl12_ticks=np.arange(0, 0.071, 0.005),
        gwl12_tick_interval=4,
        gwl12_tick_rotation=0,
        ds_gwl15=ds_1,
        ds_gwl20=ds_2,
        ds_gwl30=ds_3,
        stippling_gwl15=stippling_1,
        stippling_gwl20=stippling_2,
        stippling_gwl30=stippling_3,
        mask_australia=False,
        mask_not_australia=False,
        ticks=np.arange(-1*ticks_max, ticks_max*1.0001, 0.002),
        tick_rotation=0,
        vcentre=0,
        tick_interval=4,
        cbar_label="Change in frequency [per 6h]",
        figsize=figsize,
        title=f"{experiment.upper()}: Change in {time_selection} frequency \nof exceeding p99 {turbulence_index} - {P}hPa",
        date_range="",
        subplot_titles=subplot_titles,
        area_linewidth=0.3,
        xlim=(90, 195),
        ylim=(-53.58, 13.63),
        coastlines=True,
        projection=ccrs.PlateCarree(130),
        cmap=cmap_dict["anom"],
        watermark="",
        show_copyright=False,
        baseline="1990 to 2009",
        cbar_extend="both",
        dataset_name="BARPA-R",
        orientation="vertical",
        savefig=False,
    )


    if save_fig and outfile is not None:
        fig.savefig(outfile, bbox_inches="tight", dpi=300)

    return fig, axs


def draw_futures_1plus3_2x2(
    prepped,
    figsize=(9,8),
    ticks_max=0.02,
    outfile=None,
    save_fig=False,
):
    """
    2x2 grid version for precomputed future maps.
    Top-left: baseline (with its own colorbar underneath).
    Top-right / Bottom-left / Bottom-right: anomaly panels (shared colorbar underneath bottom row).
    """

    ds_baseline = prepped["ds_baseline"]
    ds_1, ds_2, ds_3 = prepped["ds_list"]
    stippling_1, stippling_2, stippling_3 = prepped["agreement_list"]
    time_slices = prepped["time_slices"]
    time_selection = prepped["time_selection"]
    turbulence_index = prepped["turbulence_index"]
    P = prepped["P"]
    experiment = prepped["experiment"]

    map_list = [ds_baseline, ds_1, ds_2, ds_3]
    stippling_list = [None, stippling_1, stippling_2, stippling_3]

    letters = ["(a)", "(b)", "(c)", "(d)"]
    panel_titles = [
        f"{letters[i]} {start}-{end}"
        for i, (start, end) in enumerate(time_slices)
    ]

    proj = ccrs.PlateCarree(130)

    fig, axs_2d = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=figsize,
        subplot_kw={"projection": proj, "frame_on": True},  # frame_on=True for borders
    )

    axs = axs_2d.flatten()

    # Simplified title
    fig.suptitle(
        f"{experiment.upper()} · {time_selection} · {turbulence_index} p99 · {P} hPa",
        fontsize=fontsize_title,
        fontweight="normal",
        y=0.995,
    )

    # Colormap / norms
    base_ticks = np.arange(0, 0.071, 0.005)
    anom_ticks = np.arange(-ticks_max, ticks_max * 1.0001, 0.002)

    base_cmap = cmap_dict["ipcc_wind_seq"].copy()
    anom_cmap = crop_cmap_center(
        cmap_dict["anom"].copy(), anom_ticks, 0, extend="both"
    )
    base_cmap.set_bad("lightgrey")
    anom_cmap.set_bad("lightgrey")

    base_norm = BoundaryNorm(base_ticks, base_cmap.N + 1, extend="both")
    anom_norm = BoundaryNorm(anom_ticks, anom_cmap.N + 1, extend="both")

    xticks = [100, 120, 140, 160, 180]
    yticks = [-50, -40, -30, -20, -10, 0, 10]

    last_base = None
    last_anom = None

    for i, ax in enumerate(axs):
        da = map_list[i]
        stip = stippling_list[i]

        row, col = divmod(i, 2)

        ax.set_extent([90, 195, -53.58, 13.63], crs=ccrs.PlateCarree())

        if i == 0:
            last_base = ax.pcolormesh(
                da.lon, da.lat, da,
                cmap=base_cmap, norm=base_norm,
                transform=ccrs.PlateCarree(), zorder=2,
            )
        else:
            last_anom = ax.pcolormesh(
                da.lon, da.lat, da,
                cmap=anom_cmap, norm=anom_norm,
                transform=ccrs.PlateCarree(), zorder=2,
            )

        if stip is not None:
            ax.contourf(
                stip.lon, stip.lat, stip,
                alpha=0, hatches=["", "xxxxxx"],
                transform=ccrs.PlateCarree(), zorder=4,
            )

        ax.add_geometries(
            regions_dict["aus_states_territories"]["geometry"],
            crs=ccrs.PlateCarree(),
            facecolor="none", edgecolor="black", linewidth=0.3, zorder=6,
        )
        ax.coastlines(resolution="10m", linewidth=0.25, zorder=5)
        try:
            ax.add_feature(cfeature.BORDERS, linewidth=0.2, zorder=5)
        except Exception:
            pass

        # Add a visible border/spine around each panel
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("#cccccc")
            spine.set_linewidth(0.5)

        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            linewidth=0.4, color="black", alpha=0.20,
            linestyle="--", draw_labels=True,
            xpadding = 6, ypadding =6
        )
        gl.rotate_labels = False
        gl.xlocator = mticker.FixedLocator(xticks)
        gl.ylocator = mticker.FixedLocator(yticks)
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

        gl.left_labels   = (col == 0)
        gl.right_labels  = False
        gl.top_labels    = False
        gl.bottom_labels = (row == 1)

        gl.xlabel_style = {"fontsize": 7, "rotation": 0, "ha": "center"}
        gl.ylabel_style = {"fontsize": 7, "rotation": 0, "ha": "right", "va": "center"}

        ax.set_title(
            panel_titles[i],
            loc="left", fontsize=fontsize_subtitle,
            fontweight="normal", pad=2,
        )

    # Finalise layout
    fig.subplots_adjust(
        left=0.08, right=0.97,
        top=0.93, bottom=0.22,
        hspace=0.35, wspace=0.18,
    )
    fig.canvas.draw()

    pos = [ax.get_position() for ax in axs]
    # pos[0]=top-left (a), pos[1]=top-right (b)
    # pos[2]=bottom-left (c), pos[3]=bottom-right (d)

    cbar_h   = 0.022
    cbar_gap = 0.032   # gap between panel bottom and colorbar top

    # ── Colorbar 1: baseline — under panel (a) only ───────────────────────────
    cax1 = fig.add_axes([
        pos[0].x0,
        pos[0].y0 - 0.02 - cbar_h,
        pos[0].width,
        cbar_h,
    ])
    cbar1 = fig.colorbar(
        last_base, cax=cax1,
        orientation="horizontal", extend="both",
        ticks=base_ticks,
    )
    cbar1.ax.tick_params(labelsize=7, pad=2)
    for j, lbl in enumerate(cbar1.ax.xaxis.get_ticklabels()):
        lbl.set_visible(j % 2 == 0)
    cbar1.set_label("Frequency [per 6 h]", fontsize=fontsize_cbar, labelpad=4)
    cbar1.ax.xaxis.set_label_position("bottom")   # label below the bar

    # ── Colorbar 2: anomaly — spanning panels (b), (c), (d) ─────────────────
    # Span from left edge of (c) to right edge of (d) — i.e. the full bottom row
    anom_x0 = pos[2].x0
    anom_x1 = pos[3].x1
    cax2 = fig.add_axes([
        anom_x0,
        pos[2].y0 - cbar_gap - cbar_h,
        anom_x1 - anom_x0,
        cbar_h,
    ])
    cbar2 = fig.colorbar(
        last_anom, cax=cax2,
        orientation="horizontal", extend="both",
        ticks=anom_ticks,
    )
    cbar2.ax.tick_params(labelsize=7, pad=2)
    for j, lbl in enumerate(cbar2.ax.xaxis.get_ticklabels()):
        lbl.set_visible(j % 2 == 0)
    cbar2.set_label(
        "Change in frequency [per 6 h]",
        fontsize=fontsize_cbar, labelpad=4,
    )
    cbar2.ax.xaxis.set_label_position("bottom")   # label below the bar

    if save_fig and outfile is not None:
        fig.savefig(outfile, dpi=300, bbox_inches="tight")
        print(f"Saved to: {outfile}")

    return fig, axs



# PLOTTING EVALUATION VS HISTORICAL

def plot_eval_vs_historical(
    turbulence_index,
    P,
    time_selection,          # e.g. "annual", "MJJASO", "NDJFMA", "DJF", "MAM", "JJA", "SON", "January", etc.
    ds=None,
    filelist=None,
    figsize=(10, 4),
    ticks_max=None,
    outfile=None,
    save_fig=False,
):
    """
    Two-panel spatial map: BARRA-R (left) vs historical MMM (right).

    Parameters
    ----------
    turbulence_index : str
        e.g. "TI1"
    P : int
        Pressure level in hPa, e.g. 250
    time_selection : str
        Any period string accepted by select_resample_time:
        "annual", "MJJASO", "NDJFMA", "DJF", "MAM", "JJA", "SON",
        or a month name e.g. "January"
    ds : xarray.Dataset, optional
        Pre-loaded dataset. If None, loads from filelist/default path.
    filelist : list of str, optional
        Glob file list — only used if ds is None.
    """

    # ── Resample to the requested time period (mirrors timeseries functions) ──
    ds_resampled = select_resample_time(ds, time_selection)

    da_eval = (
        ds_resampled[turbulence_index]
        .sel(run="evaluation_BARRA-R_r1i1p1f1")
        .mean("time")
    )

    da_hist = (
        ds_resampled[turbulence_index]
        .sel(run=list_historical)
        .mean(["time", "run"])
    )

    # ── Colormap / norm ───────────────────────────────────────────────────────
    base_ticks = np.arange(0, 0.071, 0.005)
    if ticks_max is not None:
        base_ticks = np.arange(0, ticks_max * 1.0001, ticks_max / 14)

    base_cmap = cmap_dict["ipcc_wind_seq"].copy()
    base_cmap.set_bad("lightgrey")
    base_norm = BoundaryNorm(base_ticks, base_cmap.N + 1, extend="both")

    xticks = [100, 120, 140, 160, 180]
    yticks = [-50, -40, -30, -20, -10, 0, 10]
    proj   = ccrs.PlateCarree(130)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axs = plt.subplots(
        nrows=1, ncols=2,
        figsize=figsize,
        subplot_kw={"projection": proj, "frame_on": True},
    )

    fig.suptitle(
        f"{turbulence_index} p99 · {P} hPa · {time_selection}",
        fontsize=fontsize_title,
        fontweight="normal",
        y=1.01,
    )

    panel_data   = [da_eval, da_hist]
    panel_titles = ["(a) BARRA-R", "(b) Historical MMM"]
    last_im = None

    for i, (ax, da, title) in enumerate(zip(axs, panel_data, panel_titles)):

        ax.set_extent([90, 195, -53.58, 13.63], crs=ccrs.PlateCarree())

        last_im = ax.pcolormesh(
            da.lon, da.lat, da,
            cmap=base_cmap, norm=base_norm,
            transform=ccrs.PlateCarree(), zorder=2,
        )

        ax.add_geometries(
            regions_dict["aus_states_territories"]["geometry"],
            crs=ccrs.PlateCarree(),
            facecolor="none", edgecolor="black", linewidth=0.3, zorder=6,
        )
        ax.coastlines(resolution="10m", linewidth=0.25, zorder=5)
        try:
            ax.add_feature(cfeature.BORDERS, linewidth=0.2, zorder=5)
        except Exception:
            pass

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("#aaaaaa")
            spine.set_linewidth(0.8)

        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            linewidth=0.4, color="black", alpha=0.20,
            linestyle="--", draw_labels=True,
        )
        gl.rotate_labels  = False
        gl.xlocator       = mticker.FixedLocator(xticks)
        gl.ylocator       = mticker.FixedLocator(yticks)
        gl.xformatter     = LONGITUDE_FORMATTER
        gl.yformatter     = LATITUDE_FORMATTER
        gl.left_labels    = (i == 0)
        gl.right_labels   = False
        gl.top_labels     = False
        gl.bottom_labels  = True
        gl.xlabel_style   = {"fontsize": 7, "rotation": 0, "ha": "center"}
        gl.ylabel_style   = {"fontsize": 7, "rotation": 0, "ha": "right", "va": "center"}

        ax.set_title(title, loc="left", fontsize=fontsize_subtitle,
                     fontweight="normal", pad=2)

    # ── Shared colorbar ───────────────────────────────────────────────────────
    fig.subplots_adjust(left=0.08, right=0.97, top=0.90, bottom=0.22, wspace=0.18)
    fig.canvas.draw()

    pos     = [ax.get_position() for ax in axs]
    cbar_h  = 0.022
    cbar_gap = 0.055

    cax = fig.add_axes([
        pos[0].x0,
        pos[0].y0 - cbar_gap - cbar_h,
        pos[1].x1 - pos[0].x0,
        cbar_h,
    ])
    cbar = fig.colorbar(
        last_im, cax=cax,
        orientation="horizontal", extend="max",
        ticks=base_ticks,
    )
    cbar.ax.tick_params(labelsize=7, pad=2)
    for j, lbl in enumerate(cbar.ax.xaxis.get_ticklabels()):
        lbl.set_visible(j % 2 == 0)
    cbar.set_label("Frequency [per 6 h]", fontsize=fontsize_cbar, labelpad=4)
    cbar.ax.xaxis.set_label_position("bottom")

    if save_fig and outfile is not None:
        fig.savefig(outfile, dpi=300, bbox_inches="tight")
        print(f"Saved to: {outfile}")

    return fig, axs


# updated func which combines all time selection options onto one figure:

# Define groupings — mirrors the timeseries functions
TIME_GROUPS = {
    "monthly": ["January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"],
    "seasonal": ["DJF", "MAM", "JJA", "SON"],
    "coolwarm": ["MJJASO", "NDJFMA"],
}

def plot_eval_vs_mmmhist(
    turbulence_index,
    P,
    time_selection,          # "annual", "monthly", "seasonal", "coolwarm",
                             # or any single period e.g. "MJJASO", "JJA", "January"
    ds=None,
    filelist=None,
    figsize=None,            # auto-set based on layout if None
    ticks_max=None,
    outfile=None,
    save_fig=False,
):
    """
    Spatial map(s) comparing BARRA-R (left) vs historical MMM (right).

    Single time_selection  → 1x2 figure
    "coolwarm"             → 2x2 figure (MJJASO top, NDJFMA bottom)
    "seasonal"             → 4x2 figure (DJF, MAM, JJA, SON)
    "monthly"              → 12x2 figure (all months)
    """

    # ── Load data if not provided ─────────────────────────────────────────────
    if ds is None:
        if filelist is None:
            filelist = glob.glob(
                f"/scratch/v46/ls7238/CAT_turbulence/{turbulence_index}/{P}hPa/"
                f"freq-above-p99/{turbulence_index}-{P}hPa-monthly-freq-above-p99_AUS-15_*.nc"
            )
        ds = xr.open_mfdataset(
            filelist,
            combine="nested",
            concat_dim="run",
            join="outer",
            coords="different",
            compat="no_conflicts",
        )

    # ── Resolve the list of periods to plot ───────────────────────────────────
    if time_selection in TIME_GROUPS:
        periods = TIME_GROUPS[time_selection]
    else:
        periods = [time_selection]   # single panel pair

    n_rows = len(periods)

    # ── Figure size ───────────────────────────────────────────────────────────
    if figsize is None:
        figsize = (10, 4 * n_rows)

    # ── Colormap / norm ───────────────────────────────────────────────────────
    base_ticks = np.arange(0, 0.071, 0.005)
    if ticks_max is not None:
        base_ticks = np.arange(0, ticks_max * 1.0001, ticks_max / 14)

    base_cmap = cmap_dict["ipcc_wind_seq"].copy()
    base_cmap.set_bad("lightgrey")
    base_norm = BoundaryNorm(base_ticks, base_cmap.N + 1, extend="both")

    xticks = [100, 120, 140, 160, 180]
    yticks = [-50, -40, -30, -20, -10, 0, 10]
    proj   = ccrs.PlateCarree(130)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axs_2d = plt.subplots(
        nrows=n_rows, ncols=2,
        figsize=figsize,
        subplot_kw={"projection": proj, "frame_on": True},
        squeeze=False,   # always 2D array even if n_rows=1
    )

    last_im = None

    for row, period in enumerate(periods):

        ds_resampled = select_resample_time(ds, period)

        da_eval = (
            ds_resampled[turbulence_index]
            .sel(run="evaluation_BARRA-R_r1i1p1f1")
            .mean("time")
        )
        da_hist = (
            ds_resampled[turbulence_index]
            .sel(run=list_historical)
            .mean(["time", "run"])
        )

        panel_data   = [da_eval, da_hist]
        panel_titles = [f"BARRA-R: {period}", f"Historical MMM: {period}"]

        for col, (ax, da, title) in enumerate(zip(axs_2d[row], panel_data, panel_titles)):

            ax.set_extent([90, 195, -53.58, 13.63], crs=ccrs.PlateCarree())

            last_im = ax.pcolormesh(
                da.lon, da.lat, da,
                cmap=base_cmap, norm=base_norm,
                transform=ccrs.PlateCarree(), zorder=2,
            )

            ax.add_geometries(
                regions_dict["aus_states_territories"]["geometry"],
                crs=ccrs.PlateCarree(),
                facecolor="none", edgecolor="black", linewidth=0.3, zorder=6,
            )
            ax.coastlines(resolution="10m", linewidth=0.25, zorder=5)
            try:
                ax.add_feature(cfeature.BORDERS, linewidth=0.2, zorder=5)
            except Exception:
                pass

            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("#aaaaaa")
                spine.set_linewidth(0.8)

            gl = ax.gridlines(
                crs=ccrs.PlateCarree(),
                linewidth=0.4, color="black", alpha=0.20,
                linestyle="--", draw_labels=True,
            )
            gl.rotate_labels  = False
            gl.xlocator       = mticker.FixedLocator(xticks)
            gl.ylocator       = mticker.FixedLocator(yticks)
            gl.xformatter     = LONGITUDE_FORMATTER
            gl.yformatter     = LATITUDE_FORMATTER
            gl.left_labels    = (col == 0)
            gl.right_labels   = False
            gl.top_labels     = False
            gl.bottom_labels  = (row == n_rows - 1)   # only on last row
            gl.xlabel_style   = {"fontsize": 7, "rotation": 0, "ha": "center"}
            gl.ylabel_style   = {"fontsize": 7, "rotation": 0, "ha": "right", "va": "center"}

            ax.set_title(title, loc="left", fontsize=fontsize_subtitle,
                         fontweight="normal", pad=2)

    # ── Shared colorbar at the bottom ─────────────────────────────────────────
    fig.subplots_adjust(left=0.08, right=0.97, top=0.99, bottom=0.08,
                        wspace=0.18)
    fig.canvas.draw()

    panel_height = 0.7 / n_rows   # total plotting area divided by rows
    panel_gap    = 0.01             # fixed small gap between rows

    for row in range(n_rows):
        for col in range(2):
            pos = axs_2d[row, col].get_position()
            axs_2d[row, col].set_position([
                pos.x0,
                0.05 + (n_rows - row - 1) * (panel_height + panel_gap),
                pos.width,
                panel_height,
            ])

    # Redraw to get updated positions for colorbar
    fig.canvas.draw()

    top_of_panels = 0.05 + (n_rows - 1) * (panel_height + panel_gap) + panel_height
    fig.suptitle(
        f"{turbulence_index} p99 · {P} hPa · {time_selection}",
        fontsize=fontsize_title,
        fontweight="normal",
        y=top_of_panels + 0.03,   # just above the top panel
    )

    # Span full width of the bottom row
    pos_bl = axs_2d[-1, 0].get_position()   # bottom-left panel
    pos_br = axs_2d[-1, 1].get_position()   # bottom-right panel

    cbar_h   = 0.018 / n_rows
    cbar_gap = 0.09 / n_rows

    cax = fig.add_axes([
        pos_bl.x0,
        pos_bl.y0 - cbar_gap - cbar_h,
        pos_br.x1 - pos_bl.x0,
        cbar_h,
    ])
    cbar = fig.colorbar(
        last_im, cax=cax,
        orientation="horizontal", extend="both",
        ticks=base_ticks,
    )
    cbar.ax.tick_params(labelsize=7, pad=2)
    for j, lbl in enumerate(cbar.ax.xaxis.get_ticklabels()):
        lbl.set_visible(j % 2 == 0)
    cbar.set_label("Frequency [per 6 h]", fontsize=fontsize_cbar, labelpad=4)
    cbar.ax.xaxis.set_label_position("bottom")

    if save_fig and outfile is not None:
        fig.savefig(outfile, dpi=300, bbox_inches="tight")
        print(f"Saved to: {outfile}")

    return fig, axs_2d


# Plot with 3 coloumns - one for the difference:

def plot_eval_vs_mmmhist_with_diff(
    turbulence_index,
    P,
    time_selection,
    ds=None,
    filelist=None,
    figsize=None,
    ticks_max=None,
    ticks_diff=None,
    share_cbar=True,   
    outfile=None,
    save_fig=False,
):
    """
    Spatial map(s): BARRA-R (left) | Historical MMM (centre) | Difference (right).

    Single time_selection  → 1×3 figure
    "coolwarm"             → 2×3 figure
    "seasonal"             → 4×3 figure
    "monthly"              → 12×3 figure

    share_cbar=True  → one shared colorbar per column group (base + diff)
    share_cbar=False → each row has its own colorbars, scaled to that row's data
                       base ticks run 0→row max, diff ticks are symmetric about 0
    """

    # ── Load data ─────────────────────────────────────────────────────────────
    if ds is None:
        if filelist is None:
            filelist = glob(
                f"/scratch/v46/ls7238/CAT_turbulence/{turbulence_index}/{P}hPa/"
                f"freq-above-p99/{turbulence_index}-{P}hPa-monthly-freq-above-p99_AUS-15_*.nc"
            )
        ds = xr.open_mfdataset(
            filelist,
            combine="nested",
            concat_dim="run",
            join="outer",
            coords="different",
            compat="no_conflicts",
        )

    # ── Periods ───────────────────────────────────────────────────────────────
    if time_selection in TIME_GROUPS:
        periods = TIME_GROUPS[time_selection]
    else:
        periods = [time_selection]

    n_rows = len(periods)

    # ── Pre-compute all data arrays (needed for tick range calculations) ───────
    row_data = []
    for period in periods:
        ds_r = select_resample_time(ds, period)
        da_eval = (
            ds_r[turbulence_index]
            .sel(run="evaluation_BARRA-R_r1i1p1f1")
            .mean("time")
        )
        da_hist = (
            ds_r[turbulence_index]
            .sel(run=list_historical)
            .mean(["time", "run"])
        )
        da_diff = da_hist - da_eval
        row_data.append((da_eval, da_hist, da_diff))

    # ── Figure size ───────────────────────────────────────────────────────────
    if figsize is None:
        figsize = (15, 4 * n_rows) if share_cbar else (15, 4.5 * n_rows)

    # ── Base colormap ─────────────────────────────────────────────────────────
    base_cmap = cmap_dict["ipcc_wind_seq"].copy()
    base_cmap.set_bad("lightgrey")

    # ── Diff colormap ─────────────────────────────────────────────────────────
    diff_cmap = cmap_dict["anom"].copy()
    diff_cmap.set_bad("lightgrey")

    # ── Global ticks (used when share_cbar=True) ──────────────────────────────
    if share_cbar:
        global_base_ticks = np.arange(0, 0.071, 0.005)
        if ticks_max is not None:
            global_base_ticks = np.arange(0, ticks_max * 1.0001, ticks_max / 14)

        if ticks_diff is None:
            global_diff_max = float(np.max([
                np.nanmax(np.abs(da_diff.values))
                for _, _, da_diff in row_data
            ]))
        else:
            global_diff_max = ticks_diff
        diff_step = global_diff_max / 7
        global_diff_ticks = np.arange(
            -global_diff_max, global_diff_max + diff_step * 0.001, diff_step
        )

    # ── Shared map settings ───────────────────────────────────────────────────
    xticks = [100, 120, 140, 160, 180]
    yticks = [-50, -40, -30, -20, -10, 0, 10]
    proj   = ccrs.PlateCarree(130)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axs_2d = plt.subplots(
        nrows=n_rows, ncols=3,
        figsize=figsize,
        subplot_kw={"projection": proj, "frame_on": True},
        squeeze=False,
    )

    # Store the last plotted images per row for per-row colorbars,
    # or globally for shared colorbars
    row_ims = []   # list of (base_im, diff_im) per row

    for row, (period, (da_eval, da_hist, da_diff)) in enumerate(
        zip(periods, row_data)
    ):
        # ── Per-row ticks ──────────────────────────────────────────────────────
        if share_cbar:
            base_ticks = global_base_ticks
            diff_ticks = global_diff_ticks
        else:
            row_base_max = float(np.nanmax([
                np.nanmax(da_eval.values),
                np.nanmax(da_hist.values),
            ]))
            if ticks_max is not None:
                row_base_max = ticks_max
            base_ticks = np.arange(0, row_base_max * 1.0001, row_base_max / 14)

            row_diff_max = float(np.nanmax(np.abs(da_diff.values)))
            diff_step    = row_diff_max / 7
            diff_ticks   = np.arange(
                -row_diff_max, row_diff_max + diff_step * 0.001, diff_step
            )

        base_norm = BoundaryNorm(base_ticks, base_cmap.N + 1, extend="both")
        diff_norm = BoundaryNorm(diff_ticks, diff_cmap.N + 1, extend="both")

        panel_data   = [da_eval,                    da_hist,                       da_diff]
        panel_titles = [f"BARRA-R: {period}",       f"Historical MMM: {period}",   f"MMM - BARRA-R: {period}"]
        panel_cmaps  = [base_cmap,                  base_cmap,                     diff_cmap]
        panel_norms  = [base_norm,                  base_norm,                     diff_norm]

        base_im = diff_im = None

        for col, (ax, da, title, cmap, norm) in enumerate(
            zip(axs_2d[row], panel_data, panel_titles, panel_cmaps, panel_norms)
        ):
            ax.set_extent([90, 195, -53.58, 13.63], crs=ccrs.PlateCarree())

            im = ax.pcolormesh(
                da.lon, da.lat, da,
                cmap=cmap, norm=norm,
                transform=ccrs.PlateCarree(), zorder=2,
            )
            if col < 2:
                base_im = im
            else:
                diff_im = im

            ax.add_geometries(
                regions_dict["aus_states_territories"]["geometry"],
                crs=ccrs.PlateCarree(),
                facecolor="none", edgecolor="black", linewidth=0.3, zorder=6,
            )
            ax.coastlines(resolution="10m", linewidth=0.25, zorder=5)
            try:
                ax.add_feature(cfeature.BORDERS, linewidth=0.2, zorder=5)
            except Exception:
                pass

            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("#aaaaaa")
                spine.set_linewidth(0.8)

            gl = ax.gridlines(
                crs=ccrs.PlateCarree(),
                linewidth=0.4, color="black", alpha=0.20,
                linestyle="--", draw_labels=True,
            )
            gl.rotate_labels  = False
            gl.xlocator       = mticker.FixedLocator(xticks)
            gl.ylocator       = mticker.FixedLocator(yticks)
            gl.xformatter     = LONGITUDE_FORMATTER
            gl.yformatter     = LATITUDE_FORMATTER
            gl.left_labels    = (col == 0)
            gl.right_labels   = False
            gl.top_labels     = False
            gl.bottom_labels  = (row == n_rows - 1) if share_cbar else False
            gl.xlabel_style   = {"fontsize": 7, "rotation": 0, "ha": "center"}
            gl.ylabel_style   = {"fontsize": 7, "rotation": 0, "ha": "right", "va": "center"}

            ax.set_title(title, loc="left", fontsize=fontsize_subtitle,
                         fontweight="normal", pad=2)

        row_ims.append((base_im, diff_im, base_ticks, diff_ticks))

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.subplots_adjust(left=0.06, right=0.97, top=0.99, bottom=0.08,
                        wspace=0.18)
    fig.canvas.draw()

    if share_cbar:
        cbar_h       = 0.018 / n_rows
        cbar_gap     = 0.09  / n_rows
        panel_gap    = 0.01
        panel_height = 0.7 / n_rows
    else:
        cbar_h       = 0.030 / n_rows
        cbar_gap     = 0.012 / n_rows
        inter_gap    = 0.020 / n_rows
        panel_gap    = cbar_gap + cbar_h + inter_gap
        panel_height = (0.85 - n_rows * (cbar_gap + cbar_h)
                        - (n_rows - 1) * inter_gap) / n_rows

    for row in range(n_rows):
        for col in range(3):
            pos = axs_2d[row, col].get_position()
            axs_2d[row, col].set_position([
                pos.x0,
                0.05 + (n_rows - row - 1) * (panel_height + panel_gap),
                pos.width,
                panel_height,
            ])

    fig.canvas.draw()

    top_of_panels = 0.05 + (n_rows - 1) * (panel_height + panel_gap) + panel_height
    fig.suptitle(
        f"{turbulence_index} p99 · {P} hPa · {time_selection}",
        fontsize=fontsize_title,
        fontweight="normal",
        y=top_of_panels + 0.03,
    )

    # ── Colorbars ─────────────────────────────────────────────────────────────
    def _add_cbar(im, ticks, ax_left, ax_right, y_bottom, label):
        pos_l = ax_left.get_position()
        pos_r = ax_right.get_position()
        cax = fig.add_axes([
            pos_l.x0,
            y_bottom,
            pos_r.x1 - pos_l.x0,
            cbar_h,
        ])
        cbar = fig.colorbar(im, cax=cax, orientation="horizontal",
                            extend="both", ticks=ticks)
        cbar.ax.tick_params(labelsize=7, pad=2)
        for j, lbl in enumerate(cbar.ax.xaxis.get_ticklabels()):
            lbl.set_visible(j % 2 == 0)
        cbar.set_label(label, fontsize=fontsize_cbar, labelpad=4)
        cbar.ax.xaxis.set_label_position("bottom")
        return cbar

    if share_cbar:
        # Two colorbars under the bottom row only
        base_im, diff_im, base_ticks, diff_ticks = row_ims[-1]
        pos_bl = axs_2d[-1, 0].get_position()
        pos_bm = axs_2d[-1, 1].get_position()
        pos_br = axs_2d[-1, 2].get_position()
        y_cbar = pos_bl.y0 - cbar_gap - cbar_h

        _add_cbar(base_im, global_base_ticks,
                  axs_2d[-1, 0], axs_2d[-1, 1], y_cbar, "Frequency [per 6 h]")
        _add_cbar(diff_im, global_diff_ticks,
                  axs_2d[-1, 2], axs_2d[-1, 2], y_cbar, "Difference [per 6 h]")
    else:
        # One pair of colorbars per row
        for row, (base_im, diff_im, base_ticks, diff_ticks) in enumerate(row_ims):
            pos_row = axs_2d[row, 0].get_position()
            y_cbar  = pos_row.y0 - cbar_gap - cbar_h

            _add_cbar(base_im, base_ticks,
                      axs_2d[row, 0], axs_2d[row, 1], y_cbar, "Frequency [per 6 h]")
            _add_cbar(diff_im, diff_ticks,
                      axs_2d[row, 2], axs_2d[row, 2], y_cbar, "Difference [per 6 h]")

    if save_fig and outfile is not None:
        fig.savefig(outfile, dpi=300, bbox_inches="tight")
        print(f"Saved to: {outfile}")

    return fig, axs_2d





# PLOT AND PREP FUTURE SCENARIOS AND HISTORICAL DIFFERENCE PER TIME SELECTION

import pickle


def prep_hist_vs_future_with_diff(
    turbulence_index,
    P,
    time_selection,
    experiment="ssp370",
    future_period=("2040", "2059"),
    all_agree_except=1,
    pickle_path=None,   
):
    """
    Load and compute all per-period arrays needed by draw_hist_vs_future_with_diff.
    Optionally saves the result to a pickle file so it can be reloaded instantly.

    Parameters
    ----------
    pickle_path : str or None
        If given, saves the prepped dict to this path after computation.
        Load it later with:
            import pickle
            with open(pickle_path, "rb") as f:
                prepped = pickle.load(f)
    """

    if time_selection in TIME_GROUPS:
        periods = TIME_GROUPS[time_selection]
    else:
        periods = [time_selection]

    # ── File discovery ────────────────────────────────────────────────────────
    glob_list = glob(
        f"/scratch/v46/ls7238/CAT_turbulence/{turbulence_index}/{P}hPa/freq-above-p99/"
        f"{turbulence_index}-{P}hPa-monthly-freq-above-p99_AUS-15_*_BOM_BARPA-R_v1-r1_6hr.nc"
    )

    def _make_path(run):
        return (
            f"/scratch/v46/ls7238/CAT_turbulence/{turbulence_index}/{P}hPa/freq-above-p99/"
            f"{turbulence_index}-{P}hPa-monthly-freq-above-p99_AUS-15_{run}_BOM_BARPA-R_v1-r1_6hr.nc"
        )

    hist_files   = [f for f in [_make_path(r) for r in list_historical]         if f in glob_list]
    future_runs  = [experiment + r[10:] for r in list_historical]
    future_files = [f for f in [_make_path(r) for r in future_runs]              if f in glob_list]

    def _preprocess(ds):
        return ds.sel(lon=lon_slice)

    ds_hist_raw = xr.open_mfdataset(
        hist_files,
        concat_dim="run", combine="nested",
        preprocess=_preprocess,
        join="outer", coords="different", compat="no_conflicts",
        chunks={"lat": 160, "lon": -1}
    )
    ds_hist_raw["run"] = [r[r.index("_")+1:] for r in ds_hist_raw["run"].values]

    ds_future_raw = xr.open_mfdataset(
        future_files,
        concat_dim="run", combine="nested",
        preprocess=_preprocess,
        join="outer", coords="different", compat="no_conflicts",
        chunks={"lat": 160, "lon": -1}
    )
    ds_future_raw["run"] = [r[r.index("_")+1:] for r in ds_future_raw["run"].values]

    common_runs = [r for r in ds_hist_raw["run"].values
                   if r in set(ds_future_raw["run"].values)]
    print(f"Common runs for {experiment}: {common_runs}")

    ds_hist_raw   = ds_hist_raw.sel(run=common_runs)
    ds_future_raw = ds_future_raw.sel(run=common_runs)

    agreement_threshold = len(common_runs) - all_agree_except

    # Concatenate so select_resample_time spans the full record
    ds_combined = xr.concat(
        [ds_hist_raw, ds_future_raw],
        dim="time", coords="minimal", join="outer", compat="override",
    )

    # ── Per-period computation ─────────────────────────────────────────────────
    row_data = {}
    for period in periods:
        print(f"  Processing {period}...")
        ds_r = select_resample_time(ds_combined, period)

        da_hist_runs   = ds_r[turbulence_index].sel(time=baseline_time_slice).mean("time")
        da_future_runs = ds_r[turbulence_index].sel(time=slice(*future_period)).mean("time")

        da_hist_mmm   = da_hist_runs.mean("run")
        da_future_mmm = da_future_runs.mean("run")

        da_delta_runs = da_future_runs - da_hist_runs
        da_diff_mmm   = da_delta_runs.mean("run")

        agree_mask = (
            ((da_delta_runs > 0).sum("run") >= agreement_threshold) |
            ((da_delta_runs < 0).sum("run") >= agreement_threshold)
        )
        agree_mask = (agree_mask.coarsen(lat=20, lon=20, boundary="pad").mean() > 0.4)

        # Load into memory now so the pickle is self-contained
        row_data[period] = (
            da_hist_mmm.load(),
            da_future_mmm.load(),
            da_diff_mmm.load(),
            agree_mask.load(),
            da_delta_runs.load()
        )
        print(f"  Done {period}.")

    # close file handles now all data is loaded into memory
    ds_combined.close()
    ds_hist_raw.close()
    ds_future_raw.close()


    prepped = {
        "periods":           periods,
        "row_data":          row_data,
        "time_selection":    time_selection,
        "turbulence_index":  turbulence_index,
        "P":                 P,
        "experiment":        experiment,
        "future_period":     future_period,
        "common_runs":       common_runs,
    }

    if pickle_path is not None:
        with open(pickle_path, "wb") as f:
            pickle.dump(prepped, f)
        print(f"Saved prepped data to: {pickle_path}")

    return prepped


def draw_hist_vs_future_with_diff(
    prepped,
    figsize=None,
    ticks_max=None,
    ticks_diff=None,
    share_cbar=True,
    outfile=None,
    save_fig=False,
    time_selection=None,
):
    """
    Draw the three-column map from prepped data produced by prep_hist_vs_future_with_diff.

    Col 0 : Historical MMM (1990-2009)
    Col 1 : Future scenario MMM
    Col 2 : Future - Historical, with model-agreement stippling

    share_cbar=True  -> one colorbar pair at the bottom of the figure
    share_cbar=False -> per-row colorbars scaled to each row's data
    """

    periods  = _resolve_periods(prepped, time_selection)
    row_data = [prepped["row_data"][p] for p in periods]
    time_selection = time_selection or prepped["time_selection"]
    turbulence_index = prepped["turbulence_index"]
    P                = prepped["P"]
    experiment       = prepped["experiment"]
    future_period    = prepped["future_period"]

    n_rows = len(periods)

    if figsize is None:
        figsize = (15, 4 * n_rows) if share_cbar else (15, 4.5 * n_rows)

    # ── Colormaps ─────────────────────────────────────────────────────────────
    base_cmap = cmap_dict["ipcc_wind_seq"].copy()
    base_cmap.set_bad("lightgrey")
    diff_cmap = cmap_dict["anom"].copy()
    diff_cmap.set_bad("lightgrey")

    # ── Global ticks ──────────────────────────────────────────────────────────
    if share_cbar:
        global_base_ticks = np.arange(0, 0.071, 0.005)
        if ticks_max is not None:
            global_base_ticks = np.arange(0, ticks_max * 1.0001, ticks_max / 14)

        global_diff_max = ticks_diff if ticks_diff is not None else float(
            np.max([np.nanmax(np.abs(da_diff.values)) for _, _, da_diff, _, _ in row_data])
        )
        diff_step = global_diff_max / 7
        global_diff_ticks = np.arange(
            -global_diff_max, global_diff_max + diff_step * 0.001, diff_step
        )

    # --------------------------------- Plot formatting ---------------------------------
    xticks = [100, 120, 140, 160, 180]
    yticks = [-50, -40, -30, -20, -10, 0, 10]
    proj   = ccrs.PlateCarree(130)

    fig, axs_2d = plt.subplots(
        nrows=n_rows, ncols=3,
        figsize=figsize,
        subplot_kw={"projection": proj, "frame_on": True},
        squeeze=False,
    )

    row_ims = []

    for row, (period, (da_hist_mmm, da_future_mmm, da_diff_mmm, agree_mask, _)) in enumerate(
        zip(periods, row_data)
    ):
        if share_cbar:
            base_ticks = global_base_ticks
            diff_ticks = global_diff_ticks
        else:
            row_base_max = ticks_max if ticks_max is not None else float(
                np.nanmax([np.nanmax(da_hist_mmm.values), np.nanmax(da_future_mmm.values)])
            )
            base_ticks = np.arange(0, row_base_max * 1.0001, row_base_max / 14)

            row_diff_max = float(np.nanmax(np.abs(da_diff_mmm.values)))
            diff_step    = row_diff_max / 7
            diff_ticks   = np.arange(
                -row_diff_max, row_diff_max + diff_step * 0.001, diff_step
            )

        base_norm = BoundaryNorm(base_ticks, base_cmap.N + 1, extend="both")
        diff_norm = BoundaryNorm(diff_ticks, diff_cmap.N + 1, extend="both")

        future_label = f"{future_period[0]}-{future_period[1]}"
        panel_data   = [da_hist_mmm,                  da_future_mmm,                                      da_diff_mmm]
        panel_titles = [f"Historical MMM: {period}",  f"{experiment.upper()} MMM ({future_label}): {period}", f"{experiment.upper()} - Historical: {period}"]
        panel_cmaps  = [base_cmap,                    base_cmap,                                          diff_cmap]
        panel_norms  = [base_norm,                    base_norm,                                          diff_norm]

        base_im = diff_im = None

        for col, (ax, da, title, cmap, norm) in enumerate(
            zip(axs_2d[row], panel_data, panel_titles, panel_cmaps, panel_norms)
        ):
            ax.set_extent([90, 195, -53.58, 13.63], crs=ccrs.PlateCarree())

            im = ax.pcolormesh(
                da.lon, da.lat, da,
                cmap=cmap, norm=norm,
                transform=ccrs.PlateCarree(), zorder=2,
            )
            if col < 2:
                base_im = im
            else:
                diff_im = im
                ax.contourf(
                    agree_mask.lon, agree_mask.lat, agree_mask,
                    alpha=0, hatches=["", "xxxxxx"],
                    transform=ccrs.PlateCarree(), zorder=4,
                )

            ax.add_geometries(
                regions_dict["aus_states_territories"]["geometry"],
                crs=ccrs.PlateCarree(),
                facecolor="none", edgecolor="black", linewidth=0.3, zorder=6,
            )
            ax.coastlines(resolution="10m", linewidth=0.25, zorder=5)
            try:
                ax.add_feature(cfeature.BORDERS, linewidth=0.2, zorder=5)
            except Exception:
                pass

            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("#aaaaaa")
                spine.set_linewidth(0.8)

            gl = ax.gridlines(
                crs=ccrs.PlateCarree(),
                linewidth=0.4, color="black", alpha=0.20,
                linestyle="--", draw_labels=True,
            )
            gl.rotate_labels  = False
            gl.xlocator       = mticker.FixedLocator(xticks)
            gl.ylocator       = mticker.FixedLocator(yticks)
            gl.xformatter     = LONGITUDE_FORMATTER
            gl.yformatter     = LATITUDE_FORMATTER
            gl.left_labels    = (col == 0)
            gl.right_labels   = False
            gl.top_labels     = False
            gl.bottom_labels  = (row == n_rows - 1) if share_cbar else False
            gl.xlabel_style   = {"fontsize": 7, "rotation": 0, "ha": "center"}
            gl.ylabel_style   = {"fontsize": 7, "rotation": 0, "ha": "right", "va": "center"}

            ax.set_title(title, loc="left", fontsize=fontsize_subtitle,
                         fontweight="normal", pad=2)

        row_ims.append((base_im, diff_im, base_ticks, diff_ticks))

    # --------------------------------- Layout ---------------------------------
    fig.subplots_adjust(left=0.06, right=0.97, top=0.99, bottom=0.08, wspace=0.18)
    fig.canvas.draw()

    if share_cbar:
        cbar_h       = 0.018 / n_rows
        cbar_gap     = 0.12  / n_rows   
        panel_gap    = 0.01
        panel_height = 0.7 / n_rows
    else:
        cbar_h       = 0.030 / n_rows
        cbar_gap     = 0.025 / n_rows   
        inter_gap    = 0.020 / n_rows
        panel_gap    = cbar_gap + cbar_h + inter_gap
        panel_height = (0.85 - n_rows * (cbar_gap + cbar_h)
                        - (n_rows - 1) * inter_gap) / n_rows

    for row in range(n_rows):
        for col in range(3):
            pos = axs_2d[row, col].get_position()
            axs_2d[row, col].set_position([
                pos.x0,
                0.05 + (n_rows - row - 1) * (panel_height + panel_gap),
                pos.width,
                panel_height,
            ])

    fig.canvas.draw()

    top_of_panels = 0.05 + (n_rows - 1) * (panel_height + panel_gap) + panel_height
    fig.suptitle(
        f"{turbulence_index} p99 · {P} hPa · {experiment.upper()} {future_period[0]}–{future_period[1]} · {time_selection}",
        fontsize=fontsize_title,
        fontweight="normal",
        y=top_of_panels + 0.03,
    )

    # ── Colorbars ---------------------------------
    def _add_cbar(im, ticks, ax_left, ax_right, y_bottom, label):
        pos_l = ax_left.get_position()
        pos_r = ax_right.get_position()
        cax = fig.add_axes([pos_l.x0, y_bottom, pos_r.x1 - pos_l.x0, cbar_h])
        cbar = fig.colorbar(im, cax=cax, orientation="horizontal",
                            extend="both", ticks=ticks)
        cbar.ax.tick_params(labelsize=7, pad=2)
        for j, lbl in enumerate(cbar.ax.xaxis.get_ticklabels()):
            lbl.set_visible(j % 2 == 0)
        cbar.set_label(label, fontsize=fontsize_cbar, labelpad=4)
        cbar.ax.xaxis.set_label_position("bottom")
        return cbar

    if share_cbar:
        y_cbar = axs_2d[-1, 0].get_position().y0 - cbar_gap - cbar_h
        _add_cbar(row_ims[-1][0], global_base_ticks,
                  axs_2d[-1, 0], axs_2d[-1, 1], y_cbar, "Frequency [per 6 h]")
        _add_cbar(row_ims[-1][1], global_diff_ticks,
                  axs_2d[-1, 2], axs_2d[-1, 2], y_cbar, "Change in frequency [per 6 h]")
    else:
        for row, (base_im, diff_im, base_ticks, diff_ticks) in enumerate(row_ims):
            y_cbar = axs_2d[row, 0].get_position().y0 - cbar_gap - cbar_h
            _add_cbar(base_im, base_ticks,
                      axs_2d[row, 0], axs_2d[row, 1], y_cbar, "Frequency [per 6 h]")
            _add_cbar(diff_im, diff_ticks,
                      axs_2d[row, 2], axs_2d[row, 2], y_cbar, "Change in frequency [per 6 h]")

    if save_fig and outfile is not None:
        fig.savefig(outfile, dpi=300, bbox_inches="tight")
        print(f"Saved to: {outfile}")

    return fig, axs_2d


def _resolve_periods(prepped, time_selection=None):
    available = list(prepped["row_data"].keys())
    if time_selection is None:
        return available
    if time_selection in TIME_GROUPS:
        return [p for p in TIME_GROUPS[time_selection] if p in prepped["row_data"]]
    if time_selection in prepped["row_data"]:
        return [time_selection]
    raise ValueError(f"'{time_selection}' not in TIME_GROUPS or prepped row_data")


def draw_per_model_diff(
    prepped,
    figsize=None,
    ticks_diff=None,          # None=auto per period, scalar=shared, dict={period: val}
    outfile=None,
    save_fig=False,
    time_selection=None,
    n_cols=3,
):
    periods  = _resolve_periods(prepped, time_selection)
    row_data = [prepped["row_data"][p] for p in periods]
    common_runs      = prepped["common_runs"]
    turbulence_index = prepped["turbulence_index"]
    P                = prepped["P"]
    experiment       = prepped["experiment"]
    future_period    = prepped["future_period"]
    time_selection_label = time_selection or prepped["time_selection"]

    n_periods    = len(periods)
    n_models     = len(common_runs)
    n_model_rows = math.ceil(n_models / n_cols)
    n_rows       = n_periods * n_model_rows

    diff_cmap = cmap_dict["anom"].copy()
    diff_cmap.set_bad("lightgrey")

    if figsize is None:
        figsize = (4 * n_cols, 3.5 * n_rows)

    proj   = ccrs.PlateCarree(130)
    xticks = [100, 120, 140, 160, 180]
    yticks = [-50, -40, -30, -20, -10, 0, 10]

    fig, axs_2d = plt.subplots(
        nrows=n_rows, ncols=n_cols,
        figsize=figsize,
        subplot_kw={"projection": proj, "frame_on": True},
        squeeze=False,
    )

    period_ims = []   # (last_im, diff_ticks) one entry per period

    for period_idx, (period, row_tuple) in enumerate(zip(periods, row_data)):
        da_delta_runs = row_tuple[4]

        # ── Per-period ticks ──────────────────────────────────────────────────
        if ticks_diff is None:
            period_diff_max = float(np.nanmax(np.abs(da_delta_runs.values)))
        elif isinstance(ticks_diff, dict):
            period_diff_max = ticks_diff.get(period, float(np.nanmax(np.abs(da_delta_runs.values))))
        else:
            period_diff_max = float(ticks_diff)

        period_diff_step  = period_diff_max / 7
        period_diff_ticks = np.arange(
            -period_diff_max, period_diff_max + period_diff_step * 0.001, period_diff_step
        )
        period_diff_norm = BoundaryNorm(period_diff_ticks, diff_cmap.N + 1, extend="both")

        last_im = None
        last_fig_row_of_period = (period_idx + 1) * n_model_rows - 1

        for model_idx, run in enumerate(common_runs):
            fig_row = period_idx * n_model_rows + model_idx // n_cols
            fig_col = model_idx % n_cols
            ax = axs_2d[fig_row, fig_col]
            da = da_delta_runs.sel(run=run)

            ax.set_extent([90, 195, -53.58, 13.63], crs=ccrs.PlateCarree())
            last_im = ax.pcolormesh(
                da.lon, da.lat, da,
                cmap=diff_cmap, norm=period_diff_norm,
                transform=ccrs.PlateCarree(), zorder=2,
            )
            ax.add_geometries(
                regions_dict["aus_states_territories"]["geometry"],
                crs=ccrs.PlateCarree(),
                facecolor="none", edgecolor="black", linewidth=0.3, zorder=6,
            )
            ax.coastlines(resolution="10m", linewidth=0.25, zorder=5)
            try:
                ax.add_feature(cfeature.BORDERS, linewidth=0.2, zorder=5)
            except Exception:
                pass
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("#aaaaaa")
                spine.set_linewidth(0.8)

            run_label = run.split("_")[0] if "_" in run else run
            ax.set_title(f"{run_label}: {period}", loc="left",
                         fontsize=fontsize_subtitle, fontweight="normal", pad=2)

            gl = ax.gridlines(
                crs=ccrs.PlateCarree(),
                linewidth=0.4, color="black", alpha=0.20,
                linestyle="--", draw_labels=True,
            )
            gl.rotate_labels  = False
            gl.xlocator       = mticker.FixedLocator(xticks)
            gl.ylocator       = mticker.FixedLocator(yticks)
            gl.xformatter     = LONGITUDE_FORMATTER
            gl.yformatter     = LATITUDE_FORMATTER
            gl.left_labels    = (fig_col == 0)
            gl.right_labels   = False
            gl.top_labels     = False
            gl.bottom_labels  = (fig_row == last_fig_row_of_period)
            gl.xlabel_style   = {"fontsize": 7, "rotation": 0, "ha": "center"}
            gl.ylabel_style   = {"fontsize": 7, "rotation": 0, "ha": "right", "va": "center"}

        # hide unused axes in last model row of this period
        for empty_col in range(n_models % n_cols or n_cols, n_cols):
            axs_2d[last_fig_row_of_period, empty_col].set_visible(False)

        period_ims.append((last_im, period_diff_ticks))

    # ── Layout — compute row y-positions from bottom up ───────────────────────
    cbar_h       = 0.018
    cbar_gap     = 0.012
    panel_gap    = 0.008   # between rows within a period
    period_sep   = 0.015   # extra gap between period groups

    # work out panel height to fill the figure
    total_fixed = (
        0.05                                              # bottom margin
        + n_periods * (cbar_h + cbar_gap)                # colorbars + gaps above them
        + (n_rows - n_periods) * panel_gap               # within-period row gaps
        + (n_periods - 1) * period_sep                   # between-period separators
        + 0.08                                            # top margin for suptitle
    )
    panel_h = (1.0 - total_fixed) / n_rows

    fig.subplots_adjust(left=0.05, right=0.97, top=0.95, bottom=0.05,
                        wspace=0.15, hspace=0.3)
    fig.canvas.draw()

    # compute y_bottom for every row and every colorbar
    row_y_bottoms = {}
    cbar_ys       = {}
    y_cursor = 0.05

    for period_idx in range(n_periods - 1, -1, -1):   # bottom period first
        cbar_ys[period_idx] = y_cursor
        y_cursor += cbar_h + cbar_gap

        for row_within in range(n_model_rows - 1, -1, -1):   # bottom row of period first
            fig_row = period_idx * n_model_rows + row_within
            row_y_bottoms[fig_row] = y_cursor
            y_cursor += panel_h
            if row_within > 0:
                y_cursor += panel_gap

        if period_idx > 0:
            y_cursor += period_sep

    for r in range(n_rows):
        for c in range(n_cols):
            pos = axs_2d[r, c].get_position()
            axs_2d[r, c].set_position([pos.x0, row_y_bottoms[r], pos.width, panel_h])

    fig.canvas.draw()

    # ── One colorbar per period ───────────────────────────────────────────────
    for period_idx, (period_im, period_diff_ticks) in enumerate(period_ims):
        first_fig_row = period_idx * n_model_rows
        pos_l = axs_2d[first_fig_row, 0].get_position()
        pos_r = axs_2d[first_fig_row, n_cols - 1].get_position()

        cax  = fig.add_axes([pos_l.x0, cbar_ys[period_idx], pos_r.x1 - pos_l.x0, cbar_h])
        cbar = fig.colorbar(period_im, cax=cax, orientation="horizontal",
                            extend="both", ticks=period_diff_ticks)
        cbar.ax.tick_params(labelsize=7, pad=2)
        for j, lbl in enumerate(cbar.ax.xaxis.get_ticklabels()):
            lbl.set_visible(j % 2 == 0)
        cbar.set_label("Change in frequency [per 6 h]", fontsize=fontsize_cbar, labelpad=4)
        cbar.ax.xaxis.set_label_position("bottom")

    top_of_panels = row_y_bottoms[0] + panel_h
    fig.suptitle(
        f"{turbulence_index} p99 · {P} hPa · {experiment.upper()} "
        f"{future_period[0]}–{future_period[1]} · {time_selection_label} · per model",
        fontsize=fontsize_title, fontweight="normal",
        y=top_of_panels + 0.03,
    )

    if save_fig and outfile is not None:
        fig.savefig(outfile, dpi=300, bbox_inches="tight")
        print(f"Saved to: {outfile}")

    return fig, axs_2d


def _muted_cmap(cmap_in, strength=0.35):
    """Blend a colormap toward white to create a muted/washed-out version."""
    rgba = cmap_in(np.linspace(0, 1, 256))
    rgba[:, :3] = rgba[:, :3] * strength + (1 - strength)
    return LinearSegmentedColormap.from_list("muted", rgba)

def draw_diff_and_agreement(
    prepped,
    figsize=None,
    ticks_diff=None,          
    outfile=None,
    save_fig=False,
    time_selection=None,
    zero_threshold=0.0,
):
    periods  = _resolve_periods(prepped, time_selection)
    row_data = [prepped["row_data"][p] for p in periods]
    time_selection_label = time_selection or prepped["time_selection"]
    turbulence_index = prepped["turbulence_index"]
    P                = prepped["P"]
    experiment       = prepped["experiment"]
    future_period    = prepped["future_period"]

    n_rows = len(periods)

    if figsize is None:
        figsize = (10, 4 * n_rows)

    # ── Diff colormap ─────────────────────────────────────────────────────────
    diff_cmap = cmap_dict["anom"].copy()
    diff_cmap.set_bad("lightgrey")

    # ── Agreement colormap ────────────────────────────────────────────────────
    _anom_muted = _muted_cmap(cmap_dict["anom"], strength=0.6)

    blues = [_anom_muted(x) for x in np.linspace(0.05, 0.42, 5)]  # −7 … −3, dark→light
    reds  = [_anom_muted(x) for x in np.linspace(0.58, 0.95, 5)]  # +3 … +7, light→dark

    agree_colors = (
        blues
        + ["#ffffff", "#ffffff"]       # −2, −1
        + ["#cecece"]                  #  0 (disagreement)
        + ["#ffffff", "#ffffff"]       # +1, +2
        + reds
    )
    agree_cmap   = ListedColormap(agree_colors)
    agree_cmap.set_bad("lightgrey")
    agree_bounds = np.arange(-7.5, 8.5, 1.0)
    agree_norm   = BoundaryNorm(agree_bounds, agree_cmap.N)
    agree_ticks  = np.arange(-7, 8, 1)


    unanimous_zero_color = "#c5f6b2bc"

    xticks = [100, 120, 140, 160, 180]
    yticks = [-50, -40, -30, -20, -10, 0, 10]
    proj   = ccrs.PlateCarree(130)

    fig, axs_2d = plt.subplots(
        nrows=n_rows, ncols=2,
        figsize=figsize,
        subplot_kw={"projection": proj, "frame_on": True},
        squeeze=False,
    )

    row_ims = []   # (diff_im, agree_im, diff_ticks) per row

    for row, (period, (da_hist_mmm, da_future_mmm, da_diff_mmm, agree_mask, da_delta_runs)) in enumerate(
        zip(periods, row_data)
    ):
        # ── Per-period diff ticks ─────────────────────────────────────────────
        if ticks_diff is None:
            diff_max = float(np.nanmax(np.abs(da_diff_mmm.values)))
        elif isinstance(ticks_diff, dict):
            diff_max = ticks_diff.get(period, float(np.nanmax(np.abs(da_diff_mmm.values))))
        else:
            diff_max = float(ticks_diff)

        diff_step  = diff_max / 7
        diff_ticks = np.arange(-diff_max, diff_max + diff_step * 0.001, diff_step)
        diff_norm  = BoundaryNorm(diff_ticks, diff_cmap.N + 1, extend="both")

        # ── Net agreement score ───────────────────────────────────────────────
        da_sign   = xr.where(da_delta_runs >  zero_threshold,  1,
                    xr.where(da_delta_runs < -zero_threshold, -1, 0))
        net_score = da_sign.sum("run")

        # coarsen to same resolution as stippling
        lat_c = 5
        lon_c = 5
        net_score = net_score.coarsen(lat=lat_c, lon=lon_c, boundary="pad").mean().round()

        n_pos = (da_delta_runs > zero_threshold).sum("run")
        n_neg = (da_delta_runs < -zero_threshold).sum("run")
        unanimous_zero = (net_score == 0) & (n_pos.coarsen(lat=lat_c, lon=lon_c, boundary="pad").mean() == 0) & (n_neg.coarsen(lat=lat_c, lon=lon_c, boundary="pad").mean() == 0)
        net_score_plot = net_score.where(~unanimous_zero)


        diff_im = agree_im = None

        for col in range(2):
            ax = axs_2d[row, col]
            ax.set_extent([90, 195, -53.58, 13.63], crs=ccrs.PlateCarree())

            if col == 0:
                diff_im = ax.pcolormesh(
                    da_diff_mmm.lon, da_diff_mmm.lat, da_diff_mmm,
                    cmap=diff_cmap, norm=diff_norm,
                    transform=ccrs.PlateCarree(), zorder=2,
                )
                # model agreement stippling
                ax.contourf(
                    agree_mask.lon, agree_mask.lat, agree_mask,
                    alpha=0, hatches=["", "xxxxxx"],
                    transform=ccrs.PlateCarree(), zorder=4,
                )
                ax.set_title(f"{experiment.upper()} − Historical MMM: {period}",
                             loc="left", fontsize=fontsize_subtitle,
                             fontweight="normal", pad=2)
            else:
                agree_im = ax.pcolormesh(
                    net_score_plot.lon, net_score_plot.lat, net_score_plot,
                    cmap=agree_cmap, norm=agree_norm,
                    transform=ccrs.PlateCarree(), zorder=2,
                )
                uz_data = xr.where(unanimous_zero, 1.0, np.nan)
                ax.pcolormesh(
                    uz_data.lon, uz_data.lat, uz_data,
                    cmap=ListedColormap([unanimous_zero_color]),
                    vmin=0.5, vmax=1.5,
                    transform=ccrs.PlateCarree(), zorder=3,
                )
                ax.set_title(f"Net model agreement: {period}",
                             loc="left", fontsize=fontsize_subtitle,
                             fontweight="normal", pad=2)

            ax.add_geometries(
                regions_dict["aus_states_territories"]["geometry"],
                crs=ccrs.PlateCarree(),
                facecolor="none", edgecolor="black", linewidth=0.3, zorder=6,
            )
            ax.coastlines(resolution="10m", linewidth=0.25, zorder=5)
            try:
                ax.add_feature(cfeature.BORDERS, linewidth=0.2, zorder=5)
            except Exception:
                pass
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("#aaaaaa")
                spine.set_linewidth(0.8)

            gl = ax.gridlines(
                crs=ccrs.PlateCarree(),
                linewidth=0.4, color="black", alpha=0.20,
                linestyle="--", draw_labels=True,
            )
            gl.rotate_labels  = False
            gl.xlocator       = mticker.FixedLocator(xticks)
            gl.ylocator       = mticker.FixedLocator(yticks)
            gl.xformatter     = LONGITUDE_FORMATTER
            gl.yformatter     = LATITUDE_FORMATTER
            gl.left_labels    = (col == 0)
            gl.right_labels   = False
            gl.top_labels     = False
            gl.bottom_labels  = (row == n_rows - 1)
            gl.xlabel_style   = {"fontsize": 7, "rotation": 0, "ha": "center"}
            gl.ylabel_style   = {"fontsize": 7, "rotation": 0, "ha": "right", "va": "center"}

        row_ims.append((diff_im, agree_im, diff_ticks))

    # ── Layout ────────────────────────────────────────────────────────────────
    cbar_h    = 0.007        # fixed colorbar height
    cbar_gap  = 0.010        # gap between panel bottom and its colorbar
    title_gap = 0.040        # gap between panel top and the cbar above it (room for title)

    total_fixed = (
        0.05                                    # bottom margin
        + 0.08                                  # top margin for suptitle
        + n_rows * (cbar_h + cbar_gap)          # colorbars + gaps below panels
        + (n_rows - 1) * title_gap              # title gaps between rows
    )
    panel_height = (1.0 - total_fixed) / n_rows

    fig.subplots_adjust(left=0.06, right=0.97, top=0.95, bottom=0.05, wspace=0.18)
    fig.canvas.draw()

    row_y_bottoms = {}
    cbar_ys       = {}
    y_cursor = 0.05

    for r in range(n_rows - 1, -1, -1):
        cbar_ys[r]       = y_cursor
        y_cursor        += cbar_h + cbar_gap
        row_y_bottoms[r] = y_cursor
        y_cursor        += panel_height
        if r > 0:
            y_cursor += title_gap              # space for the title of the row above

    for r in range(n_rows):
        for c in range(2):
            pos = axs_2d[r, c].get_position()
            axs_2d[r, c].set_position([pos.x0, row_y_bottoms[r], pos.width, panel_height])

    fig.canvas.draw()

    # ── Colorbars: diff per row, agreement shared at bottom ───────────────────
    for r, (diff_im, agree_im, diff_ticks) in enumerate(row_ims):
        pos_l = axs_2d[r, 0].get_position()

        # diff colorbar under col 0
        cax_diff = fig.add_axes([pos_l.x0, cbar_ys[r], pos_l.width, cbar_h])
        cbar_diff = fig.colorbar(diff_im, cax=cax_diff, orientation="horizontal",
                                 extend="both", ticks=diff_ticks)
        cbar_diff.ax.tick_params(labelsize=7, pad=2)
        for j, lbl in enumerate(cbar_diff.ax.xaxis.get_ticklabels()):
            lbl.set_visible(j % 2 == 0)
        cbar_diff.set_label("Change in frequency [per 6 h]",
                            fontsize=fontsize_cbar, labelpad=4)
        cbar_diff.ax.xaxis.set_label_position("bottom")

    # agreement colorbar under col 1, last row only
    pos_r = axs_2d[-1, 1].get_position()
    cax_agree = fig.add_axes([pos_r.x0, cbar_ys[n_rows - 1], pos_r.width, cbar_h])
    agree_cbar = fig.colorbar(row_ims[-1][1], cax=cax_agree, orientation="horizontal",
                              extend="neither", ticks=agree_ticks)
    agree_cbar.ax.tick_params(labelsize=7, pad=2)
    ticklabels = [str(int(t)) for t in agree_ticks]
    ticklabels[7] = "0*"
    agree_cbar.ax.set_xticklabels(ticklabels, fontsize=6)
    agree_cbar.set_label("Net model agreement [n models]",
                         fontsize=fontsize_cbar, labelpad=4)
    agree_cbar.ax.xaxis.set_label_position("bottom")

    from matplotlib.patches import Patch
    fig.legend(
        handles=[Patch(facecolor=unanimous_zero_color, label="0* = all models no change")],
        fontsize=6, framealpha=0.8,
        bbox_to_anchor=(axs_2d[-1, 1].get_position().x1,
                        cbar_ys[n_rows - 1] + cbar_h + 0.005),
        bbox_transform=fig.transFigure,
        loc="lower right",
    )



    top_of_panels = row_y_bottoms[0] + panel_height
    fig.suptitle(
        f"{turbulence_index} p99 · {P} hPa · {experiment.upper()} "
        f"{future_period[0]}–{future_period[1]} · {time_selection_label}",
        fontsize=fontsize_title, fontweight="normal",
        y=top_of_panels + 0.03,
    )

    if save_fig and outfile is not None:
        fig.savefig(outfile, dpi=300, bbox_inches="tight")
        print(f"Saved to: {outfile}")

    return fig, axs_2d




# SYNOPTIC ANALYSIS FUNCTIONS:

BARPA_MODELS = {
    "ACCESS-CM2":    "/g/data/py18/BARPA/output-CMIP6/DD/AUS-15/BOM/ACCESS-CM2/historical/r4i1p1f1/BARPA-R/v1-r1/day/psl/latest",
    "ACCESS-ESM1-5": "/g/data/py18/BARPA/output-CMIP6/DD/AUS-15/BOM/ACCESS-ESM1-5/historical/r6i1p1f1/BARPA-R/v1-r1/day/psl/latest",
    "CESM2":         "/g/data/py18/BARPA/output-CMIP6/DD/AUS-15/BOM/CESM2/historical/r11i1p1f1/BARPA-R/v1-r1/day/psl/latest",
    "CMCC-ESM2":     "/g/data/py18/BARPA/output-CMIP6/DD/AUS-15/BOM/CMCC-ESM2/historical/r1i1p1f1/BARPA-R/v1-r1/day/psl/latest",
    "EC-Earth3":     "/g/data/py18/BARPA/output-CMIP6/DD/AUS-15/BOM/EC-Earth3/historical/r1i1p1f1/BARPA-R/v1-r1/day/psl/latest",
    "MPI-ESM1-2-HR": "/g/data/py18/BARPA/output-CMIP6/DD/AUS-15/BOM/MPI-ESM1-2-HR/historical/r1i1p1f1/BARPA-R/v1-r1/day/psl/latest",
    "NorESM2-MM":    "/g/data/py18/BARPA/output-CMIP6/DD/AUS-15/BOM/NorESM2-MM/historical/r1i1p1f1/BARPA-R/v1-r1/day/psl/latest",
}

BARRA_ROOT      = "/g/data/cj37/BARRA/BARRA_R/v1/analysis/slv/av_mslp"
OVERLAP_PERIOD  = slice("1990", "2009")
from glob import glob as glob_files


def _load_barra_mslp(months=None, test_year=None):
    if months is None:
        month_strs = [f"{m:02d}" for m in range(1, 13)]
    else:
        month_strs = [f"{m:02d}" for m in months]

    years = [test_year] if test_year else range(1990, 2010)

    files = []
    for year in years:
        for m in month_strs:
            files.extend(glob_files(f"{BARRA_ROOT}/{year}/{m}/*.nc"))
    files.sort()

    if not files:
        raise FileNotFoundError(f"No BARRA-R files found for years={list(years)}, months={month_strs}")

    ds = xr.open_mfdataset(
        files,
        combine="nested",
        concat_dim="time",
        decode_timedelta=True,
        parallel=True,
    )
    da = ds["av_mslp"].rename({"latitude": "lat", "longitude": "lon"})

    if da.attrs.get("units", "") == "Pa" or float(da.isel(time=0).mean()) > 10000:
        da = da / 100.0
        da.attrs["units"] = "hPa"

    return da


def _load_barpa_mmm(months=None, test_year=None):
    model_means = []
    for model_name, path in BARPA_MODELS.items():
        files = sorted(glob_files(f"{path}/*.nc"))
        if not files:
            print(f"  Warning: no files for {model_name}")
            continue

        ds = xr.open_mfdataset(files, combine="by_coords", parallel=True)
        da = ds["psl"].sel(time=str(test_year) if test_year else OVERLAP_PERIOD)

        if months is not None:
            da = da.isel(time=da.time.dt.month.isin(months))

        if da.attrs.get("units", "") == "Pa" or float(da.isel(time=0).mean()) > 10000:
            da = da / 100.0
            da.attrs["units"] = "hPa"

        model_means.append(da.mean("time"))
        print(f"  {model_name} done")

    return xr.concat(model_means, dim="model").mean("model")


def prep_mslp_eval_vs_mmmhist(
    time_selection="May",
    compute_anomaly=False,
    use_cache=True,
    cache_dir="/scratch/v46/ls7238/cache",
    test_year=None,
):
    """
    Load and compute BARRA-R and BARPA MMM MSLP fields.
    Returns a dict with da_eval, da_hist, da_diff ready for plotting.

    test_year: int, e.g. 2000 — loads only that year for a quick check.
               Caching is disabled when test_year is set.
    """
    import os, pickle

    MONTH_MAP = {
        "January":1, "February":2, "March":3,    "April":4,
        "May":5,     "June":6,     "July":7,      "August":8,
        "September":9,"October":10,"November":11, "December":12,
    }
    SEASON_MAP = {
        "DJF":[12,1,2], "MAM":[3,4,5], "JJA":[6,7,8], "SON":[9,10,11],
        "MJJASO":[5,6,7,8,9,10], "NDJFMA":[11,12,1,2,3,4],
        "annual": list(range(1, 13)),
    }

    if isinstance(time_selection, str) and time_selection in MONTH_MAP:
        sel_months = [MONTH_MAP[time_selection]]
    elif isinstance(time_selection, str) and time_selection in SEASON_MAP:
        sel_months = SEASON_MAP[time_selection]
    elif isinstance(time_selection, list):
        sel_months = time_selection
    else:
        raise ValueError(f"Unrecognised time_selection: {time_selection!r}")

    # disable cache for test runs
    if test_year:
        use_cache = False

    anom_tag   = "anom" if compute_anomaly else "raw"
    cache_file = os.path.join(cache_dir, f"mslp_{time_selection}_{anom_tag}.pkl")

    if use_cache and os.path.exists(cache_file):
        print(f"Loading from cache: {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    # ── BARRA-R ───────────────────────────────────────────────────────────────
    print("Loading BARRA-R...")
    if compute_anomaly:
        da_barra_sel = _load_barra_mslp(months=sel_months, test_year=test_year)
        da_barra_ann = _load_barra_mslp(months=None,       test_year=test_year)
        da_eval = da_barra_sel.mean("time") - da_barra_ann.mean("time")
    else:
        da_eval = _load_barra_mslp(months=sel_months, test_year=test_year).mean("time")

    # ── BARPA MMM ─────────────────────────────────────────────────────────────
    print("Loading BARPA MMM...")
    if compute_anomaly:
        model_anoms = []
        for model_name, path in BARPA_MODELS.items():
            files = sorted(glob_files(f"{path}/*.nc"))
            if not files:
                print(f"  Warning: no files for {model_name}")
                continue
            ds  = xr.open_mfdataset(files, combine="by_coords", parallel=True)
            da  = ds["psl"].sel(time=str(test_year) if test_year else OVERLAP_PERIOD)
            if float(da.isel(time=0).mean()) > 10000:
                da = da / 100.0
            da_sel = da.isel(time=da.time.dt.month.isin(sel_months)).mean("time")
            da_ann = da.mean("time")
            model_anoms.append(da_sel - da_ann)
            print(f"  {model_name} done")
        da_hist = xr.concat(model_anoms, dim="model").mean("model")
    else:
        da_hist = _load_barpa_mmm(months=sel_months, test_year=test_year)

    
    da_hist = da_hist.interp(lat=da_eval.lat, lon=da_eval.lon, method="linear")
    da_diff = da_hist - da_eval

    from dask.diagnostics import ProgressBar

    print("Computing arrays...")
    with ProgressBar():
        da_eval = da_eval.compute()
        da_hist = da_hist.compute()
        da_diff = da_diff.compute()



    result = {
        "da_eval":         da_eval,
        "da_hist":         da_hist,
        "da_diff":         da_diff,
        "time_selection":  time_selection,
        "compute_anomaly": compute_anomaly,
    }

    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump(result, f)
        print(f"Cached to: {cache_file}")

    return result

def _muted_cmap(cmap_in, strength=0.35):
    """Blend a colormap toward white to create a muted/washed-out version."""
    rgba = cmap_in(np.linspace(0, 1, 256))
    rgba[:, :3] = rgba[:, :3] * strength + (1 - strength)
    return LinearSegmentedColormap.from_list("muted", rgba)


def plot_mslp_eval_vs_mmmhist_diff(
    data,
    figsize=None,
    ticks_abs=None,
    ticks_diff=None,
    contour_interval=4,    # hPa interval for labeled contour lines on base panels
    outfile=None,
    save_fig=False,
    single_label = False,
    year = '',
    show_contours = True
):
    """
    Plot BARRA-R | BARPA MMM | MMM - BARRA-R for MSLP.
    Left/centre: muted filled contours + labeled contour lines.
    Right: full diverging pcolormesh.
    Expects the dict returned by prep_mslp_eval_vs_mmmhist.
    """
    da_eval         = data["da_eval"]
    da_hist         = data["da_hist"]
    da_diff         = data["da_diff"]
    time_selection  = data["time_selection"]
    compute_anomaly = data["compute_anomaly"]
    test_year       = data.get("test_year")
    var_name = data.get("var_name", "MSLP")
    units    = data.get("units",    "hPa")


    # ── Colormaps ─────────────────────────────────────────────────────────────
    base_cmap_full = plt.get_cmap("RdBu_r").copy()
    base_cmap_full.set_bad("lightgrey")
    base_cmap_muted = _muted_cmap(base_cmap_full, strength=0.35)
    base_cmap_muted.set_bad("lightgrey")

    diff_cmap = cmap_dict["anom"].copy()
    diff_cmap.set_bad("lightgrey")

    # ── Ticks ─────────────────────────────────────────────────────────────────
    def _sym_ticks(maxval, n=14):
        step = maxval / (n // 2)
        return np.arange(-maxval, maxval + step * 0.001, step)

    vmin = float(np.nanmin([da_eval.min(), da_hist.min()]))
    vmax = float(np.nanmax([da_eval.max(), da_hist.max()]))
    if ticks_abs:
        vmin, vmax = ticks_abs
    base_ticks = np.linspace(vmin, vmax, 15)

    diff_max   = ticks_diff if ticks_diff else float(np.nanmax(np.abs(da_diff.values)))
    diff_ticks = _sym_ticks(diff_max)

    base_norm = BoundaryNorm(base_ticks, base_cmap_muted.N + 1, extend="both")
    diff_norm = BoundaryNorm(diff_ticks, diff_cmap.N + 1,       extend="both")

    # Labeled contour line levels — rounded to nearest contour_interval
    c_min = np.ceil(vmin  / contour_interval) * contour_interval
    c_max = np.floor(vmax / contour_interval) * contour_interval
    contour_levels = np.arange(c_min, c_max + 0.1, contour_interval)

    # ── Figure ────────────────────────────────────────────────────────────────
    proj = ccrs.PlateCarree(130)
    if figsize is None:
        figsize = (15, 4)

    fig, axs = plt.subplots(
        1, 3, figsize=figsize,
        subplot_kw={"projection": proj, "frame_on": True},
    )

    anom_str     = " anomaly" if compute_anomaly else ""
    panel_data   = [da_eval, da_hist, da_diff]
    panel_titles = [
        f"BARRA-R: {time_selection}{anom_str}",
        f"BARPA MMM: {time_selection}{anom_str}",
        f"MMM − BARRA-R: {time_selection}",
    ]

    xticks = [100, 120, 140, 160, 180]
    yticks = [-50, -40, -30, -20, -10, 0, 10]

    base_im = diff_im = None

    for col, (ax, da, title) in enumerate(zip(axs, panel_data, panel_titles)):
        ax.set_extent([90, 195, -53.58, 13.63], crs=ccrs.PlateCarree())

        if col < 2:
            # Muted filled contours
            im = ax.contourf(
                da.lon, da.lat, da,
                levels=base_ticks,
                cmap=base_cmap_muted,
                norm=base_norm,
                transform=ccrs.PlateCarree(),
                zorder=2,
                extend="both",
            )
            base_im = im
            
            if show_contours:
                # Labeled contour lines on top
                cs = ax.contour(
                    da.lon, da.lat, da,
                    levels=contour_levels,
                    colors="black",
                    linewidths=0.5,
                    transform=ccrs.PlateCarree(),
                    zorder=3,
                )
                ax.clabel(cs, fmt="%g", fontsize=6, inline=True)

        else:
            # Diff panel: full diverging pcolormesh
            im = ax.pcolormesh(
                da.lon, da.lat, da,
                cmap=diff_cmap, norm=diff_norm,
                transform=ccrs.PlateCarree(), zorder=2,
            )
            diff_im = im

        ax.coastlines(resolution="10m", linewidth=0.25, zorder=5)
        try:
            ax.add_feature(cfeature.BORDERS, linewidth=0.2, zorder=5)
        except Exception:
            pass

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("#aaaaaa")
            spine.set_linewidth(0.8)

        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            linewidth=0.4, color="black", alpha=0.20,
            linestyle="--", draw_labels=True,
        )
        gl.rotate_labels  = False
        gl.xlocator       = mticker.FixedLocator(xticks)
        gl.ylocator       = mticker.FixedLocator(yticks)
        gl.xformatter     = LONGITUDE_FORMATTER
        gl.yformatter     = LATITUDE_FORMATTER
        gl.left_labels    = (col == 0)
        gl.right_labels   = False
        gl.top_labels     = False
        gl.bottom_labels  = True
        gl.xlabel_style   = {"fontsize": 7, "rotation": 0, "ha": "center"}
        gl.ylabel_style   = {"fontsize": 7, "rotation": 0, "ha": "right", "va": "center"}

        ax.set_title(title, loc="left", fontsize=fontsize_subtitle,
                     fontweight="normal", pad=2)

    # ── Colorbars ─────────────────────────────────────────────────────────────
    fig.subplots_adjust(left=0.06, right=0.97, top=0.92, bottom=0.18, wspace=0.18)
    fig.canvas.draw()

    def _add_cbar(im, ticks, ax_l, ax_r, label):
        pos_l = ax_l.get_position()
        pos_r = ax_r.get_position()
        cax   = fig.add_axes([pos_l.x0, 0.06, pos_r.x1 - pos_l.x0, 0.025])
        cbar  = fig.colorbar(im, cax=cax, orientation="horizontal",
                             extend="both", ticks=ticks)
        cbar.ax.tick_params(labelsize=7, pad=2)
        for j, lbl in enumerate(cbar.ax.xaxis.get_ticklabels()):
            lbl.set_visible(j % 2 == 0)
        cbar.set_label(label, fontsize=fontsize_cbar, labelpad=4)
        cbar.ax.xaxis.set_label_position("bottom")

    _add_cbar(base_im, base_ticks, axs[0], axs[1], f"{var_name}{anom_str} [{units}]")
    _add_cbar(diff_im, diff_ticks, axs[2], axs[2], f"Difference [{units}]")

    if single_label:
        period_str = year
    else:
        period_str = "1990-2009"

    fig.suptitle(
        f"{var_name}{anom_str} | {time_selection} | {period_str}",
        fontsize=fontsize_title, fontweight="normal", y=0.97,
    )

    if save_fig and outfile is not None:
        fig.savefig(outfile, dpi=300, bbox_inches="tight")
        print(f"Saved: {outfile}")

    return fig, axs


    ## Single model evaluation:

def prep_mslp_per_model(
    time_selection="May",
    compute_anomaly=False,
    use_cache=True,
    cache_dir="/scratch/v46/ls7238/cache",
    test_year=None,
):
    """
    Like prep_mslp_eval_vs_mmmhist but returns one diff per model
    instead of the MMM. Returns dict with da_eval and da_diffs.
    """
    import os, pickle
    from dask.diagnostics import ProgressBar

    MONTH_MAP = {
        "January":1,"February":2,"March":3,"April":4,"May":5,"June":6,
        "July":7,"August":8,"September":9,"October":10,"November":11,"December":12,
    }
    SEASON_MAP = {
        "DJF":[12,1,2],"MAM":[3,4,5],"JJA":[6,7,8],"SON":[9,10,11],
        "MJJASO":[5,6,7,8,9,10],"NDJFMA":[11,12,1,2,3,4],
        "annual":list(range(1,13)),
    }

    if isinstance(time_selection, str) and time_selection in MONTH_MAP:
        sel_months = [MONTH_MAP[time_selection]]
    elif isinstance(time_selection, str) and time_selection in SEASON_MAP:
        sel_months = SEASON_MAP[time_selection]
    elif isinstance(time_selection, list):
        sel_months = time_selection
    else:
        raise ValueError(f"Unrecognised time_selection: {time_selection!r}")

    if test_year:
        use_cache = False

    anom_tag   = "anom" if compute_anomaly else "raw"
    cache_file = os.path.join(cache_dir, f"mslp_permodel_{time_selection}_{anom_tag}.pkl")

    if use_cache and os.path.exists(cache_file):
        print(f"Loading from cache: {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    # ── BARRA-R ───────────────────────────────────────────────────────────────
    print("Loading BARRA-R...")
    if compute_anomaly:
        da_barra_sel = _load_barra_mslp(months=sel_months, test_year=test_year)
        da_barra_ann = _load_barra_mslp(months=None,       test_year=test_year)
        da_eval = (da_barra_sel.mean("time") - da_barra_ann.mean("time")).compute()
    else:
        with ProgressBar():
            da_eval = _load_barra_mslp(months=sel_months, test_year=test_year).mean("time").compute()

    # ── Per-model diffs ───────────────────────────────────────────────────────
    da_diffs = {}
    for model_name, path in BARPA_MODELS.items():
        print(f"  Loading {model_name}...")
        files = sorted(glob_files(f"{path}/*.nc"))
        if not files:
            print(f"    Warning: no files found")
            continue

        ds = xr.open_mfdataset(files, combine="by_coords", parallel=False)
        da = ds["psl"].sel(time=str(test_year) if test_year else OVERLAP_PERIOD)
        da = da.isel(time=da.time.dt.month.isin(sel_months))

        if da.attrs.get("units","") == "Pa" or float(da.isel(time=0).mean()) > 10000:
            da = da / 100.0

        if compute_anomaly:
            da_ann = ds["psl"].sel(time=str(test_year) if test_year else OVERLAP_PERIOD)
            if float(da_ann.isel(time=0).mean()) > 10000:
                da_ann = da_ann / 100.0
            da_model = da.mean("time") - da_ann.mean("time")
        else:
            da_model = da.mean("time")

        da_model = da_model.interp(lat=da_eval.lat, lon=da_eval.lon, method="linear")
        with ProgressBar():
            da_diffs[model_name] = (da_model - da_eval).compute()
        ds.close()
        print(f"    done")

    result = {
        "da_eval":         da_eval,
        "da_diffs":        da_diffs,       # dict: model_name → diff DataArray
        "time_selection":  time_selection,
        "compute_anomaly": compute_anomaly,
        "test_year":       test_year,
    }

    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump(result, f)
        print(f"Cached to: {cache_file}")

    return result


def plot_mslp_per_model_diff(
    data,
    figsize=None,
    ticks_diff=None,
    outfile=None,
    save_fig=False,
):
    """
    7-panel plot of MSLP difference (model − BARRA-R), one panel per BARPA model.
    Layout: 4×2 grid, last cell empty.
    """
    da_diffs       = data["da_diffs"]
    time_selection = data["time_selection"]
    compute_anomaly = data["compute_anomaly"]
    test_year      = data.get("test_year")
    var_name = data.get("var_name", "MSLP")
    units    = data.get("units",    "hPa")

    model_names = list(da_diffs.keys())   # 7 models
    n_models    = len(model_names)
    ncols, nrows = 2, 4                   # 4×2 = 8 slots, last empty

    # ── Colormap ──────────────────────────────────────────────────────────────
    diff_cmap = cmap_dict["anom"].copy()
    diff_cmap.set_bad("lightgrey")

    # ── Ticks ─────────────────────────────────────────────────────────────────
    def _sym_ticks(maxval, n=14):
        step = maxval / (n // 2)
        return np.arange(-maxval, maxval + step * 0.001, step)

    if ticks_diff is None:
        diff_max = float(np.nanmax([
            np.nanmax(np.abs(da_diffs[m].values)) for m in model_names
        ]))
    else:
        diff_max = float(ticks_diff)

    diff_ticks = _sym_ticks(diff_max)
    diff_norm  = BoundaryNorm(diff_ticks, diff_cmap.N + 1, extend="both")

    # ── Figure ────────────────────────────────────────────────────────────────
    proj = ccrs.PlateCarree(130)
    if figsize is None:
        figsize = (12, 4 * nrows)

    fig, axs_2d = plt.subplots(
        nrows, ncols, figsize=figsize,
        subplot_kw={"projection": proj, "frame_on": True},
        squeeze=False,
    )

    xticks = [100, 120, 140, 160, 180]
    yticks = [-50, -40, -30, -20, -10, 0, 10]

    last_im = None

    for idx, model_name in enumerate(model_names):
        row, col = divmod(idx, ncols)
        ax = axs_2d[row, col]
        da = da_diffs[model_name]

        ax.set_extent([90, 195, -53.58, 13.63], crs=ccrs.PlateCarree())

        im = ax.pcolormesh(
            da.lon, da.lat, da,
            cmap=diff_cmap, norm=diff_norm,
            transform=ccrs.PlateCarree(), zorder=2,
        )
        last_im = im

        ax.coastlines(resolution="10m", linewidth=0.25, zorder=5)
        try:
            ax.add_feature(cfeature.BORDERS, linewidth=0.2, zorder=5)
        except Exception:
            pass

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("#aaaaaa")
            spine.set_linewidth(0.8)

        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            linewidth=0.4, color="black", alpha=0.20,
            linestyle="--", draw_labels=True,
        )
        gl.rotate_labels  = False
        gl.xlocator       = mticker.FixedLocator(xticks)
        gl.ylocator       = mticker.FixedLocator(yticks)
        gl.xformatter     = LONGITUDE_FORMATTER
        gl.yformatter     = LATITUDE_FORMATTER
        gl.left_labels    = (col == 0)
        gl.right_labels   = False
        gl.top_labels     = False
        gl.bottom_labels  = (row == nrows - 1)
        gl.xlabel_style   = {"fontsize": 7, "rotation": 0, "ha": "center"}
        gl.ylabel_style   = {"fontsize": 7, "rotation": 0, "ha": "right", "va": "center"}

        ax.set_title(f"{model_name} − BARRA-R: {time_selection}",
                     loc="left", fontsize=fontsize_subtitle,
                     fontweight="normal", pad=2)

    # Hide the unused 8th panel
    axs_2d[-1, -1].set_visible(False)

    # ── Colorbar ──────────────────────────────────────────────────────────────
    fig.subplots_adjust(left=0.06, right=0.97, top=0.96, bottom=0.06, wspace=0.15, hspace=0.15)
    fig.canvas.draw()

    # Span full width of the bottom-left panel (last visible panel)
    pos_last = axs_2d[nrows - 1, 0].get_position()
    pos_br   = axs_2d[nrows - 2, ncols - 1].get_position()  # last row with content on right

    cax = fig.add_axes([pos_last.x0, pos_last.y0 - 0.03, pos_br.x1 - pos_last.x0, 0.015])
    cbar = fig.colorbar(last_im, cax=cax, orientation="horizontal",
                        extend="both", ticks=diff_ticks)
    cbar.ax.tick_params(labelsize=7, pad=2)
    for j, lbl in enumerate(cbar.ax.xaxis.get_ticklabels()):
        lbl.set_visible(j % 2 == 0)
    cbar.set_label(f"Difference [{units}]", fontsize=fontsize_cbar, labelpad=4)
    cbar.ax.xaxis.set_label_position("bottom")

    anom_str   = " anomaly" if compute_anomaly else ""
    period_str = str(test_year) if test_year else "1990–2009"
    fig.suptitle(f"{var_name}{anom_str} per model − BARRA-R · {time_selection} · {period_str}",
             fontsize=fontsize_title, fontweight="normal", y=0.99)


    if save_fig and outfile is not None:
        fig.savefig(outfile, dpi=300, bbox_inches="tight")
        print(f"Saved: {outfile}")

    return fig, axs_2d


# 500hPa GEOPOTENTIAL HEIGHT

BARRA_Z500_ROOT = "/g/data/cj37/BARRA/BARRA_R/v1/analysis/prs/geop_ht"

BARPA_ZG500_MODELS = {
    "ACCESS-CM2":    "/g/data/py18/BARPA/output-CMIP6/DD/AUS-15/BOM/ACCESS-CM2/historical/r4i1p1f1/BARPA-R/v1-r1/day/zg500/latest",
    "ACCESS-ESM1-5": "/g/data/py18/BARPA/output-CMIP6/DD/AUS-15/BOM/ACCESS-ESM1-5/historical/r6i1p1f1/BARPA-R/v1-r1/day/zg500/latest",
    "CESM2":         "/g/data/py18/BARPA/output-CMIP6/DD/AUS-15/BOM/CESM2/historical/r11i1p1f1/BARPA-R/v1-r1/day/zg500/latest",
    "CMCC-ESM2":     "/g/data/py18/BARPA/output-CMIP6/DD/AUS-15/BOM/CMCC-ESM2/historical/r1i1p1f1/BARPA-R/v1-r1/day/zg500/latest",
    "EC-Earth3":     "/g/data/py18/BARPA/output-CMIP6/DD/AUS-15/BOM/EC-Earth3/historical/r1i1p1f1/BARPA-R/v1-r1/day/zg500/latest",
    "MPI-ESM1-2-HR": "/g/data/py18/BARPA/output-CMIP6/DD/AUS-15/BOM/MPI-ESM1-2-HR/historical/r1i1p1f1/BARPA-R/v1-r1/day/zg500/latest",
    "NorESM2-MM":    "/g/data/py18/BARPA/output-CMIP6/DD/AUS-15/BOM/NorESM2-MM/historical/r1i1p1f1/BARPA-R/v1-r1/day/zg500/latest",
}


def _load_barra_z500(months=None, test_year=None):
    if months is None:
        month_strs = [f"{m:02d}" for m in range(1, 13)]
    else:
        month_strs = [f"{m:02d}" for m in months]

    years = [test_year] if test_year else range(1990, 2010)

    files = []
    for year in years:
        for m in month_strs:
            files.extend(glob_files(f"{BARRA_Z500_ROOT}/{year}/{m}/*.nc"))
    files.sort()

    if not files:
        raise FileNotFoundError(f"No BARRA-R Z500 files found")

    ds = xr.open_mfdataset(
        files, combine="nested", concat_dim="time",
        decode_timedelta=True, parallel=False,
    )
    da = (
        ds["geop_ht"]
        .sel(pressure=500, method="nearest")
        .rename({"latitude": "lat", "longitude": "lon"})
    )
    return da   # units: metres


def _load_barpa_z500_mmm(months=None, test_year=None, barpa_var="zg500"):
    model_means = []
    for model_name, path in BARPA_ZG500_MODELS.items():
        files = sorted(glob_files(f"{path}/*.nc"))
        if not files:
            print(f"  Warning: no files for {model_name}")
            continue

        ds = xr.open_mfdataset(files, combine="by_coords", parallel=False)
        da = ds[barpa_var].sel(time=str(test_year) if test_year else OVERLAP_PERIOD)

        if months is not None:
            da = da.isel(time=da.time.dt.month.isin(months))

        model_means.append(da.mean("time").compute())
        ds.close()
        print(f"  {model_name} done")

    return xr.concat(model_means, dim="model").mean("model")


def prep_z500_eval_vs_mmmhist(
    time_selection="May",
    use_cache=True,
    cache_dir="/scratch/v46/ls7238/cache",
    test_year=None,
    barpa_var="zg500",
):
    """
    Load and compute BARRA-R and BARPA MMM Z500 fields.
    Returns dict with da_eval, da_hist, da_diff ready for plotting.
    """
    import os, pickle
    from dask.diagnostics import ProgressBar

    MONTH_MAP = {
        "January":1,"February":2,"March":3,"April":4,"May":5,"June":6,
        "July":7,"August":8,"September":9,"October":10,"November":11,"December":12,
    }
    SEASON_MAP = {
        "DJF":[12,1,2],"MAM":[3,4,5],"JJA":[6,7,8],"SON":[9,10,11],
        "MJJASO":[5,6,7,8,9,10],"NDJFMA":[11,12,1,2,3,4],
        "annual":list(range(1,13)),
    }

    if isinstance(time_selection, str) and time_selection in MONTH_MAP:
        sel_months = [MONTH_MAP[time_selection]]
    elif isinstance(time_selection, str) and time_selection in SEASON_MAP:
        sel_months = SEASON_MAP[time_selection]
    elif isinstance(time_selection, list):
        sel_months = time_selection
    else:
        raise ValueError(f"Unrecognised time_selection: {time_selection!r}")

    if test_year:
        use_cache = False

    cache_file = os.path.join(cache_dir, f"z500_mmm_{time_selection}.pkl")

    if use_cache and os.path.exists(cache_file):
        print(f"Loading from cache: {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print("Loading BARRA-R Z500...")
    with ProgressBar():
        da_eval = _load_barra_z500(months=sel_months, test_year=test_year).mean("time").compute()

    print("Loading BARPA MMM Z500...")
    da_hist = _load_barpa_z500_mmm(months=sel_months, test_year=test_year, barpa_var=barpa_var)
    da_hist = da_hist.interp(lat=da_eval.lat, lon=da_eval.lon, method="linear")

    with ProgressBar():
        da_hist = da_hist.compute()

    da_diff = da_hist - da_eval

    result = {
        "da_eval":        da_eval,
        "da_hist":        da_hist,
        "da_diff":        da_diff,
        "time_selection": time_selection,
        "compute_anomaly": False,
        "test_year":      test_year,
        "var_name":       "Z500",
        "units":          "m",
    }

    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump(result, f)
        print(f"Cached to: {cache_file}")

    return result


def prep_z500_per_model(
    time_selection="May",
    use_cache=True,
    cache_dir="/scratch/v46/ls7238/cache",
    test_year=None,
    barpa_var="zg500",
):
    """
    Load BARRA-R and each BARPA model Z500 separately.
    Returns dict with da_eval and da_diffs (per-model).
    """
    import os, pickle
    from dask.diagnostics import ProgressBar

    MONTH_MAP = {
        "January":1,"February":2,"March":3,"April":4,"May":5,"June":6,
        "July":7,"August":8,"September":9,"October":10,"November":11,"December":12,
    }
    SEASON_MAP = {
        "DJF":[12,1,2],"MAM":[3,4,5],"JJA":[6,7,8],"SON":[9,10,11],
        "MJJASO":[5,6,7,8,9,10],"NDJFMA":[11,12,1,2,3,4],
        "annual":list(range(1,13)),
    }

    if isinstance(time_selection, str) and time_selection in MONTH_MAP:
        sel_months = [MONTH_MAP[time_selection]]
    elif isinstance(time_selection, str) and time_selection in SEASON_MAP:
        sel_months = SEASON_MAP[time_selection]
    elif isinstance(time_selection, list):
        sel_months = time_selection
    else:
        raise ValueError(f"Unrecognised time_selection: {time_selection!r}")

    if test_year:
        use_cache = False

    cache_file = os.path.join(cache_dir, f"z500_permodel_{time_selection}.pkl")

    if use_cache and os.path.exists(cache_file):
        print(f"Loading from cache: {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print("Loading BARRA-R Z500...")
    with ProgressBar():
        da_eval = _load_barra_z500(months=sel_months, test_year=test_year).mean("time").compute()

    da_diffs = {}
    for model_name, path in BARPA_ZG500_MODELS.items():
        print(f"  Loading {model_name}...")
        files = sorted(glob_files(f"{path}/*.nc"))
        if not files:
            print(f"    Warning: no files found")
            continue

        ds = xr.open_mfdataset(files, combine="by_coords", parallel=False)
        da = ds[barpa_var].sel(time=str(test_year) if test_year else OVERLAP_PERIOD)
        da = da.isel(time=da.time.dt.month.isin(sel_months))
        da_model = da.mean("time").interp(lat=da_eval.lat, lon=da_eval.lon, method="linear")

        with ProgressBar():
            da_diffs[model_name] = (da_model - da_eval).compute()
        ds.close()
        print(f"    done")

    result = {
        "da_eval":        da_eval,
        "da_diffs":       da_diffs,
        "time_selection": time_selection,
        "compute_anomaly": False,
        "test_year":      test_year,
        "var_name":       "Z500",
        "units":          "m",
    }

    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump(result, f)
        print(f"Cached to: {cache_file}")

    return result



# Surface temperature:

BARRA_TEMP_ROOT = "/g/data/cj37/BARRA/BARRA_R/v1/analysis/slv/av_temp_scrn"

BARPA_TAS_MODELS = {
    "ACCESS-CM2":    "/g/data/py18/BARPA/output-CMIP6/DD/AUS-15/BOM/ACCESS-CM2/historical/r4i1p1f1/BARPA-R/v1-r1/day/tas/latest",
    "ACCESS-ESM1-5": "/g/data/py18/BARPA/output-CMIP6/DD/AUS-15/BOM/ACCESS-ESM1-5/historical/r6i1p1f1/BARPA-R/v1-r1/day/tas/latest",
    "CESM2":         "/g/data/py18/BARPA/output-CMIP6/DD/AUS-15/BOM/CESM2/historical/r11i1p1f1/BARPA-R/v1-r1/day/tas/latest",
    "CMCC-ESM2":     "/g/data/py18/BARPA/output-CMIP6/DD/AUS-15/BOM/CMCC-ESM2/historical/r1i1p1f1/BARPA-R/v1-r1/day/tas/latest",
    "EC-Earth3":     "/g/data/py18/BARPA/output-CMIP6/DD/AUS-15/BOM/EC-Earth3/historical/r1i1p1f1/BARPA-R/v1-r1/day/tas/latest",
    "MPI-ESM1-2-HR": "/g/data/py18/BARPA/output-CMIP6/DD/AUS-15/BOM/MPI-ESM1-2-HR/historical/r1i1p1f1/BARPA-R/v1-r1/day/tas/latest",
    "NorESM2-MM":    "/g/data/py18/BARPA/output-CMIP6/DD/AUS-15/BOM/NorESM2-MM/historical/r1i1p1f1/BARPA-R/v1-r1/day/tas/latest",
}


def _to_celsius(da):
    """Convert to Celsius if in Kelvin."""
    if float(da.isel(time=0).mean()) > 200:
        da = da - 273.15
        da.attrs["units"] = "°C"
    return da


def _load_barra_temp(months=None, test_year=None):
    if months is None:
        month_strs = [f"{m:02d}" for m in range(1, 13)]
    else:
        month_strs = [f"{m:02d}" for m in months]

    years = [test_year] if test_year else range(1990, 2010)

    files = []
    for year in years:
        for m in month_strs:
            files.extend(glob_files(f"{BARRA_TEMP_ROOT}/{year}/{m}/*.nc"))
    files.sort()

    if not files:
        raise FileNotFoundError("No BARRA-R temperature files found")

    ds = xr.open_mfdataset(
        files, combine="nested", concat_dim="time",
        decode_timedelta=True, parallel=False,
    )
    da = (
        ds["av_temp_scrn"]
        .rename({"latitude": "lat", "longitude": "lon"})
    )
    return _to_celsius(da)


def _load_barpa_tas_mmm(months=None, test_year=None):
    model_means = []
    for model_name, path in BARPA_TAS_MODELS.items():
        files = sorted(glob_files(f"{path}/*.nc"))
        if not files:
            print(f"  Warning: no files for {model_name}")
            continue

        ds = xr.open_mfdataset(files, combine="by_coords", parallel=False)
        da = ds["tas"].sel(time=str(test_year) if test_year else OVERLAP_PERIOD)

        if months is not None:
            da = da.isel(time=da.time.dt.month.isin(months))

        da = _to_celsius(da)
        model_means.append(da.mean("time").compute())
        ds.close()
        print(f"  {model_name} done")

    return xr.concat(model_means, dim="model").mean("model")


def prep_temp_eval_vs_mmmhist(
    time_selection="May",
    use_cache=True,
    cache_dir="/scratch/v46/ls7238/cache",
    test_year=None,
):
    import os, pickle
    from dask.diagnostics import ProgressBar

    MONTH_MAP = {
        "January":1,"February":2,"March":3,"April":4,"May":5,"June":6,
        "July":7,"August":8,"September":9,"October":10,"November":11,"December":12,
    }
    SEASON_MAP = {
        "DJF":[12,1,2],"MAM":[3,4,5],"JJA":[6,7,8],"SON":[9,10,11],
        "MJJASO":[5,6,7,8,9,10],"NDJFMA":[11,12,1,2,3,4],
        "annual":list(range(1,13)),
    }

    if isinstance(time_selection, str) and time_selection in MONTH_MAP:
        sel_months = [MONTH_MAP[time_selection]]
    elif isinstance(time_selection, str) and time_selection in SEASON_MAP:
        sel_months = SEASON_MAP[time_selection]
    elif isinstance(time_selection, list):
        sel_months = time_selection
    else:
        raise ValueError(f"Unrecognised time_selection: {time_selection!r}")

    if test_year:
        use_cache = False

    cache_file = os.path.join(cache_dir, f"temp_mmm_{time_selection}.pkl")

    if use_cache and os.path.exists(cache_file):
        print(f"Loading from cache: {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print("Loading BARRA-R temperature...")
    with ProgressBar():
        da_eval = _load_barra_temp(months=sel_months, test_year=test_year).mean("time").compute()

    print("Loading BARPA MMM temperature...")
    da_hist = _load_barpa_tas_mmm(months=sel_months, test_year=test_year)
    da_hist = da_hist.interp(lat=da_eval.lat, lon=da_eval.lon, method="linear")

    with ProgressBar():
        da_hist = da_hist.compute()

    da_diff = da_hist - da_eval

    result = {
        "da_eval":         da_eval,
        "da_hist":         da_hist,
        "da_diff":         da_diff,
        "time_selection":  time_selection,
        "compute_anomaly": False,
        "test_year":       test_year,
        "var_name":        "T2m",
        "units":           "°C",
    }

    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump(result, f)
        print(f"Cached to: {cache_file}")

    return result


def prep_temp_per_model(
    time_selection="May",
    use_cache=True,
    cache_dir="/scratch/v46/ls7238/cache",
    test_year=None,
):
    import os, pickle
    from dask.diagnostics import ProgressBar

    MONTH_MAP = {
        "January":1,"February":2,"March":3,"April":4,"May":5,"June":6,
        "July":7,"August":8,"September":9,"October":10,"November":11,"December":12,
    }
    SEASON_MAP = {
        "DJF":[12,1,2],"MAM":[3,4,5],"JJA":[6,7,8],"SON":[9,10,11],
        "MJJASO":[5,6,7,8,9,10],"NDJFMA":[11,12,1,2,3,4],
        "annual":list(range(1,13)),
    }

    if isinstance(time_selection, str) and time_selection in MONTH_MAP:
        sel_months = [MONTH_MAP[time_selection]]
    elif isinstance(time_selection, str) and time_selection in SEASON_MAP:
        sel_months = SEASON_MAP[time_selection]
    elif isinstance(time_selection, list):
        sel_months = time_selection
    else:
        raise ValueError(f"Unrecognised time_selection: {time_selection!r}")

    if test_year:
        use_cache = False

    cache_file = os.path.join(cache_dir, f"temp_permodel_{time_selection}.pkl")

    if use_cache and os.path.exists(cache_file):
        print(f"Loading from cache: {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print("Loading BARRA-R temperature...")
    with ProgressBar():
        da_eval = _load_barra_temp(months=sel_months, test_year=test_year).mean("time").compute()

    da_diffs = {}
    for model_name, path in BARPA_TAS_MODELS.items():
        print(f"  Loading {model_name}...")
        files = sorted(glob_files(f"{path}/*.nc"))
        if not files:
            print(f"    Warning: no files found")
            continue

        ds = xr.open_mfdataset(files, combine="by_coords", parallel=False)
        da = ds["tas"].sel(time=str(test_year) if test_year else OVERLAP_PERIOD)
        da = da.isel(time=da.time.dt.month.isin(sel_months))
        da = _to_celsius(da)

        da_model = da.mean("time").interp(lat=da_eval.lat, lon=da_eval.lon, method="linear")

        with ProgressBar():
            da_diffs[model_name] = (da_model - da_eval).compute()
        ds.close()
        print(f"    done")

    result = {
        "da_eval":         da_eval,
        "da_diffs":        da_diffs,
        "time_selection":  time_selection,
        "compute_anomaly": False,
        "test_year":       test_year,
        "var_name":        "T2m",
        "units":           "°C",
    }

    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump(result, f)
        print(f"Cached to: {cache_file}")

    return result

BARRA_SFC_TEMP_ROOT = "/g/data/cj37/BARRA/BARRA_R/v1/analysis/spec/sfc_temp"

BARPA_TS_MODELS = {
    "ACCESS-CM2":    "/g/data/py18/BARPA/output-CMIP6/DD/AUS-15/BOM/ACCESS-CM2/historical/r4i1p1f1/BARPA-R/v1-r1/day/ts/latest",
    "ACCESS-ESM1-5": "/g/data/py18/BARPA/output-CMIP6/DD/AUS-15/BOM/ACCESS-ESM1-5/historical/r6i1p1f1/BARPA-R/v1-r1/day/ts/latest",
    "CESM2":         "/g/data/py18/BARPA/output-CMIP6/DD/AUS-15/BOM/CESM2/historical/r11i1p1f1/BARPA-R/v1-r1/day/ts/latest",
    "CMCC-ESM2":     "/g/data/py18/BARPA/output-CMIP6/DD/AUS-15/BOM/CMCC-ESM2/historical/r1i1p1f1/BARPA-R/v1-r1/day/ts/latest",
    "EC-Earth3":     "/g/data/py18/BARPA/output-CMIP6/DD/AUS-15/BOM/EC-Earth3/historical/r1i1p1f1/BARPA-R/v1-r1/day/ts/latest",
    "MPI-ESM1-2-HR": "/g/data/py18/BARPA/output-CMIP6/DD/AUS-15/BOM/MPI-ESM1-2-HR/historical/r1i1p1f1/BARPA-R/v1-r1/day/ts/latest",
    "NorESM2-MM":    "/g/data/py18/BARPA/output-CMIP6/DD/AUS-15/BOM/NorESM2-MM/historical/r1i1p1f1/BARPA-R/v1-r1/day/ts/latest",
}


def _load_barra_sfc_temp(months=None, test_year=None):
    if months is None:
        month_strs = [f"{m:02d}" for m in range(1, 13)]
    else:
        month_strs = [f"{m:02d}" for m in months]

    years = [test_year] if test_year else range(1990, 2010)

    files = []
    for year in years:
        for m in month_strs:
            files.extend(glob_files(f"{BARRA_SFC_TEMP_ROOT}/{year}/{m}/*.nc"))
    files.sort()

    if not files:
        raise FileNotFoundError("No BARRA-R sfc_temp files found")

    ds = xr.open_mfdataset(
        files, combine="nested", concat_dim="time",
        decode_timedelta=True, parallel=False,
    )
    # update variable name if ncdump shows something different
    da = (
        ds["sfc_temp"]
        .rename({"latitude": "lat", "longitude": "lon"})
    )
    return _to_celsius(da)


def _load_barpa_ts_mmm(months=None, test_year=None):
    model_means = []
    for model_name, path in BARPA_TS_MODELS.items():
        files = sorted(glob_files(f"{path}/*.nc"))
        if not files:
            print(f"  Warning: no files for {model_name}")
            continue

        ds = xr.open_mfdataset(files, combine="by_coords", parallel=False)
        da = ds["ts"].sel(time=str(test_year) if test_year else OVERLAP_PERIOD)

        if months is not None:
            da = da.isel(time=da.time.dt.month.isin(months))

        da = _to_celsius(da)
        model_means.append(da.mean("time").compute())
        ds.close()
        print(f"  {model_name} done")

    return xr.concat(model_means, dim="model").mean("model")


def prep_sst_eval_vs_mmmhist(
    time_selection="May",
    use_cache=True,
    cache_dir="/scratch/v46/ls7238/cache",
    test_year=None,
):
    import os, pickle
    from dask.diagnostics import ProgressBar

    MONTH_MAP = {
        "January":1,"February":2,"March":3,"April":4,"May":5,"June":6,
        "July":7,"August":8,"September":9,"October":10,"November":11,"December":12,
    }
    SEASON_MAP = {
        "DJF":[12,1,2],"MAM":[3,4,5],"JJA":[6,7,8],"SON":[9,10,11],
        "MJJASO":[5,6,7,8,9,10],"NDJFMA":[11,12,1,2,3,4],
        "annual":list(range(1,13)),
    }

    if isinstance(time_selection, str) and time_selection in MONTH_MAP:
        sel_months = [MONTH_MAP[time_selection]]
    elif isinstance(time_selection, str) and time_selection in SEASON_MAP:
        sel_months = SEASON_MAP[time_selection]
    elif isinstance(time_selection, list):
        sel_months = time_selection
    else:
        raise ValueError(f"Unrecognised time_selection: {time_selection!r}")

    if test_year:
        use_cache = False

    cache_file = os.path.join(cache_dir, f"sst_mmm_{time_selection}.pkl")

    if use_cache and os.path.exists(cache_file):
        print(f"Loading from cache: {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print("Loading BARRA-R surface temperature...")
    with ProgressBar():
        da_eval = _load_barra_sfc_temp(months=sel_months, test_year=test_year).mean("time").compute()

    print("Loading BARPA MMM surface temperature...")
    da_hist = _load_barpa_ts_mmm(months=sel_months, test_year=test_year)
    da_hist = da_hist.interp(lat=da_eval.lat, lon=da_eval.lon, method="linear")

    with ProgressBar():
        da_hist = da_hist.compute()

    da_diff = da_hist - da_eval

    result = {
        "da_eval":         da_eval,
        "da_hist":         da_hist,
        "da_diff":         da_diff,
        "time_selection":  time_selection,
        "compute_anomaly": False,
        "test_year":       test_year,
        "var_name":        "Sfc Temp",
        "units":           "°C",
    }

    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump(result, f)
        print(f"Cached to: {cache_file}")

    return result


def prep_sst_per_model(
    time_selection="May",
    use_cache=True,
    cache_dir="/scratch/v46/ls7238/cache",
    test_year=None,
):
    import os, pickle
    from dask.diagnostics import ProgressBar

    MONTH_MAP = {
        "January":1,"February":2,"March":3,"April":4,"May":5,"June":6,
        "July":7,"August":8,"September":9,"October":10,"November":11,"December":12,
    }
    SEASON_MAP = {
        "DJF":[12,1,2],"MAM":[3,4,5],"JJA":[6,7,8],"SON":[9,10,11],
        "MJJASO":[5,6,7,8,9,10],"NDJFMA":[11,12,1,2,3,4],
        "annual":list(range(1,13)),
    }

    if isinstance(time_selection, str) and time_selection in MONTH_MAP:
        sel_months = [MONTH_MAP[time_selection]]
    elif isinstance(time_selection, str) and time_selection in SEASON_MAP:
        sel_months = SEASON_MAP[time_selection]
    elif isinstance(time_selection, list):
        sel_months = time_selection
    else:
        raise ValueError(f"Unrecognised time_selection: {time_selection!r}")

    if test_year:
        use_cache = False

    cache_file = os.path.join(cache_dir, f"sst_permodel_{time_selection}.pkl")

    if use_cache and os.path.exists(cache_file):
        print(f"Loading from cache: {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print("Loading BARRA-R surface temperature...")
    with ProgressBar():
        da_eval = _load_barra_sfc_temp(months=sel_months, test_year=test_year).mean("time").compute()

    da_diffs = {}
    for model_name, path in BARPA_TS_MODELS.items():
        print(f"  Loading {model_name}...")
        files = sorted(glob_files(f"{path}/*.nc"))
        if not files:
            print(f"    Warning: no files found")
            continue

        ds = xr.open_mfdataset(files, combine="by_coords", parallel=False)
        da = ds["ts"].sel(time=str(test_year) if test_year else OVERLAP_PERIOD)
        da = da.isel(time=da.time.dt.month.isin(sel_months))
        da = _to_celsius(da)

        da_model = da.mean("time").interp(lat=da_eval.lat, lon=da_eval.lon, method="linear")

        with ProgressBar():
            da_diffs[model_name] = (da_model - da_eval).compute()
        ds.close()
        print(f"    done")

    result = {
        "da_eval":         da_eval,
        "da_diffs":        da_diffs,
        "time_selection":  time_selection,
        "compute_anomaly": False,
        "test_year":       test_year,
        "var_name":        "Sfc Temp",
        "units":           "°C",
    }

    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump(result, f)
        print(f"Cached to: {cache_file}")

    return result




















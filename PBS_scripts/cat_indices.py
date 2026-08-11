"""Calculate Clear Air Turbulence indices for BARPA-R data and other data on a regular lat lon grid"""

import xarray as xr
import numpy as np

P=200
p=P
p0 = P-50
p1 = P+50
P0 = 1000
step_size=  0.1545 # fraction of degree
metres_per_degree = 111111

# these functions are used to calculate turbulence in climate models
# calculate differential components
def dy(ds, y= step_size,):
    """Step size in degrees. This function converts to metres"""
    return metres_per_degree * y
    # * ds[lon]**0

def dx(ds, lat="lat", x= step_size,):
    """Step size in degrees. This function converts to metres"""
    return metres_per_degree * np.cos( np.deg2rad(ds[lat])) * x

def dz(ds, z="z", p0=p0, p1=p1):
    """Change in height"""
    return ds[z].sel(pressure=p0) - ds[z].sel(pressure=p1)
    
def du_dx(ds, u="u", lon="lon",lat="lat", p=P, x=step_size):
    """Change in horizontal wind u with change in longitude."""
    return ds[u].sel(pressure=p).diff(lon)/dx(ds, lat=lat, x= 1,)

def dv_dx(ds, v="v", lon="lon",lat="lat", p=P, x= step_size):
    """Change in horizontal wind v with change in longitude (step_size for x).
    This function wraps similar du_dx function"""
    return du_dx(ds, u=v, lon=lon, lat=lat, p=p, x=x)

def du_dy(ds, u="u", lon="lon",lat="lat", p=P, y=step_size):
    """Change in horizontal wind u with change in latitude."""
    # return ds[u].sel(pressure=p).diff(lat)*y/dy(ds, y= y,)
    return (ds[u].sel(pressure=p)).diff(lat)/dy(ds, y= 1,)

def dv_dy(ds, v="v", lon="lon",lat="lat", p=P, y=step_size):
    """Change in horizontal wind v with change in latitude.
    This function wraps similar du_dy function"""
    return du_dy(ds, u=v, lon=lon, lat=lat, p=p, y=y)
    # return (ds[v].sel(pressure=p)* np.cos( np.deg2rad(ds[lat]))).diff(lat)*y/(dy(ds, y= y,)* np.cos( np.deg2rad(ds[lat])))

def du_dz(ds, u="u", z="z", p0=p0, p1=p1):
    """Change in horizontal wind u with change in  height."""
    return (ds[u].sel(pressure=p0) - ds[u].sel(pressure=p1))/dz(ds, z=z, p0=p0, p1=p1)

def dv_dz(ds, v="v", z="z", p0=p0, p1=p1):
    """Change in horizontal wind v with change in  height.
    This function wraps similar du_dz function"""
    return du_dz(ds, u=v, z=z, p0=p0, p1=p1)

# Calculate indices
def windspeed(ds, u="u", v="v",):
    """Horizontal windspeed"""
    return np.hypot(ds[u], ds[v])
    
def air_density(ds, t="t", p=P,):
    """Calculates air density from temperature (T) and pressure (P) according to the ideal gas law.
    R_specific is the specific gas constant for dry air
    Units: 
    * P in hPa (convert to Pa)
    * T in K
    * R_specific in J/kg/K
    """
    R_specific = 287.05
    T = ds[t]
    return p*100/(R_specific * T)
rho = air_density
    
def potential_temperature(ds, t = "t", p=P, P0=P0):
    """Calculates potential temperature from temperature (T) and pressure (P) according to the ideal gas law.
    R/c_p = 0.286 for air
    Units: 
    * P in hPa
    * T in K
    p is pressure level you are interested in
    P0 is the reference pressure usually 1000hPa
    """
    T = ds[t].sel(pressure = p)
    return T*(P0/p)**(0.2857)
theta = potential_temperature
    
def vertical_temperature_gradient(ds, t="t", P0=P0, z="z", p0=p0, p1=p1):
    return (potential_temperature(ds, t=t, p=p0, P0=P0) - potential_temperature(ds, t=t, p=p1, P0=P0))/dz(ds, z=z, p0=p0, p1=p1)
dtheta_dz = vertical_temperature_gradient

def BruntVaisala_frequency(ds, t="t", p=P, P0=P0, z="z", p0=p0, p1=p1, g=9.81):
    """N^2 Brunt-Vӓisӓlӓ frequency"""
    return g/(potential_temperature(ds, t=t, p=p, P0=P0)) * dtheta_dz(ds, t=t, z=z, p0=p0, p1=p1)
N2 = BruntVaisala_frequency

def coriolis_freq(ds, lat="lat"):
    # omega constant for earth, change for other planets
    omega = 7.2921*10**-5
    return 2*omega*np.sin(np.deg2rad(ds[lat]))
f = coriolis_freq

def vertical_vorticity(ds, u="u", v="v",
                        p=P,
                        lon="lon", lat="lat",
                        x= step_size, y=step_size):
    return dv_dx(ds, v=v, lon=lon, lat=lat, p=p, x=x) - du_dy(ds, u=u, lon=lon, lat=lat, p=p, y=y)
zeta = vertical_vorticity

def potential_vorticity(ds, u="u", v="v", t="t", z="z",
                        p=P, P0=P0, p0=p0, p1=p1, 
                        lon="lon", lat="lat",
                        x= step_size, y=step_size):
    "PV Potential Vorticity"
    f = coriolis_freq(ds, lat=lat)
    return 1/air_density(ds, t=t, p=p) * (zeta(ds, u=u, v=v, p=p, lon=lon, lat=lat, x= x, y=y) + f)* dtheta_dz(ds, t=t, P0=P0, z=z, p0=p0, p1=p1)
PV = potential_vorticity

def vertical_wind_shear(ds, u="u", v="v", z="z", p0=p0, p1=p1):
    return np.hypot(du_dz(ds, u=u, z=z, p0=p0, p1=p1),
                    dv_dz(ds, v=v, z=z, p0=p0, p1=p1))
VWS = vertical_wind_shear

def DSH(ds, u="u", v="v", lon="lon", lat="lat", p=P, x= step_size, y= step_size):
    """Shearing deformation"""
    return dv_dx(ds, v=v, lon=lon, lat=lat, p=p, x=x) + du_dy(ds, u=u, lon=lon, lat=lat, p=p, y=y)

def DST(ds, u="u", v="v", lon="lon", lat="lat", p=P, x= step_size, y= step_size):
    """Stretching deformation"""
    return du_dx(ds, u=u, lon=lon, lat=lat, p=p, x=x) - dv_dy(ds, v=v, lon=lon, lat=lat, p=p, y=y)
    
def deformation(ds, u="u", v="v", lon="lon", lat="lat", p=P, x= step_size, y= step_size):
    """sqrt((dvdx+dudy)**2 + (dudx-dvdy)**2)"""
    return np.hypot(DSH(ds, u=u, v=v, lon=lon, lat=lat, p=p, x= x, y= y),
                    DST(ds, u=u, v=v, lon=lon, lat=lat, p=p, x= x, y= y))
DEF = deformation

def divergence(ds, u="u", v="v", lon="lon", lat="lat", p=P, x= step_size, y= step_size):
    """dudx + dvdy"""
    return du_dx(ds, u=u, lon=lon, lat=lat, p=p, x=x) + dv_dy(ds, v=v, lon=lon, lat=lat, p=p, y=y)
DIV = divergence

def divergence_tendency(ds, u="u", v="v", lon="lon", lat="lat", p=P, x= step_size, y= step_size):
    C=0.01
    return C*((DIV(ds, u=u, v=v, lon=lon, lat=lat, p=p, x= step_size, y= step_size).differentiate("time", datetime_unit="6h")**2)**0.5)
DVT = divergence_tendency

def richardson_number(ds, u="u", v="v", t="t", z="z", p=P, p0=p0, p1=p1, P0=P0):
    """The Ri less than some small number indicates Kelvin-Helmholtz instability when VWS is large and/or N2 is small."""
    return  N2(ds, t=t, z=z, p=p, p0=p0, p1=p1, P0=P0)/(VWS(ds, u=u, v=v, z=z, p0=p0, p1=p1)**2)
Ri = richardson_number

def turbulence_index_1(ds, u="u", v="v", z="z", p0=p0, p1=p1, lon="lon", lat="lat", p=P, x= step_size, y= step_size):
    """Ellrod1 index = VWS*DEF """
    return VWS(ds, u=u, v=v, z=z, p0=p0, p1=p1) * DEF(ds, u=u, v=v, lon=lon, lat=lat, p=p, x=x, y=y)
TI1 = turbulence_index_1
ellrod1 = turbulence_index_1

def turbulence_index_2(ds, u="u", v="v", z="z", p0=p0, p1=p1, lon="lon", lat="lat", p=P, x= step_size, y= step_size):
    """Ellrod2 index = VWS*(DEF-DIV)"""
    return VWS(ds, u=u, v=v, z=z, p0=p0, p1=p1) * (DEF(ds, u=u, v=v, lon=lon, lat=lat, p=p, x=x, y=y) - DIV(ds, u=u, v=v, lon=lon, lat=lat, p=p, x= x, y= y))
TI2 = turbulence_index_2
ellrod2 = turbulence_index_2
    
def turbulence_index_3(ds, u="u", v="v", z="z", p0=p0, p1=p1, lon="lon", lat="lat", p=P, x= step_size, y= step_size):
    """Ellrod3 index - VWS * DEF + DVT"""
    return VWS(ds, u=u, v=v, z=z, p0=p0, p1=p1) * DEF(ds, u=u, v=v, lon=lon, lat=lat, p=p, x=x, y=y) + DVT(ds, u=u, v=v, lon=lon, lat=lat, p=p, x= x, y= y)
TI3 = turbulence_index_3
ellrod3 = turbulence_index_3

def absolute_vorticity(ds, u="u", v="v", lon="lon", lat="lat", p=P, x=step_size, y=step_size):
    """Absolute vorticity = relative vorticity (zeta) + Coriolis parameter (f)"""
    return zeta(ds, u=u, v=v, lon=lon, lat=lat, p=p, x=x, y=y) + coriolis_freq(ds, lat=lat)
AbsVort = absolute_vorticity



# calculate all turbulence indices   
def calc_turbulence_indices(
    ds, which=None,
    u="u", v="v", t="t", z="z",
    x=step_size, y=step_size,
    p=P, P0=P0, p0=None, p1=None, g=9.81,
    lon="lon", lat="lat"
):
    """
    Apply the turbulence index calculation to the dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset.
    which : str
        Index to compute. Case-sensitive.
        Options: ['air_density', 'theta', 'dtheta_dz', 'coriolis_freq', 'windspeed',
                  'VWS', 'N2', 'PV', 'DEF', 'DIV', 'DVT', 'Ri', 'TI1', 'TI2', 'TI3']
    """
    if p0 is None:
        p0=p-50
    if p1 is None:
        p1=p+50
    

    # Map index names to functions and their arguments
    index_map = {
        # "air_density": (air_density, dict(t=t, p=p)),
        # "theta": (potential_temperature, dict(t=t, p=p0, P0=P0)),
        # "dtheta_dz": (vertical_temperature_gradient, dict(t=t, P0=P0, z=z, p0=p0, p1=p1)),
        # "coriolis_freq": (coriolis_freq, dict(lat=lat)),
        "windspeed": (windspeed, dict(u=u, v=v)),
        "VWS": (VWS, dict(u=u, v=v, z=z, p0=p0, p1=p1)),
        "N2": (N2, dict(t=t, p=p, P0=P0, z=z, p0=p0, p1=p1)),
        "PV": (PV, dict(u=u, v=v, t=t, z=z, p=p, P0=P0, p0=p0, p1=p1, lon=lon, lat=lat, x=x, y=y)),
        "DEF": (DEF, dict(u=u, v=v, lon=lon, lat=lat, p=p, x=x, y=y)),
        "DIV": (DIV, dict(u=u, v=v, lon=lon, lat=lat, p=p, x=x, y=y)),
        "DVT": (DVT, dict(u=u, v=v, lon=lon, lat=lat, p=p, x=x, y=y)),
        "Ri": (Ri, dict(u=u, v=v, t=t, z=z, p=p, p0=p0, p1=p1, P0=P0)),
        "TI1": (TI1, dict(u=u, v=v, z=z, p0=p0, p1=p1, lon=lon, lat=lat, p=p, x=x, y=y)),
        "TI2": (TI2, dict(u=u, v=v, z=z, p0=p0, p1=p1, lon=lon, lat=lat, p=p, x=x, y=y)),
        "TI3": (TI3, dict(u=u, v=v, z=z, p0=p0, p1=p1, lon=lon, lat=lat, p=p, x=x, y=y)),
        "AbsVort": (AbsVort, dict(u=u, v=v, lon=lon, lat=lat, p=p, x=x, y=y)),

    }

    # Apply requested indices
    
    if which in index_map:
        func, kwargs = index_map[which]
        ds[which] = func(ds, **kwargs)
    else:
        raise ValueError(f"Unknown index: {which}. Use one of ['windspeed', 'VWS', 'N2', 'PV', 'DEF', 'DIV', 'DVT', 'Ri', 'TI1', 'TI2', 'TI3']")

    return ds

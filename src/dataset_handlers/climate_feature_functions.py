import os

import numpy as np  
import xarray as xr 

def temperature_aggregate(temperature_object: xr.DataArray, method : str) -> xr.DataArray:
    """
    Resample to daily values and aggregate according to the chosen method

    Args:
        temperature_object (xr.DataArray): temperature file 
        method (str): Method to be used after a xarray resampling.
    
    Returns:
        aggregated_temperature (xr.DataArray): daily temperature file according to chosen method
    """
    
    temperature_resampled = temperature_object.resample(valid_time="1D")

    aggregated_temperature = getattr(temperature_resampled, method)

    aggregated_temperature = aggregated_temperature(dim="valid_time")
    
    aggregated_temperature.attrs['long_name'] = f'Daily {method} temperature'
    aggregated_temperature.attrs['standard_name'] = f'{method}_temperature'

    return aggregated_temperature


def buck_vapour_pressure(filepath_directory, year, ea : bool = False) -> xr.DataArray:
    """
    Compute saturation vapour pressure according to Buck 1996 constants if the temperature is above or below 0°C.

    Args:
        ea (bool): If we wish to get vapour pressure at dewpoint temperature, else returns vapour pressure at air temperature. Defaults to False.

    Returns:
        x (xr.DataArray): saturation vapour pressure (in hPa)
    
    """
    feat_name = "t2m"

    if ea:
        # actual saturation vapour pressure or vapour pressure at dewpoint temperature
        feat_name = "d2m"

    with xr.open_dataset(os.path.join(filepath_directory, f"{feat_name}_{year}.nc")) as tmp:
        x = tmp[feat_name]
        #Switch K to °C
        if x.attrs.get("units") == "K":
            x = x - 273.15
            x.attrs["units"] = "°C"

        x = xr.where(x > 0,
                    6.1121*np.exp(
                    (18.678 - x/234.5)*
                    (x/(257.14+x))),
                    6.1115*np.exp(
                    (23.036 - x/333.7)*
                    (x/(279.82+x)))
                    )
            
    return x

def windspeed(u:xr.DataArray, v: xr.DataArray) -> xr.DataArray:
    """
    Compute wind speed in m/s

    Args:
        u (xr.DataArray): 10m-u component of wind
        v (xr.DataArray): 10m-v component of wind

    Returns:
        wind speed (xr.DataArray): (in m/s)
    """
    return np.sqrt(u**2+v**2)

def relative_humidity(ea:xr.DataArray, es:xr.DataArray) -> xr.DataArray:
    """
    Compute relative humidity in % 

    Args:
        ea (xr.DataArray): actual vapour pressure, or vapour pressure at dewpoint temperature (in mb)
        es (xr.DataArray): saturation vapour pressure or vapor pressure at air temperature (in mb)

    Returns:
        relative humidity (xr.DataArray): (in %) 
    """
    return (ea/es)*100

def specific_humidity(ea:xr.DataArray, p:xr.DataArray) -> xr.DataArray:
    """
    Compute specific humidity in (kg/kg)

    Args:
        ea (xr.DataArray): actual vapour pressure, or vapour pressure at dewpoint temperature (in mb)
        p (xr.DataArray): surface pressure (in Pa) we convert it into hPa (*100)
    
    Returns:
        specific humidity (xr.DataArray): (in kg/kg)
    """
    return np.divide(0.622*ea, (p*100)-(0.378*ea))

def vapour_pressure_deficit(es:xr.DataArray, rh:xr.DataArray) -> xr.DataArray:
    """
    Compute vapour pressure deficit, we divide by 10 to get kPa units
    
    Args:
        es (xr.DataArray): saturation vapour pressure or vapour pressure at air temperature (in mb)
        rh (xr.DataArray): relative humidity (in %)

    Returns:
        vapour pressure deficit (xr.DataArray): (in kPa)
    """
    return (es * (1 - rh/100))/10

def geopotential_height(x: xr.DataArray) -> xr.DataArray:
    """
    Get geopotential height by using Earth gravity constant 9.80665 m/s²

    Args:
        x (xr.DataArray): geopotential height array in m²/s²
    
    Returns:
        geopotential height (xr.DataArray): Geopotential height in meters.
        
    """
    earth_gravitational_acceleration = 9.80665
    
    return x/earth_gravitational_acceleration

from attrs import define
import os

import numpy as np
import xarray as xr 

@define
class Era5Info:
    short_name: str 
    uid: int

@define
class RectangularPolygon:
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float 

def instanciate_mapping(data: list[tuple[str, str, int]]) -> dict[str, Era5Info]:
    """
    
    Instanciate a mapping of CDS data to enable us easier access during our tasks. 

    Args:
        data (list[tuple[str,str,int]]): Required Input should be of the following form [("2m_temperature","t2m",168)] with full_name, short_name, uid
    
    Returns:
        dictionary with full_name as keys and Dataclass for short_name, uid
    
    """
    return {
        key: Era5Info(short_name, uid)
        for key, short_name, uid in data
    }

def geo_filter(data:xr.DataArray, lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> xr.DataArray:
    """
    Geographical filter on xarray arrays. 

    Args: 
        data (xr.DataArray): Data to undergo geographical filter, that has "latitude" and "longitude" dimensions available.
        lat_min (float): Minimum latitude bound
        lat_max (float): Maximum latitude bound
        lon_min (float): Minimum longitude bound
        lon_max (float): Maximum longitude bound
    
    Returns:
        data (xr.DataArray): Geographically filtered data
    """
    lat_indices = np.where((data['latitude'] >= lat_min) & (data['latitude'] <= lat_max))[0]
    lon_indices = np.where((data['longitude'] >= lon_min) & (data['longitude'] <= lon_max))[0]
    return data.isel(latitude=lat_indices, longitude=lon_indices)


def remove_files(filepath_dir: str, year: str , *features: str) -> None:
    """
    Remove temporary files that were used to create features downstream

    Args:
        filepath_dir (str): Direct path to the desired file 
        year (str): Year of the desired file
        features (str): features to be removed such as "tp","sp" or else.

    Example:
        remove_files("./data", "2001", "tp","sp","vpd")
    """
    for feature_name in features:
        filepath = os.path.join(filepath_dir, f"{feature_name}_{year}.nc")
        if os.path.exists(filepath):
            os.remove(filepath) 

def assess_file_existence(year:str,filepath_dir:str, features: list = ["tmnn","tmmx","tmean","10ws","rmin","rmax","vpd","sph","pr","srad"]) -> bool:
    """
    Assess if all files are present in the given directory 

    Args:
        year (str): Year focus
        filepath_dir (str): Directory of the files 
        features (list): List of features to be assessed. By default we place features that are relative to our study

    Returns: 
        boolean True if all files are present in the given directory 
    """
    return all(os.path.exists(os.path.join(filepath_dir, f"{singular_feat}_{year}.nc")) for singular_feat in features)
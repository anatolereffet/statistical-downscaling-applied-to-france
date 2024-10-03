import os 
from urllib.request import urlretrieve

import cdsapi
import xarray as xr

from src.dataset_handlers.climate_feature_functions import geopotential_height
from src.dataset_handlers import EraInterimApiCall, Era5SingleLevelsApiCall
from src.utils import instanciate_mapping, RectangularPolygon

def generate_invariant_filepaths(download_parent_dir:str, data_dir:str, *invariant_features: str) -> tuple:
    """
    Generate filepaths for invariant (i.e. static over time) features such as geopotential

    Args:
        download_parent_dir (str): Parent directory holding all data 
        data_dir (str): directory name for specific dataset 
        invariant_features (str): desired invariant features such as geopotential, elevation
    
    Returns:
        tuple of invariant features direct path
    """
    invariant_allocator = []

    for feature in invariant_features:
        feature_filepath = os.path.join(os.getcwd(), download_parent_dir, data_dir, f"{feature}.nc")
        invariant_allocator.append(feature_filepath)

    return tuple(invariant_allocator)

def get_geopotential_data(download_parent_dir:str, year:str, region: RectangularPolygon) -> None:
    """
    Create specific requests for geopotential data 

    Args:
        download_parent_dir (str): Parent directory holding all data 
        year (str): Year of the desired file
        region (RectangularPolygon): Desired area
    """

    geopotential_info = [
        ("geopotential", "z", 129)
    ]
    geopotential_feature_identifier = instanciate_mapping(geopotential_info)

    era5_hourly = Era5SingleLevelsApiCall(download_parent_dir, region)

    era5_hourly_request = era5_hourly._create_request(year, list(geopotential_feature_identifier.keys()))
    
    era5_hourly_request["day"] = ["01"]
    era5_hourly_request["month"] = ["01"]
    era5_hourly_request["time"] = ["01"]

    era5_interim = EraInterimApiCall(download_parent_dir, region)
    
    era5_interim_request = era5_interim._create_request(year, f"{geopotential_feature_identifier["geopotential"].uid}.128", False)
    era5_interim_request["date"] = f"{year}-01-01"
    era5_interim_request["time"] = "00:00"

    geopotential_filepath, elevation_filepath = generate_invariant_filepaths(download_parent_dir, era5_hourly.cds_apiname, "geopotential","elevation")

    if not os.path.exists(geopotential_filepath) and not os.path.exists(elevation_filepath):
        cdsapi.Client().retrieve(era5_hourly.cds_apiname, era5_hourly_request, geopotential_filepath)
    
    geopotential_filepath, elevation_filepath = generate_invariant_filepaths(download_parent_dir, era5_interim.cds_apiname, "geopotential","elevation")

    if not os.path.exists(geopotential_filepath) and not os.path.exists(elevation_filepath):
        cdsapi.Client().retrieve(era5_interim.cds_apiname, era5_interim_request, geopotential_filepath)

    geopotential_filepath, elevation_filepath = generate_invariant_filepaths(download_parent_dir, "reanalysis-era5-land", "geopotential","elevation")

    if not os.path.exists(geopotential_filepath) and not os.path.exists(elevation_filepath):
        # This might break in the future, the link can be found in the documentation url below
        # https://confluence.ecmwf.int/display/CKB/ERA5-Land%3A+data+documentation#heading-Table1surfaceparametersinvariantsintime
        urlretrieve("https://confluence.ecmwf.int/download/attachments/140385202/geo_1279l4_0.1x0.1.grib2_v4_unpack.nc?version=1&modificationDate=1591983422003&api=v2",
                    geopotential_filepath)
        
def get_elevation_data(download_parent_dir:str) -> None:
    """
    Transform geopotential files in geopotential height

    Args:
        download_parent_dir (str): Parent directory holding all data
    """
    for dataset_apiname in ["reanalysis-era5-land","reanalysis-era-interim","reanalysis-era5-single-levels"]:
        
        geopotential_filepath, elevation_filepath = generate_invariant_filepaths(download_parent_dir, dataset_apiname, "geopotential","elevation")

        
        if os.path.exists(elevation_filepath):
            print('Elevation already processed')
            continue
        
        geopotential_data = xr.open_dataset(geopotential_filepath)

        elevation = geopotential_height(geopotential_data)

        elevation.to_netcdf(elevation_filepath)

        # Cleanup directory space
        os.remove(geopotential_filepath)
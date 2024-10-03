import os

import cdsapi 
import xarray as xr

from src.utils import RectangularPolygon, assess_file_existence, remove_files
from src.dataset_handlers.climate_feature_functions import (
    buck_vapour_pressure,  
    relative_humidity,
    specific_humidity,
    temperature_aggregate,
    vapour_pressure_deficit,
    windspeed, 
)

class Era5LandApiCall:
    """
    Creates request specific to ERA5-Land dataset
    """
    def __init__(self, output_dir:str, region:RectangularPolygon) -> None:
        """
        Initializes Era5LandApiCall class
        """
        self.client = cdsapi.Client()
        self.output_dir = output_dir 
        self.region = region
        self.cds_apiname = "reanalysis-era5-land"

    def create_request(self, year:str, features:list[str], month:str="01") -> dict:

        request = {
            "variable": features,
            "year": year,
            "day": [f"{day:02}" for day in range(1,32)],
            "month": [month],
            "time": [f"{hour:02}:00" for hour in range(24)],
            "data_format":"netcdf",
            "download_format":"unarchived",
            "area":[
                self.region.max_lon,
                self.region.min_lat,
                self.region.min_lon,
                self.region.max_lat 
            ]
        }
        return request 
    
    def download_data(self, year:str, features:dict) -> None:

        if assess_file_existence(year, self.output_dir):
            # Early return if all end files are already there
            return

        # Get only keys as we solely need for the request
        features = list(features.keys())

        for month in range(1,13):
            month_str = f"{month:02}"

            output_filename = f"{year}-{month_str}-rawdata.nc"
            output_filepath = os.path.join(self.output_dir, output_filename)

            if os.path.isfile(output_filepath):
                continue

            request = self._create_request(year, features, month_str)
            self.client.retrieve(self.cds_apiname, request, output_filepath)

class Era5LandProcessor:
    """
    Process data specific to ERA5-Land dataset
    """
    def __init__(self, download_parent_dir: str, year: str, features_info: dict, region:RectangularPolygon) -> None:
        """
        Initializes Era5LandProcessor class
        """
        self.cds_apiname = "reanalysis-era5-land"

        self.download_parent_dir = download_parent_dir
        self.year = year 
        self.features_info = features_info
        self.region = region
        self.filepath_dir = os.path.join(self.download_parent_dir, self.cds_apiname)

    def _singularize_features(self) -> None:
        """
        Singularize features, according to months as retrieval is allowed on a month basis only

        Concatenate the features to get back a year per feature
        """
        if assess_file_existence(self.year, self.filepath_dir):
            # Early return if all end files are already there
            return
        if assess_file_existence(self.year, self.filepath_dir,["u10","v10","d2m","sp","ssrd","t2m","tp"]):
            #Early return if intermediary files for feature engineering are already available
            return

        data_holder = {var: [] for var in list(self.features_info.keys())}

        for var in list(self.features_info.keys()):
            short_name_feature = self.features_info.get(var).short_name

            for month_nb in range(1, 13):
                filepath = os.path.join(self.download_parent_dir, self.cds_apiname, f"{self.year}-{month_nb:02}-rawdata.nc")

                if not os.path.exists(filepath):
                    raise ValueError(f"File not found on this path {filepath}")
            
                data = xr.open_dataset(filepath)[short_name_feature]
                data_holder[var].append(data)
            
            data_holder[var] = xr.concat(data_holder[var], dim = "valid_time")
            output_filepath = os.path.join(self.download_parent_dir, self.cds_apiname, f"{short_name_feature}_{self.year}.nc")
            data_holder[var].to_netcdf(output_filepath)
        
        data.close()
        for month_nb in range(1, 13):
            filepath = os.path.join(self.download_parent_dir, self.cds_apiname, f"{self.year}-{month_nb:02}-rawdata.nc")
            os.remove(filepath)

    def _feature_engineer(self) -> None:
        """
        Create the necessary features for our study
        """
        if assess_file_existence(self.year, self.filepath_dir):
            # Early return if all end files are already there
            return

        with xr.open_dataset(os.path.join(self.filepath_dir, f"t2m_{self.year}.nc")) as tmp:
            temperature = tmp["t2m"]

            tmin = temperature_aggregate(temperature, "min")
            tmin.name = "tmmn"
            tmin.to_netcdf(os.path.join(self.filepath_dir, f"tmmn_{self.year}.nc"))

            tmax = temperature_aggregate(temperature, "max")
            tmax.name = "tmmx"
            tmax.to_netcdf(os.path.join(self.filepath_dir, f"tmmx_{self.year}.nc"))

            tmean = temperature_aggregate(temperature, "mean")
            tmean.name = "tmean"
            tmean.to_netcdf(os.path.join(self.filepath_dir, f"tmean_{self.year}.nc"))

        ea = buck_vapour_pressure(self.filepath_dir, self.year, ea = True)
        es = buck_vapour_pressure(self.filepath_dir, self.year, ea = False)

        rh = relative_humidity(ea, es)
        rh.attrs["unit"] = "%"

        rmin = rh.resample(valid_time='1D').min(dim='valid_time')
        rmin.name = "rmin"
        rmin.attrs['long_name'] = 'Minimum Relative humidity calculated as (ea/es)*100 with ea actual saturation vapour pressure, es saturation vapour pressure'
        rmin.attrs['standard_name'] = 'min_relative_humidity'
        rmin.to_netcdf(os.path.join(self.filepath_dir, f"rmin_{self.year}.nc"))

        rmax = rh.resample(valid_time='1D').max(dim='valid_time')
        rmax.name = "rmax"
        rmax.attrs['long_name'] = 'Maximum Relative humidity calculated as (ea/es)*100 with ea actual saturation vapour pressure, es saturation vapour pressure'
        rmax.attrs['standard_name'] = 'max_relative_humidity'
        rmax.to_netcdf(os.path.join(self.filepath_dir, f"rmax_{self.year}.nc"))

        # Wind speed & direction
        with xr.open_dataset(os.path.join(self.filepath_dir, f"u10_{self.year}.nc")) as tmp_u, \
             xr.open_dataset(os.path.join(self.filepath_dir, f"v10_{self.year}.nc")) as tmp_v:
            u = tmp_u["u10"]
            v = tmp_v["v10"]

            ws = windspeed(u, v).resample(valid_time="1D").mean(dim="valid_time")
            ws.name = "10ws"
            ws.attrs['long_name'] = 'Daily averaged wind speed at 10m calculated as np.sqrt(u**2+v**2)'
            ws.attrs['standard_name'] = 'wind_speed'
            ws.attrs["unit"] = "m s**-1"
            ws.to_netcdf(os.path.join(self.filepath_dir, f"10ws_{self.year}.nc"))
        
        with xr.open_dataset(os.path.join(self.filepath_dir, f"sp_{self.year}.nc")) as tmp:
            pressure = tmp['sp']
            sph = specific_humidity(ea, pressure).resample(valid_time='1D').mean(dim='valid_time')
            sph.name = "sph"
            sph.attrs['long_name'] = 'Specific humidity calculated as (0.622*ea/(p-(0.378*ea)) with p surface pressure, ea actual saturation vapour pressure'
            sph.attrs['standard_name'] = 'specific_humidity'
            sph.attrs["unit"] = "kg/kg"
            sph.to_netcdf(os.path.join(self.filepath_dir, f"sph_{self.year}.nc"))

        vpd = vapour_pressure_deficit(es, rh)
        vpd.name = "vpd"
        vpd.attrs['long_name'] = 'Vapour pressure deficit calculated as es * (1 - rh/100) with rh relative humidity, es saturation vapour pressure'
        vpd.attrs['standard_name'] = 'vapor_pressure_deficit'
        vpd.attrs["unit"] = "kPa"
        mean_vpd = vpd.resample(valid_time="1D").mean(dim="valid_time")
        mean_vpd.to_netcdf(os.path.join(self.filepath_dir, f"vpd_{self.year}.nc"))

        precipitation = self._aggregate_fluxes("tp")
        # m to mm/day
        precipitation *= 1000 
        precipitation.name = "pr"
        precipitation.attrs["unit"] = "mm"
        precipitation.to_netcdf(os.path.join(self.filepath_dir, f"pr_{self.year}.nc"))

        shortwave_flux = self._aggregate_fluxes("ssrd")
        shortwave_flux /= (60*60*24)
        shortwave_flux.name = "srad"
        shortwave_flux.attrs['long_name'] = 'Surface solar radiation downwards daily accumulation'
        shortwave_flux.attrs['standard_name'] = 'surface_solar_radiation_downwards'
        shortwave_flux.attrs["unit"] = "W m^-2"
        shortwave_flux.to_netcdf(os.path.join(self.filepath_dir, f"srad_{self.year}.nc"))

        remove_files(self.filepath_dir, self.year, "tp","t2m","d2m","sp","ssrd","u10","v10")

    def _aggregate_fluxes(self, feature: str) -> xr.DataArray:
        """
        Aggregate flux features such as precipitation or downwards shortwave radiation 

        This requires to have year N and year N+1 as convention wise requires to use the following day
        https://confluence.ecmwf.int/pages/viewpage.action?pageId=197702790
        """
        next_year = int(self.year) + 1
        current_filepath = os.path.join(self.filepath_dir, f"{feature}_{self.year}.nc")
        next_filepath = os.path.join(self.filepath_dir, f"{feature}_{str(next_year)}.nc")
    
        if not os.path.exists(current_filepath):
            raise FileNotFoundError(f"{feature} data for year {self.year} not found")
        
        with xr.open_dataset(current_filepath) as current_flux:
            flux_sync = current_flux[feature]

            if os.path.exists(next_filepath):
                with xr.open_dataset(next_filepath) as next_flux:
                    flux_next_year = next_flux[feature]

                    flux_sync = xr.concat([flux_sync, flux_next_year], dim="valid_time")
            
            # Get utc 00:00 data
            flux_sync = flux_sync.sel(valid_time=flux_sync["valid_time"].dt.hour == 0)
            # Pull d+1 utc 00:00
            flux_sync = flux_sync.shift(valid_time=-1)
            
            # Keep only current year
            flux_sync = flux_sync.sel(valid_time=flux_sync["valid_time"].dt.year == int(self.year))
            #Drop expver dimension to sync with other feature 
            flux_sync = flux_sync.drop_vars("expver", errors="ignore")
            
        return flux_sync
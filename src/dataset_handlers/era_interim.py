import os 

import cdsapi
import xarray as xr

from src.utils import RectangularPolygon, assess_file_existence
from src.dataset_handlers.climate_feature_functions import (
    buck_vapour_pressure,  
    relative_humidity,
    specific_humidity,
    temperature_aggregate,
    vapour_pressure_deficit,
    windspeed, 
)

class EraInterimApiCall:
    """
    Creates request specific to ERA Interim dataset
    """
    def __init__(self, output_dir:str, region:RectangularPolygon) -> None:
        """
        Initializes EraInterimApiCall class
        """
        self.client = cdsapi.Client()
        self.output_dir = output_dir 
        self.region = region
        self.cds_apiname = "reanalysis-era-interim"

    def _create_request(self, year:str, features:str, flux:bool=False) -> dict:

        request = {
            'class': 'ei',
            'format': 'netcdf',
            'param': features,
            'expver': 1,
            "stream": "oper",
            'levtype':'sfc',
            'type':'an',
            'date': f'{year}-01-01/to/{year}-12-31',
            'time': '00:00/06:00/12:00/18:00',
            'area': self._convert_region_to_str(),
            'grid':[0.75,0.75]
        }
        if flux:
            request.update({'type':'fc',
                            'step': 12,
                            'time':'00:00/12:00'})

        return request
    
    def download_data(self, year, features)-> None:
        if assess_file_existence(year, self.output_dir):
            # Early return if all end files are already there
            return
        
        flux_features = self._create_uid_strings(features, flux = True)
        nonflux_features = self._create_uid_strings(features, flux = False)

        output_filepath = os.path.join(self.output_dir, f"{year}-fluxrawdata.nc")
        self._singular_request(year, flux_features, output_filepath, flux=True)

        output_filepath = os.path.join(self.output_dir, f"{year}-nonfluxrawdata.nc")
        self._singular_request(year, nonflux_features, output_filepath, flux=False)

    def _singular_request(self, year:str, features:list, output_filepath:str, flux:bool) -> None:

        if os.path.isfile(output_filepath):
            print('raw data already exists, skip')
            return
        
        request = self._create_request(year, features, flux)
        self.client.retrieve(self.cds_apiname, request, output_filepath)

    def _create_uid_strings(self, mapping:dict, flux: bool=True) -> str:
        """
        We create uid strings compatible with Era Interim API request
        X.128/X.128/... is convention

        Args:
            mapping (dict): Feature mapping given as input {feature: EraInfo(short_name, uid)}
            flux (bool): If we are generating uid strings for flux features or not. Defaults to True.
        """
        return "/".join(f"{info.uid}.128" for info in mapping.values() if (info.uid in (169, 228)) == flux)
    
    def _convert_region_to_str(self) -> str:
        """
        Input requested by CDS N/W/S/E

        Returns:
            area_request (str): Geographical filter as required by CDS API
        """
        N = self.region.max_lat
        S = self.region.min_lat
        W = self.region.min_lon 
        E = self.region.max_lon 

        area_request = f"{N}/{W}/{S}/{E}"

        return area_request
    
class EraInterimProcessor:
    """
    Process data specific to ERA Interim dataset
    """
    def __init__(self, download_parent_dir: str, year: str, features_info: dict, region:RectangularPolygon)-> None:
        """
        Initializes ERAInterimProcessor class
        """
        self.cds_apiname = "reanalysis-era-interim"

        self.download_parent_dir = download_parent_dir 
        self.year = year 
        self.features_info = features_info
        self.region = region
        self.filepath_dir = os.path.join(self.download_parent_dir, self.cds_apiname)

    def _singularize_features(self):
        """
        Separate the initial aggregated features in singular files
        """
        if assess_file_existence(self.year, self.filepath_dir):
            # Early return if all end files are already there
            return
        if assess_file_existence(self.year, self.filepath_dir,["u10","v10","d2m","sp","ssrd","t2m","tp"]):
            #Early return if intermediary files for feature engineering are already available
            return
        
        # Shortwave downwards and total precipitation uids
        flux_uids = {169, 228}
        flux_features = {key: info for key, info in self.features_info.items() if info.uid in flux_uids}
        non_flux_features = {key: info for key, info in self.features_info.items() if info.uid not in flux_uids}

        self._singularize(flux_features, flux = True)
        self._singularize(non_flux_features, flux = False)

    def _feature_engineer(self):
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
        # Switch to W m^-2 (from W m^-2 s)
        shortwave_flux /= (60*60*24)
        shortwave_flux.name = "srad"
        shortwave_flux.attrs['long_name'] = 'Surface solar radiation downwards daily accumulation'
        shortwave_flux.attrs['standard_name'] = 'surface_solar_radiation_downwards'
        shortwave_flux.attrs["unit"] = "W m^-2"
        shortwave_flux.to_netcdf(os.path.join(self.filepath_dir, f"srad_{self.year}.nc"))
        

    def _singularize(self, features, flux: bool) -> None:
        """
        Singularize according to file source.
        """
        file_id_default = 'nonfluxrawdata'

        if flux:
            file_id_default = 'fluxrawdata'

        filepath = os.path.join(self.download_parent_dir, self.cds_apiname, f"{self.year}-{file_id_default}.nc")
        if not os.path.exists(filepath):
            raise ValueError(f"File not found on this path {filepath}")
        
        data = xr.open_dataset(filepath)
        for i in features:
            short_name_feature = features.get(i).short_name

            feature_data = data[short_name_feature]

            feature_data.to_netcdf(os.path.join(self.download_parent_dir, self.cds_apiname, f"{short_name_feature}_{self.year}.nc"))
        
        feature_data.close()

    def _aggregate_fluxes(self, feature: str) -> xr.DataArray:
        """
        Aggregate flux feature such as precipitation or downwards shortwave radiation 

        When possible take year N-1 and year N to have an accurate flux computation.
        """
        previous_year = int(self.year) - 1
        current_filepath = os.path.join(self.filepath_dir, f"{feature}_{self.year}.nc")
        previous_filepath = os.path.join(self.filepath_dir, f"{feature}_{str(previous_year)}.nc")
    
        if not os.path.exists(current_filepath):
            raise FileNotFoundError(f"{feature} data for year {self.year} not found")

        with xr.open_dataset(current_filepath) as current_flux:
            flux_sync = current_flux[feature]

            if os.path.exists(previous_filepath):
                with xr.open_dataset(previous_filepath) as previous_flux:
                    flux_previous_year = previous_flux[feature]
                    flux_sync = xr.concat([flux_previous_year, flux_sync], dim="valid_time")
            
            # Filter on year 
            flux_sync = flux_sync.sel(valid_time=flux_sync["valid_time"].dt.year == int(self.year))
            
            flux_sync = flux_sync.resample(valid_time="1D").sum(dim="valid_time")
        
        return flux_sync
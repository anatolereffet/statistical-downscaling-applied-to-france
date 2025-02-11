import os 

from src.constants import DATASET_MAPPER, PERIMETER_POLYGON
from src.utils import geo_filter, RectangularPolygon

import numpy as np 
import xarray as xr 

class DataHoarder():
    """
    Class enabling the computation of X,y train and test set easily
    """
    def __init__(self, target:str, features:list[int], months: list[int], scenario_params:dict, resolution_params:dict, bool_fit_per_grid:bool=False, data_directory:str = "./data") -> None:
        """
        Initializes DataHoarder class

        Args:
            target (str): target feature
            features (list[str]): list of features 
            months (list[int]): List of months to consider, set to None to consider the full year
            scenario_params (dict): Dictionary of scenarios holding train, test years and train, test polygon according to scenario
            resolution_params (dict): Dictionary that needs the following keys ['y_f','X_c','Z_topo','y_c'] or it will break such as 
                                      resolution_params ={"y_f":"10km",
                                                          "X_c":"25km",
                                                          "Z_topo":"50km",
                                                          "y_c":"25km} 
                                    Note however that 'y_c' is optional, the features here are X_c, Z_topo, y_c; c stands for coarse, f for fine.
            bool_fit_per_grid (bool): Boolean to choose between location based data processing or flattening. Defaults to False
            data_directory (str): Directory where the data is stored. Defaults to "./data"
        
        """
        self.target = target 
        self.features = features 
        self.bool_fit_per_grid = bool_fit_per_grid 
        self.scenario_params = scenario_params
        self.resolution_params = resolution_params
        self.data_directory = data_directory

        # Early check to avoid issues downstream
        if not all(key in ["y_f", "X_c", "Z_topo", "y_c"] for key in resolution_params.keys()) or \
            not all(required_key in resolution_params for required_key in ["y_f", "X_c", "Z_topo"]):
            raise ValueError("Wrong resolution parameters keys")

        if months is None:
            self.months_list = list(range(1,13))
        else:
            self.months_list = months

    def _assess_single_predictor_set(self) -> bool:
        """
        Returns True if we do not aggregate lower resolution target with the predictors
        """
        return "y_c" not in self.resolution_params

    def pull_X_y(self, set: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Pull X and y according to the chosen set 

        Args:
            set (str): "train" or "test"

        Returns:
            X, y
        """

        if self._assess_single_predictor_set():
            X = self._pull_single_predictor_set(set, self.bool_fit_per_grid)

        else:
            X = self._pull_aggregated_predictors(set)

        y = self._pull_target(set)

        if not self.bool_fit_per_grid:
            nan_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
            self.nan_mask = nan_mask
            X = X[nan_mask]
            y = y[nan_mask]

        return X, y
    
    def generate_baseline_data_by_mapping_LR_to_HR(self, set:str) -> np.ndarray:
        """
        Generates baseline predictions using lower resolution target data for the specified dataset.

        Raises a RuntimeError if `pull_X_y` has not been called prior to this method when `bool_fit_per_grid` is False.

        Args:
            set (str): Specifies which dataset to use; either "train" or "test".

        Returns:
            np.ndarray: Baseline predictions corresponding to the lower resolution target data.
        """
        # We need the nan mask on a global grid setup.
        if not hasattr(self, "nan_mask") and not self.bool_fit_per_grid:
            raise RuntimeError("pull_X_y needs to be called prior to pulling baseline prediction")
        
        # The double get dictionary is simply to work through the fact that we do not integrate always y_c as a parameter
        y_LRtoHR = self._get_ydata(self.target, self.scenario_params[f"{set}_time"],
                               self.months_list, self.resolution_params.get("y_c",self.resolution_params["X_c"]),
                               self.scenario_params[f"{set}_polygon"])

        if self.bool_fit_per_grid:
            return y_LRtoHR
        

        baseline_prediction = y_LRtoHR.values.reshape(len(y_LRtoHR.valid_time) *
                                                      len(y_LRtoHR.latitude) *
                                                      len(y_LRtoHR.longitude)) 
            
        baseline_prediction = baseline_prediction[self.nan_mask]

        return baseline_prediction
    
    def _pull_single_predictor_set(self, set:str, bool_fit_per_grid:bool) -> np.ndarray:
        """
        Retrieves predictor data for a single-resolution predictor set for the specified dataset.

        Args:
            set (str): Specifies which dataset to use; either "train" or "test".
            bool_fit_per_grid (bool): Determines whether to keep the data in its original 4D shape if True, or flattened otherwise.

        Returns:
            np.ndarray: Predictor data array, either in 4D shape (time, latitude, longitude, variables) if `bool_fit_per_grid` is True, or flattened otherwise.
        """

        X = self._get_xdata(self.features,
                    years = self.scenario_params[f'{set}_time'],
                    months = self.months_list,
                    resolution=self.resolution_params.get("X_c"),
                    region = self.scenario_params[f'{set}_polygon'])
        
        if not bool_fit_per_grid:
            #len variables + elevation data (1)
            X = X.values.reshape(len(X.valid_time) * len(X.latitude) * len(X.longitude), len(self.features) + 1)

        return X
    
    def _pull_aggregated_predictors(self, set:str) -> np.ndarray:
        """
        Retrieves and aggregates predictor data with lower-resolution target data for the specified dataset.
        
        Args:
            set (str): Specifies which dataset to use; either "train" or "test".

        Returns:
            np.ndarray: Aggregated predictor data array, either in 4D shape (time, latitude, longitude, variables) if `bool_fit_per_grid` is True, or flattened otherwise.
        """
        # We set to true location based as we need the 4D matrix shape
        self.X = self._pull_single_predictor_set(set, True)

        y_lr = self._get_ydata(self.target,
                               years = self.scenario_params[f"{set}_time"],
                               months = self.months_list, 
                               resolution=self.resolution_params.get("y_c"),
                               region = self.scenario_params[f"{set}_polygon"])

        self.X = xr.concat([self.X, y_lr], dim = "variable")

        #len variables + elevation data (1) + lower resolution target `y_c` (1)
        if not self.bool_fit_per_grid:
            self.X = self.X.values.reshape(len(self.X.valid_time) * len(self.X.latitude) * len(self.X.longitude), len(self.features) + 2)

        return self.X
        
    def _pull_target(self, set:str) -> np.ndarray:
        """
        Retrieves target data for the specified dataset.

        Args:
            set (str): Specifies which dataset to use; either "train" or "test".

        Returns:
            np.ndarray: Target data array, either in 3D shape (time, latitude, longitude) if `bool_fit_per_grid` is True, or flattened otherwise.
        """
        y = self._get_ydata(self.target, 
                      years = self.scenario_params[f'{set}_time'],
                      months = self.months_list, 
                      resolution = self.resolution_params.get("y_f"),
                      region = self.scenario_params[f'{set}_polygon'])
        
        if not self.bool_fit_per_grid:
            y = y.values.reshape(len(y.valid_time) * len(y.latitude) * len(y.longitude)) 

        return y   
    
    def _load_sample_grid(self) -> xr.DataArray:
        """
        Loads a sample data higher grid based on the target variable for the first training year.

        Returns:
            xr.DataArray: Sample data grid array used for reindexing and alignment purposes.
        """
        data_grid = load_nc(f"{self.target}_{self.scenario_params['train_time'][0]}.nc", self.resolution_params.get("y_f"))
        data_grid = data_grid.sel(valid_time=data_grid.valid_time.dt.month.isin([1]))  
        return data_grid
    
    def _get_xdata(self, features:list[str], years:list[int], months:list[int], resolution:str, region:RectangularPolygon) -> xr.DataArray:
        """
        Pull feature data according to features of interest, years and
        polygon of interest

        Note: 
        We apply a reindexation of the features on a higher grid resolution to ensure X and Y shape matches throughout the experiment train/test setup.
        Elevation undergoes a specific treatment as it might come an alternative source and not be on the native grid shared by all features.
        
        Args:
            features (list[str]): list of features
            years (list[int]): list of years of interest
            months (list[int]) : list of months to filter 
            resolution (str): resolution to be loaded
            region (RectangularPolygon): Polygon of interest (min_lat,max_lat,min_lon,max_lon)
        
        Returns:
            xdata (xr.DataArray): shape(rows, len(features))
        """
        data_dict = {var: [] for var in features}

        if months is None:
            months = list(range(1,13))
        
        # Pre emptively load the sample higher grid. 
        sample_higher_grid = self._load_sample_grid()

        if region is not None:
            padded_region = self._pad_polygon(region)

            sample_higher_grid = geo_filter(sample_higher_grid, *padded_region)

        latitude_higher_grid, longitude_higher_grid = sample_higher_grid["latitude"], sample_higher_grid["longitude"]

        for var in features:
            for year in years:
                file_name = f"{var}_{year}.nc"
                
                data = load_nc(file_name, resolution)
                data = data.sel(valid_time=data.valid_time.dt.month.isin(months))
                
                if region is not None:
                    # here do a pre geographical filter. 
                    # We want to reduce data points prior to reindexing (which is quite costly.)
                    data = geo_filter(data, *padded_region)
                
                
                data_dict[var].append(data)
            
            data_dict[var] = xr.concat(data_dict[var], dim = "valid_time")

        xdata = xr.concat([data_dict[var] for var in features], dim = "variable")    

        xdata = xdata.reindex({"latitude":latitude_higher_grid,
                                "longitude":longitude_higher_grid},
                                method = "nearest")

        elevation = load_elevation(xdata.isel(variable=0), self.resolution_params.get("Z_topo", "10km"))

        xdata = xr.concat([xdata, elevation], dim="variable")

        xdata = xdata.transpose("valid_time", "latitude","longitude","variable")
        
        # Turn to NaN any values that isn't on Land. Only applies to 25km and 80km datasets
        if resolution != '10km':
            # Get missing values ie Sea values
            landseamask = ~np.isnan(sample_higher_grid.isel(valid_time=0))

            # We do not need to use landseamask of lower resolution 
            # as we just remapped to a higher grid. 
            xdata = xdata.where(landseamask)

        # Once we reindex, we can filter on the real region focus.
        xdata = geo_filter(xdata, *region) if region is not None else xdata

        return xdata
    
    def _get_ydata(self, target:str, years:list[int], months:list[int], resolution:str, region:RectangularPolygon) -> xr.DataArray:
        """
        Pull target data according to target of interest, years and
        polygon of interest
        
        Args:
            target (str): target variable
            years (List[int]): list of years of interest
            months (list[int]) : list of months to filter 
            resolution (str): resolution to be loaded
            region (RectangularPolygon): Polygon of interest (min_lat,max_lat,min_lon,max_lon)
        
        Returns:
            ydata (xr.DataArray): shape(rows, )
        """
        data_alloc = []

        if months is None:
            months = list(range(1,13))

        landseamask = load_landseamask(resolution, self.data_directory)

        sample_higher_grid = self._load_sample_grid()

        if region is not None: 
            padded_region = self._pad_polygon(region)
            sample_higher_grid = geo_filter(sample_higher_grid, *padded_region)

        latitude_higher_grid, longitude_higher_grid = sample_higher_grid["latitude"], sample_higher_grid["longitude"]

        for year in years:
            file_name = f"{target}_{year}.nc"
            
            data = load_nc(file_name, resolution)
            data = data.sel(valid_time=data.valid_time.dt.month.isin(months))

            if region is not None:
                data = geo_filter(data, *padded_region)

            data_alloc.append(data)

        ydata = xr.concat(data_alloc, dim="valid_time")
        
        if resolution != "10km":
            # Turn to NaN any values that isn't on Land. Only applies to 25km and 80km datasets
            ydata = ydata.where(landseamask)

            ydata = ydata.reindex({"latitude":latitude_higher_grid,
                                   "longitude":longitude_higher_grid},
                                    method="nearest")

            
        ydata = ydata.transpose("valid_time", "latitude","longitude")
        
        # Filter on the real polygon
        ydata = geo_filter(ydata, *region) if region is not None else ydata

        # Having downloaded the data in .nc instead of grib files, it may happen to get negative values on precipitation due to the conversion process by Copernicus.
        # Therefore we simply map out negative values to 0 if it happens.
        if target == "pr":
            ydata = xr.where(ydata < 0, 0, ydata)

        return ydata
    
    def _pad_polygon(self, polygon:tuple, delta: int = 1) -> tuple:
        """
        Expands the given polygon by a specified delta while ensuring it remains within the domain perimeter.

        Args:
            polygon (tuple): Original polygon defined as (min_lat, max_lat, min_lon, max_lon).
            delta (int, optional): The amount by which to expand the polygon boundaries. Defaults to 1.

        Returns:
            tuple: Padded polygon coordinates as (padded_min_lat, padded_max_lat, padded_min_lon, padded_max_lon).
        """
        min_lat, max_lat, min_lon, max_lon = polygon
        perimeter_polygon = RectangularPolygon(*PERIMETER_POLYGON)

        # Pad the polygon with boundary checks
        padded_min_lat = max(min_lat - delta, perimeter_polygon.min_lat)
        padded_max_lat = min(max_lat + delta, perimeter_polygon.max_lat)
        padded_min_lon = max(min_lon - delta, perimeter_polygon.min_lon)
        padded_max_lon = min(max_lon + delta, perimeter_polygon.max_lon)

        padded_polygon = (padded_min_lat, padded_max_lat, padded_min_lon, padded_max_lon)

        return padded_polygon


def load_landseamask(resolution:str, data_directory: str) -> np.ndarray:
    """
    Load land-sea masks for ERA Interim and ERA5 Hourly on single levels

    Args:
        resolution (str): Land-sea mask resolution to load
        data_directory (str): directory where the landseamasks are stored

    Returns:
        lsm (np.ndarray): Boolean numpy array; True for land, else False
    """
    if resolution == "10km":
        return None
    
    lsm = xr.open_dataset(os.path.join(data_directory,DATASET_MAPPER.get(resolution),"land_sea_mask.nc"))["lsm"]
    
    #Drop unnecessary dimension if present
    lsm = lsm.drop_vars("expver", errors="ignore")
    
    # Dimensionless, we can remove time first dimension
    lsm = lsm[0,:,:]

    # Create boolean mask for land/sea
    lsm = xr.where(lsm > 0.5, 1, np.nan)
    lsm = ~np.isnan(lsm)

    return lsm

def load_nc(file_name:str, resolution:str, data_directory = "./data") -> xr.DataArray:
    """
    Small helper function to load features stored in .nc files

    Args:
        file_name (str): feature name to be loaded
        resolution (str): resolution to be loaded
        data_directory (str): Directory where the .nc file is stored. Defaults to "./data"

    Returns:
        opened_file (xr.DataArray): .nc file according to the chosen resolution.

    """
    dataset_name = DATASET_MAPPER.get(resolution)
    feature_name = file_name.split("_")[0]

    return xr.open_dataset(os.path.join(data_directory, dataset_name, file_name))[feature_name]

def load_elevation(single_feat: xr.DataArray, resolution:str, data_directory:str = "./data") -> xr.DataArray:
    """
    Load elevation data, and duplicate it along the time dimension by taking
    into account neighbouring features with 'single_feat'
    
    Args:
        single_feat (xr.DataArray): Any .nc loaded file to use as reference for days and dimensions 
        resolution (str): resolution to be loaded
        data_directory (str): Directory where the elevation file is stored. Defaults to "./data"
        
    Returns:
        elevation_data (xr.DataArray): elevation data specific to the geo_coords
            and to the neighbouring data. shape(days, lat, lon)
    """
    
    elevation_data = xr.open_dataset(os.path.join(data_directory, DATASET_MAPPER.get(resolution), "elevation.nc"))["z"]

    # Remove the time dimension
    elevation_data = elevation_data[0,:,:]
    
    # Remap if needed from 0 to 360 -> -180 to 180
    # This is the case on ERA Land as the available data was interpolated to 0.1 x 0.1°
    elevation_data.coords['longitude'] = (elevation_data.coords['longitude'] + 180) % 360 - 180
    elevation_data = elevation_data.sortby(elevation_data.longitude)

    # We try to find the nearest elevation point
    elevation_data = elevation_data.sel(
        latitude = single_feat.latitude, 
        longitude = single_feat.longitude,
        method = "nearest")
    
    # We re-assign coords as using nearest method might interpolate or have inexact coordinates
    elevation_data = elevation_data.assign_coords(
        longitude = single_feat.longitude, 
        latitude = single_feat.latitude)
    
    # Drop sea values that were assigned due to `nearest` in .sel()
    seamask = np.isnan(single_feat.isel(valid_time=0))

    elevation_data = elevation_data.where(~seamask)

    elevation_data = elevation_data.expand_dims(valid_time=single_feat['valid_time']).transpose(*single_feat.dims)

    # Remove residual initial "time" dim if still there
    elevation_data  = elevation_data.drop_vars("time", errors="ignore")

    return elevation_data


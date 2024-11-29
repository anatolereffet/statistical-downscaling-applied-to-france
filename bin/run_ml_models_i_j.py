from src.constants import SEASON_MAPPER
from src.dataset_handlers.data_hoarder import DataHoarder
from src.metrics import compute_metrics, MetricsHoarder
from src.modeling.model_factory import model_factory
from src.modeling.scenario_handler import scenario_factory
from src.utils import filter_nan_entries

import numpy as np
from sklearn.preprocessing import StandardScaler


def main(scenario_name:str, model_name:str,target:str, years:tuple[list[int],list[int]], months:list[int], resolution_params:dict, model_kwargs:dict, data_directory:str ="./data", is_baseline:bool=False) -> None:
    """
    
    Computes for a given scenario setup location-wise model results
    The results are given averaged and with their associated standard deviation. 

    Args:
        scenario_name (str): Letter ranging from A to C 
            All scenarios have train time A and test time B
            A: Trained on Paris, tested on Paris
            B: Trained on Paris, tested on Pyrenees
            C: Trained on Paris, tested on Alpes
        model_name (str): Model acronym within ["linear_reg", "svr", "linear_svr", "rf", "gbr"]
        target (str): Target feature to trial
        years (tuple[list[int],list[int]]): Tuple with train years then test years as they are unpacked downstream in this order
        months (list[int]): List of months to filter (None to take the full year)
        resolution_params (dict): Dictionary that needs the following keys ['y_f','X_c','Z_topo','y_c'] or it will break such as 
                                      resolution_params ={"y_f":"10km",
                                                          "X_c":"25km",
                                                          "Z_topo":"50km",
                                                          "y_c":"25km} 
                                    Note however that 'y_c' is optional, the features here are X_c, Z_topo, y_c; c stands for coarse, f for fine.
        data_directory (str): Directory where the data is stored after being downloaded. Defaults to "./data"
        is_baseline (bool): If the user wishes to compute the baseline results. Defaults to False
    """
    features = [feat for feat in ["pr", "tmmn", "tmmx", "tmean", "rmin", "rmax", "10ws", "sph", "srad", "vpd"] if feat != target]
    metric_names = ["R²", "Adj-R²", "KGE", "NMSE"]

    # Boolean True if target is `pr`
    clip_precipitation_predictions = target == "pr"

    scenario_params = scenario_factory(scenario_name, *years)

    print("Scenario:", scenario_name)

    if scenario_name != "A":
        scenario_params["train_polygon"] = scenario_params["test_polygon"]

    db = DataHoarder(target, features, months, scenario_params, resolution_params, True, data_directory)

    model = model_factory(model_name, **model_kwargs)

    X_LR_historical, y_HR_historical = db.pull_X_y("train")
    X_LR, y_HR = db.pull_X_y("test")

    if is_baseline:
        model = None
        y_HR_historical_baseline = db.pull_baseline_prediction("train")
        y_HR_baseline = db.pull_baseline_prediction("test")


    # For each (i,j) at fit, we want to see which is the closest pair y_HR(i,j);y_HR_historical(i,j)

    # We need here the grid n_f x m_f
    _, lat_hr, lon_hr = y_HR_historical.shape
    
    train_metrics_allocator = MetricsHoarder(f"{'Baseline ' if is_baseline else ''}Train", metric_names)
    test_metrics_allocator = MetricsHoarder(f"{'Baseline ' if is_baseline else ''}Test", metric_names)

    for lat_hr_idx in range(lat_hr):
        for lon_hr_idx in range(lon_hr):
            # We recall the format (time, lat, lon, features)
            X_LR_ij_historical = X_LR_historical[:, lat_hr_idx, lon_hr_idx, :]
            y_HR_ij_historical = y_HR_historical[:, lat_hr_idx, lon_hr_idx]

            # Storing necessary subgrid focus 
            latitude_ij, longitude_ij = y_HR_ij_historical.latitude, y_HR_ij_historical.longitude
            
            X_LR_ij_historical, y_HR_ij_historical = X_LR_ij_historical.values, y_HR_ij_historical.values

            X_LR_ij_historical, y_HR_ij_historical, mask = filter_nan_entries(X_LR_ij_historical, y_HR_ij_historical)

            if X_LR_ij_historical is None:
                # Edge case verified, skip gridpoint
                continue 

            if not is_baseline:
                std_scaler = StandardScaler()
                X_LR_ij_historical = std_scaler.fit_transform(X_LR_ij_historical)
                model.fit(X_LR_ij_historical, y_HR_ij_historical)

                y_HR_pred_historical = model.predict(X_LR_ij_historical)

                if clip_precipitation_predictions:
                    y_HR_pred_historical = np.maximum(0, y_HR_pred_historical)
            else:
                # Apply the same mask to avoid dimension conflict and ensure we align arrays.
                y_HR_ij_historical_baseline = y_HR_historical_baseline[:, lat_hr_idx, lon_hr_idx].values
                y_HR_ij_historical_baseline = y_HR_ij_historical_baseline[mask]



            train_metrics = compute_metrics(X_LR_ij_historical.shape[1],
                                            y_HR_ij_historical,
                                            y_HR_ij_historical_baseline if is_baseline else y_HR_pred_historical)
            train_metrics_allocator.add(train_metrics)


            X_LR_ij = X_LR.sel(latitude=latitude_ij, longitude=longitude_ij, method="nearest").values
            y_HR_ij = y_HR.sel(latitude=latitude_ij, longitude=longitude_ij, method="nearest").values 

            X_LR_ij, y_HR_ij, mask = filter_nan_entries(X_LR_ij, y_HR_ij)

            if X_LR_ij is None:
                # Edge case verified, skip gridpoint 
                continue 


            if not is_baseline:
                X_LR_ij = std_scaler.transform(X_LR_ij)

                y_HR_pred = model.predict(X_LR_ij)

                if clip_precipitation_predictions:
                    y_HR_pred = np.maximum(0, y_HR_pred)
            else:
                y_HR_baseline_ij = y_HR_baseline.sel(latitude=latitude_ij, longitude=longitude_ij, method="nearest").values
                y_HR_baseline_ij = y_HR_baseline_ij[mask]
                

            
            test_metrics = compute_metrics(X_LR_ij.shape[1], y_HR_ij, y_HR_baseline_ij if is_baseline else y_HR_pred)
            test_metrics_allocator.add(test_metrics)


    print(train_metrics_allocator.summary())
    print(test_metrics_allocator.summary())

    if MINMAX:
        print(train_metrics_allocator.minmax())
        print(test_metrics_allocator.minmax())  
            

if __name__ == "__main__":
    MINMAX = True
    main("C",
        "linear_reg",
        "pr",
        ([year for year in range(2001,2015)],
         [year for year in range(2015,2019)]),
        SEASON_MAPPER.get("winter"),
        {
            "y_f":"10km",
            "X_c":"80km",
            "Z_topo":"10km",
            "y_c":"80km"
        },
          {"verbose":0,
         "n_jobs":-1,
         "random_state":42,
         "max_iter":100,
         "n_estimators":100},
         is_baseline=False
         )
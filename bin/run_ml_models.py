from src.constants import SEASON_MAPPER
from src.metrics import compute_metrics, MetricsHoarder, get_r2perbucket, abs_error
from src.dataset_handlers.data_hoarder import DataHoarder
from src.modeling.model_factory import model_factory
from src.modeling.scenario_handler import scenario_factory
from src.utils import save_results

import numpy as np 
from sklearn.preprocessing import StandardScaler


def main(scenarios: list[str], model_name:str, target:str, years: tuple[list[int],list[int]], months:list[int], resolution_params:dict, model_kwargs:dict, data_directory:str = "./data", save_pred:bool =False) -> None:
    """
    Computes for a given scenario a model and its results

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
        model_kwargs : Supplementary arguments to pass to sklearn
                       during model initialisation if needed. 
                       These need to follow sklearn concerned model convention or they won't be registered.
        data_directory (str): Directory where the data is stored after being downloaded. Defaults to "./data"
        save_pred (bool): Boolean to save predictions and ground truth. Defaults to False

    """
    print(f"Model : {model_name}")
    
    features = [feat for feat in ["pr","tmmn","tmmx","tmean","rmin","rmax","10ws","sph","srad","vpd"] if feat != target]
    metric_names = ["R²","Adj-R²","KGE", "NMSE"]

    # Boolean True if target is `pr`
    clip_precipitation_predictions = target == "pr"

    model = model_factory(model_name, **model_kwargs)
    scenario_params = scenario_factory(scenarios[0], *years)

    db = DataHoarder(target, features, months, scenario_params, resolution_params, False, data_directory)
    
    X_LRtoHR_historical, y_HR_historical = db.pull_X_y("train")

    train_metrics_allocator = MetricsHoarder("Train", metric_names)

    standard_scaler = StandardScaler()
    X_LRtoHR_historical = standard_scaler.fit_transform(X_LRtoHR_historical)
    
    model.fit(X_LRtoHR_historical, y_HR_historical)
    y_HR_pred = model.predict(X_LRtoHR_historical)

    baseline_prediction = db.pull_baseline_prediction("train")

    if clip_precipitation_predictions:
        # This is a simple feedback to see how many rows were clipped
        print(f"Affected rows: {np.sum(y_HR_pred < 0)/len(y_HR_pred):.2f}")
        y_HR_pred = np.maximum(0, y_HR_pred)
        baseline_prediction = np.maximum(0, baseline_prediction)

    train_metrics = compute_metrics(X_LRtoHR_historical.shape[1], y_HR_historical, y_HR_pred)
    train_metrics_allocator.add(train_metrics)

    baseline_metrics = compute_metrics(X_LRtoHR_historical.shape[1], y_HR_historical, baseline_prediction)
    print('Baseline is', baseline_metrics)
    
    print(train_metrics_allocator.summary())

    # Compute Delta R² between baseline and train
    print('Delta R² baseline - prediction_train', baseline_metrics["R²"] - train_metrics["R²"])

    if save_pred:
        save_results(y_HR_pred, target, model_name, "train", scenarios[0], resolution_params, months, ij=False, file_type="prediction")
        save_results(baseline_prediction, target, model_name, "train", scenarios[0], resolution_params, months, ij=False, file_type="baseline")
        save_results(y_HR_historical, target, model_name, "train", scenarios[0], resolution_params, months, ij=False, file_type="groundtruth")

    if BUCKET:
        get_r2perbucket(y_HR_historical, y_HR_pred)
        get_r2perbucket(y_HR_historical, baseline_prediction)

    for sub_scenario in scenarios:

        test_metrics_allocator = MetricsHoarder(f"Scenario {sub_scenario} Test", metric_names)
        scenario_params = scenario_factory(sub_scenario, *years)
        db = DataHoarder(target, features, months, scenario_params, resolution_params, False)

        X_LR, y_HR = db.pull_X_y("test")
        X_LR = standard_scaler.transform(X_LR)

        y_HR_pred = model.predict(X_LR)
        
        # Compare quickly how we do against a baseline prediction
        # i.e. Y_c reindexed on an HR grid.
        baseline_prediction = db.pull_baseline_prediction("test")

        if clip_precipitation_predictions:
            # This is a simple feedback to see how many rows were clipped
            print(f"Affected rows: {np.sum(y_HR_pred < 0)/len(y_HR_pred):.2f}")
            y_HR_pred = np.maximum(0, y_HR_pred)
            baseline_prediction = np.maximum(0 , baseline_prediction)

        baseline_metrics = compute_metrics(X_LR.shape[1], y_HR, baseline_prediction)
        print('Baseline is', baseline_metrics)

        test_metrics = compute_metrics(X_LR.shape[1], y_HR, y_HR_pred)
        test_metrics_allocator.add(test_metrics)
        print(test_metrics_allocator.summary())

        # Compute Delta R² between baseline and test
        print('Delta R² baseline - prediction_test =>', baseline_metrics["R²"] - test_metrics["R²"])
        
        if save_pred:
            save_results(y_HR_pred, target, model_name, "test", sub_scenario, resolution_params, months, ij=False, file_type="prediction")
            save_results(baseline_prediction, target, model_name, "test", sub_scenario, resolution_params, months, ij=False, file_type="baseline")
            save_results(y_HR, target, model_name, "test", sub_scenario, resolution_params, months, ij=False, file_type="groundtruth")

        if BUCKET:
            get_r2perbucket(y_HR, y_HR_pred)
            get_r2perbucket(y_HR, baseline_prediction)




if __name__ == "__main__":
    BUCKET = False
    main(["A","B","C"],
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
         "max_iter":4000,
         "n_estimators":100},
         save_pred = False
          )
    
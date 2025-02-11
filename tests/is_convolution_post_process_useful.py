"""

We try out here the idea of convoluting predictions output to smooth out predictions.

"""

import os

from src.metrics import kge, rsquared
from scipy.signal import convolve2d
import numpy as np 

def load_vectorized_fit_results(area, set, season, model_names):

    grid_shape = np.load(
        os.path.join("./", "results", "pred_pr", season, "location", f"test_{area}to{area}_10km_groundtruth.npy")
        )
    
    if season == "winter":
        grid_shape= grid_shape[:-1, :,:]

    gt = np.load(os.path.join(DIR_PATH_VECTORIZED_FIT, f"{set}_{area}_10km_groundtruth.npy")).reshape(-1, grid_shape.shape[0], grid_shape.shape[1])
    
    baseline = np.load(os.path.join(DIR_PATH_VECTORIZED_FIT, f"{set}_{area}_10-80_baseline.npy")).reshape(-1, grid_shape.shape[0], grid_shape.shape[1])

    paths = ["linear_reg","linear_svr","rf","gbr"]

    # Horrendous way of loading
    prediction_hoarder = {i: np.load(os.path.join(DIR_PATH_VECTORIZED_FIT,"10-80","X_c-y_c",paths[idx],f"{set}_{area}.npy")).reshape(-1, grid_shape.shape[0], grid_shape.shape[1]) for (idx,i) in enumerate(model_names[1:])}

    return gt, baseline, prediction_hoarder

def load_per_grid_fit_results(area, set, model_names):

    gt = np.load(os.path.join(DIR_PATH_PER_GRID_FIT, f"{set}_{area}to{area}_10km_groundtruth.npy"))
    
    baseline = np.load(os.path.join(DIR_PATH_PER_GRID_FIT, f"{set}_{area}to{area}_10-80_baseline.npy"))

    paths = ["linear_reg","linear_svr","rf","gbr"]

    # Horrendous way of loading
    prediction_hoarder = {i: np.load(os.path.join(DIR_PATH_PER_GRID_FIT,"10-80","X_c-y_c",paths[idx],f"{set}_{area}to{area}.npy")) for (idx,i) in enumerate(model_names[1:])}

    return gt, baseline, prediction_hoarder

def generate_kernel(size:int):
    return np.full((size,size),1/(size**2))

def compute_metrics_per_model(gt, baseline, pred_hoarder, season, model_names):

    kge_scores = {m: [] for m in model_names}
    r2_scores = {m: [] for m in model_names}
    _, lat, lon = gt.shape
    for i in range(lat):
        for j in range(lon):
            if season == "winter":
                gt_vec = gt[:-1, i, j]          # remove last day in the case where we cover winter season as we will not have this data for the last day in 2018 due to feature processing
                base_vec = baseline[:-1, i, j]
            else:
                gt_vec = gt[:,i,j]
                base_vec = baseline[:, i, j]

            # Baseline
            kge_scores["Baseline"].append(kge(gt_vec, base_vec))
            r2_scores["Baseline"].append(rsquared(gt_vec, base_vec))

            # Other models
            for submodel in model_names[1:]:
                pred_vec = pred_hoarder[submodel][:-1, i, j] if season == "winter" else pred_hoarder[submodel][:,i,j]
                kge_scores[submodel].append(kge(gt_vec, pred_vec))
                r2_scores[submodel].append(rsquared(gt_vec, pred_vec))

    for i in model_names:
        print(f"Model {i} KGE: {np.mean(kge_scores[i]):.2f}, R^2: {np.mean(r2_scores[i]):.2f}")

def main(area: str, set: str, season: str, kernel_size: int, bool_fit_per_grid: bool) -> None:

    model_names = ["Baseline","MLR","LinearSVR","RF","GBR"]

    # Define global 
    global DIR_PATH_PER_GRID_FIT
    global DIR_PATH_VECTORIZED_FIT
    
    DIR_PATH_PER_GRID_FIT = os.path.join("./","results","pred_pr",season,"location")
    DIR_PATH_VECTORIZED_FIT = os.path.join("./","results","pred_pr",season,"grid")

    if bool_fit_per_grid:
        gt, baseline, pred_hoarder = load_per_grid_fit_results(area, set, model_names)
    else:
        gt, baseline, pred_hoarder = load_vectorized_fit_results(area, set, model_names)

    # Compute metrics quickly
    compute_metrics_per_model(gt, baseline, pred_hoarder, season, model_names)

    # Create a kernel     
    kernel = generate_kernel(kernel_size)

    for i in range(gt.shape[0]):
        baseline[i,:,:] = convolve2d(baseline[i,:,:], kernel, mode="same")

        for submodel in model_names[1:]:
                pred_hoarder[submodel][i,:,:] = convolve2d(pred_hoarder[submodel][i,:,:], kernel, mode="same")

    print(f"--- After convolution metrics Kernel size {kernel_size} ---")
    compute_metrics_per_model(gt, baseline, pred_hoarder, season, model_names)

if __name__ == "__main__":
    main("A", "test","winter", kernel_size= 3, bool_fit_per_grid= True)
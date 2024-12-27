from sklearn.metrics import mean_squared_error

import numpy as np 
import xarray as xr 

def kge(y: np.ndarray, yhat: np.ndarray) -> float:
    """
    Implementation of Kling-Gupta efficiency 2012 version ensuring bias and 
    variability ratios aren't cross-correlated.
    
    This takes 1D arrays as input
    
    Args:
        y (np.ndarray): observed array
        yhat (np.ndarray): predicted array
    
    Returns:
        kge (float): Float indicator for KGE
        
    Source:
        https://doi.org/10.1016/j.jhydrol.2012.01.011
    """
    
    y_mean = np.mean(y)
    yhat_mean = np.mean(yhat)
    r_num = np.sum((yhat - yhat_mean) * (y - y_mean))
    r_denom = np.sqrt(np.sum((yhat - yhat_mean) ** 2) * np.sum((y - y_mean) ** 2))
    
    r = r_num / r_denom if r_denom != 0 else np.nan

    if yhat_mean != 0 and y_mean != 0:
        gamma = (np.std(yhat) / yhat_mean) / (np.std(y) / y_mean)
    else:
        gamma = np.nan 

    beta = yhat_mean / y_mean if y_mean != 0 else np.nan 
    
    if any(np.isnan([r, gamma, beta])):
        kge = np.nan 
    else:
        kge = 1 - np.sqrt((r - 1) ** 2 + (gamma - 1) ** 2 + (beta - 1) ** 2)

    return kge

def rsquared(y: xr.DataArray, yhat: xr.DataArray) -> float:
    """
    Implementation of rsquared
    
    Args:
        y (xr.DataArray): Vector holding target
        yhat (xr.DataArray): Vector holding predictions
        
    Returns:
        r_squared (float): metric
        
    """
    ss_residuals = sum((y-yhat)**2)
    ss_total = sum((y-np.mean(y))**2)
    
    r_squared = 1 - (float(ss_residuals)) / ss_total 
    
    return r_squared

def adjrsquared(nbfeatures:int,
                y: xr.DataArray,
                rsquared: float,
                ddof: int = 1) -> float:
    """
    Implementation of adj-rsquared with ddof
    
    Args:
        nbfeatures (int): Number of features used to train the model
        y (xr.DataArray): Vector holding target ground truth
        ddof (int): Degrees of freedom to apply Adj-Rsquared. Defaults to 1.
    
    Returns:
        r_squared (float): metric
    """
    return 1 - (1-rsquared)*(len(y) - 1) / (len(y) - nbfeatures - ddof)

def nmse(y:np.ndarray, yhat:np.ndarray) -> float:
    """
    Implementation of normalized mean squared error 

    Args:
        y (np.ndarray): Vector holding target ground truth
        yhat (np.ndarray): Vector holding predictions

    Returns:
        nmse (float): metric
    """
    mse = mean_squared_error(y, yhat)

    nmse = mse / np.var(y)

    return nmse

def compute_metrics(nbfeatures:int, y:np.ndarray, yhat:np.ndarray, ddof:int=1) -> dict:
    """
    Helper function for MetricsHoarder to compute metrics easily

    Args:
        nbfeatures (int): Number of features used to train the model
        y (np.ndarray): Vector holding target ground truth
        yhat (np.ndarray): Vector holding predictions
        ddof (int): Degrees of freedom to apply Adj-Rsquared. Defaults to 1.

    Returns:
        dictionary with rsquared, adj-rsquared and kge.
    """
    r2 = rsquared(y, yhat)
    adj_r2 = adjrsquared(nbfeatures, y, r2, ddof)
    kge_metric = kge(y, yhat)
    nmse_metric = nmse(y, yhat)
    return {"R²": r2, "Adj-R²": adj_r2, "KGE": kge_metric, "NMSE": nmse_metric}

class MetricsHoarder:
    """
    Class to enable us to compute and store metrics swiftly
    """
    def __init__(self, name:str, metric_names:list[str]) -> None:
        """
        Initializes MetricsHoarder class

        Args:
            name (str): Name of the set of metrics being stored should it be Train,Test,Val or anything else
            metric_names (list[str]): Names of the metrics being stored such as ["R²","Adj-R²"]
        """
        self.name = name
        self.metrics = {metric: [] for metric in metric_names}

    def add(self, metrics_dict:dict) -> None:
        """
        Add a set of metrics to the inner dictionary
        """
        for metric, value in metrics_dict.items():
            self.metrics[metric].append(value)

    def mean_std(self) -> dict:
        """
        Computes the mean and standard deviation of each array of stored metrics
        """
        return {metric: (np.mean(values), np.std(values)) for metric, values in self.metrics.items()}

    def summary(self) -> str:
        """
        Prints the summary metrics of the inner dictionary in a mean ± std manner.

        Returns:
            summary_str (str): String holding the printed information, therefore this can be parsed later on for our tables
        """
        mean_std_dict = self.mean_std()
        summary_str = f"{self.name} Metrics:\n"
        for metric, (mean, std) in mean_std_dict.items():
            summary_str += f"{metric}: {mean:.2f} ± {std:.2f}, "
        return summary_str
    def minmax(self):
        """
        Computes the min and maximum of each array of stored metrics

        Returns:
            summary_str (str): String holding the printed information, therefore this can be parsed later on for our tables
        """
        res_dict = {metric: (np.min(values), np.max(values)) for metric, values in self.metrics.items()}
        minmax_str = f"{self.name} Metrics:\n"
        for metric, (min, max) in res_dict.items():
            minmax_str += f"{metric} min/max: {min:.2f}/{max:.2f}, "
        return minmax_str
    

def get_r2perbucket(y:np.ndarray, yhat:np.ndarray, n_buckets:int = 6) -> None:
    """
    Place R² metrics in N buckets and prints the R² per bucket

    Args:
        y (np.ndarray): ground truth vector 
        yhat (np.ndarray): prediction vector
        n_buckets (int): Number of buckets to split R² in. Defaults to 6.

    Returns:
        None
    """
    
    bucket_min = min(np.min(y), np.min(yhat))
    bucket_max = max(np.max(y), np.max(yhat))

    bins = np.linspace(bucket_min, bucket_max + 1, n_buckets) 

    # Create masks to group data into each bucket
    masks = [(y >= bins[i]) & (y < bins[i+1]) for i in range(len(bins) - 1)]
    
    # Calculate R^2 per bucket, and handle empty buckets by setting R^2 to 0
    r2_per_bucket = [
        rsquared(y[mask], yhat[mask]) if np.any(mask) else 0 for mask in masks
    ]
    counts_per_bucket = [np.sum(mask) for mask in masks]

    if not np.sum(counts_per_bucket) == len(y):
        print('all values were not accounted for')

    counts_per_bucket /= np.sum(counts_per_bucket)
    
    for i in range(len(bins)-1):
        print(f'[{bins[i]:05.2f}-{bins[i+1]:<6.2f}]; R²:{r2_per_bucket[i]:>6.2f}; % Count:{counts_per_bucket[i]:<3.2f}')

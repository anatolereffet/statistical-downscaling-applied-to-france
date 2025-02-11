from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR, LinearSVR 


def model_factory(model_name, **kwargs):
    """
    Simple model factory enabling the user to choose algorithms to run
    
    Args:
        model_name (str): Model name to choose from ['linear_reg','linear_svr','svr','rf','gbr']
        for linear regression, support vector regression or random forest
        kwargs: We rely here on sklearn specific model arguments. 
        We filter any kwargs that can't be applied to a specific model to avoid any downstream crash.
        
    Returns:
        sklearn corresponding model object
    """
    if model_name not in ['linear_reg','linear_svr','svr','rf', 'gbr']:
        raise ValueError('model_name was not recognized.')
    model_registry = {
        "linear_reg": LinearRegression,
        "linear_svr": LinearSVR,
        "svr": SVR,
        "rf": RandomForestRegressor,
        "gbr": GradientBoostingRegressor
        }
    model_class = model_registry.get(model_name, LinearRegression)
    
    valid_params = model_class().get_params().keys()
    
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    
    return model_class(**filtered_kwargs)
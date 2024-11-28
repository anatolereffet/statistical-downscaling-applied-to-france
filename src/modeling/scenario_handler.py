from src.constants import PARIS_POLYGON as POLYGON_A
from src.constants import PYRENEES_POLYGON as POLYGON_B 
from src.constants import ALPES_POLYGON as POLYGON_C


def scenario_factory(scenario_name, timeframe_train, timeframe_test):
    """
    Simply factory to retrieve necessary parameters to run our experiments 
    
    Args:
        scenario_name (str): Scenario letter ranging from A to C
            All scenarios have train time A and test time B
                A: Trained on Paris, tested on Paris
                B: Trained on Paris, tested on Pyrenees
                C: Trained on Paris, tested on Alpes 
            
            defaults to scenario A.
        
        timeframe_train (list): list of years to consider for training 
        timeframe_test (list): list of years to consider for test
        
    Returns:
        (dict): scenario parameters 
    """
    
    if scenario_name not in ['A','B','C']:
        raise ValueError
    
    TIME_A = timeframe_train
    TIME_B = timeframe_test

    scenario_registry = {
        "A":{"train_time": TIME_A,
             "test_time" : TIME_B,
             "train_polygon": POLYGON_A,
             "test_polygon": POLYGON_A},    
        "B":{"train_time": TIME_A,
             "test_time" : TIME_B,
             "train_polygon": POLYGON_A,
             "test_polygon": POLYGON_B},
        "C":{"train_time": TIME_A,
             "test_time" : TIME_B,
             "train_polygon": POLYGON_A,
             "test_polygon": POLYGON_C},
             
        }

    return scenario_registry.get(scenario_name, scenario_registry['A'])

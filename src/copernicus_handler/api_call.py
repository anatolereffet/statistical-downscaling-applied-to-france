import os 

from src.constants import DATASET_MAPPER
from src.dataset_handlers import EraInterimApiCall,Era5LandApiCall,Era5SingleLevelsApiCall
from src.utils import RectangularPolygon

class CdsApiCall:
    """
    Global ApiCall for datasets
    """
    def __init__(self, dataset_name:str, download_parent_dir:str, year:str, features:dict, region : RectangularPolygon) -> None:
        """
        Initializes CdsApiCall class
        """
        assert dataset_name in list(DATASET_MAPPER.keys()), "Unrecognized dataset name"
        
        self.region = region
        self.era5_apiname = DATASET_MAPPER.get(dataset_name)
        self.download_parent_dir = download_parent_dir
        self.output_dir = os.path.join(os.getcwd(), self.download_parent_dir,self.era5_apiname)
        self.year = year 
        self.features = features
        os.makedirs(self.output_dir, exist_ok=True)

        self.dataset_handler = self.create_handler()

    def create_handler(self):
        handlers = {
            "reanalysis-era-interim": EraInterimApiCall,
            "reanalysis-era5-land": Era5LandApiCall,
            "reanalysis-single-levels": Era5SingleLevelsApiCall
        }

        dataset_class = handlers[self.era5_apiname]
        return dataset_class(self.output_dir, self.region)
    
    def download_data(self):
        return self.dataset_handler.download_data(self.year, self.features)
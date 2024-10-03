import os 

from src.constants import DATASET_MAPPER
from src.dataset_handlers import EraInterimProcessor, Era5LandProcessor, Era5SingleLevelsProcessor
from src.utils import RectangularPolygon

class CdsDataPreparation:
    """
    Global data preparation for each dataset
    """
    def __init__(self, dataset_name, download_parent_dir,year, features, region: RectangularPolygon) -> None:
        """
        Initializes CdsDataPreparation class
        """
        assert dataset_name in list(DATASET_MAPPER.keys()), "Unrecognized dataset name"
        
        self.region = region
        self.era5_apiname = DATASET_MAPPER.get(dataset_name)
        self.download_parent_dir = download_parent_dir
        self.output_dir = os.path.join(os.getcwd(), self.download_parent_dir,self.era5_apiname)
        self.year = year 
        self.features = features
        self.dataset_handler = self.create_handler()

    def create_handler(self):
        handlers = {
            "reanalysis-era-interim": EraInterimProcessor,
            "reanalysis-era5-land": Era5LandProcessor,
            "reanalysis-single-levels": Era5SingleLevelsProcessor,
        }

        dataset_class = handlers[self.era5_apiname]
        return dataset_class(self.download_parent_dir, self.year, self.features, self.region)
    
    def singularize_features(self):
        return self.dataset_handler._singularize_features()
    
    def create_features(self):
        return self.dataset_handler._feature_engineer()
import argparse 
import os

from src.copernicus_handler.api_call import CdsApiCall
from src.constants import DATASET_MAPPER, PERIMETER_POLYGON
from src.copernicus_handler.data_preparation import CdsDataPreparation
from src.dataset_handlers.static_features_extraction import get_elevation_data, get_geopotential_data, get_landseamasks
from src.utils import instanciate_mapping, RectangularPolygon, remove_files


def main(dataset: str, output_dir: str, start_year: int, end_year: int) -> None: 
    """
    
    Launches the download for a given dataset on a given year range in the specified directory

    Args:
        dataset (str): Dataset name to be handled from ["10km","25km","80km"]
        output_dir (str): directory name where the data should be placed 
    
    """
    era5_keys_info = [
    ("2m_dewpoint_temperature","d2m",168),
    ("2m_temperature","t2m",167),
    ("surface_solar_radiation_downwards","ssrd",169),
    ("10m_u_component_of_wind","u10",165),
    ("10m_v_component_of_wind","v10",166),
    ("surface_pressure","sp",134),
    ("total_precipitation","tp",228)
    ]

    perimeter_polygon = RectangularPolygon(*PERIMETER_POLYGON)
    era5_features_identifiers = instanciate_mapping(era5_keys_info)

    for year in range(start_year, end_year + 1):
        CdsApiCall(dataset, output_dir, year, era5_features_identifiers, perimeter_polygon)\
        .download_data()

    for year in range(start_year, end_year + 1):
        CdsDataPreparation(dataset, output_dir, year, era5_features_identifiers, perimeter_polygon)\
            .singularize_features()
        
    for year in range(start_year, end_year + 1):
        CdsDataPreparation(dataset, output_dir, year, era5_features_identifiers, perimeter_polygon)\
            .create_features()
    
    # Pulling elevation and land sea mask data if absent
    get_geopotential_data(output_dir, start_year, perimeter_polygon)
    get_landseamasks(output_dir, start_year, perimeter_polygon)
    get_elevation_data(output_dir)

    # remove the initial files having processed their final version
    for year in range(start_year, end_year+1):
        remove_files(os.path.join(output_dir,DATASET_MAPPER.get(dataset)), year, "tp","t2m","d2m","sp","ssrd","u10","v10")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--dataset",
        choices = ["10km","25km","80km"],
        default = "80km",
        help = "Choosing which Era5 dataset to pull (default is '25km')."
    )

    parser.add_argument(
        "-o","--output_dir",
        default = "./data",
        help = "Parent directory where files will be stored (default is './data')."
    )

    parser.add_argument(
        "--start_year",
        type=int,
        required=True,
        help="Start year of the data to download and process"
    )
    
    parser.add_argument(
        "--end_year",
        type=int,
        required=True,
        help="End year of the data to download and process"
    )

    args = parser.parse_args()

    main(args.dataset, args.output_dir, args.start_year, args.end_year)
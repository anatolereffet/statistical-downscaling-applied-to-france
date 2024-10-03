import argparse 

from src.copernicus_handler.api_call import CdsApiCall
from src.dataset_handlers.geopotential_height_extraction import get_elevation_data, get_geopotential_data
from src.copernicus_handler.data_preparation import CdsDataPreparation
from src.utils import instanciate_mapping, RectangularPolygon

def main(dataset: str, output_dir: str, start_year: int, end_year: int): 

    era5_keys_info = [
    ("2m_dewpoint_temperature","d2m",168),
    ("2m_temperature","t2m",167),
    ("surface_solar_radiation_downwards","ssrd",169),
    ("10m_u_component_of_wind","u10",165),
    ("10m_v_component_of_wind","v10",166),
    ("surface_pressure","sp",134),
    ("total_precipitation","tp",228)
    ]


    era5_features_identifiers = instanciate_mapping(era5_keys_info)

    france_polygon = RectangularPolygon(-6.84, 13.71, 41.11, 51.4)

    # Pulling elevation data.
    get_geopotential_data(output_dir, start_year, france_polygon)
    get_elevation_data(output_dir)

    for year in range(start_year, end_year + 1):
        CdsApiCall(dataset, output_dir, year, era5_features_identifiers, france_polygon)\
        .download_data()
    
    for year in range(start_year, end_year + 1):
        CdsDataPreparation(dataset, output_dir, year, era5_features_identifiers, france_polygon)\
            .singularize_features()
        
    for year in range(start_year, end_year + 1):
        CdsDataPreparation(dataset, output_dir, year, era5_features_identifiers, france_polygon)\
            .create_features()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--dataset",
        choices = ["10km","25km","80km"],
        default = "10km",
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
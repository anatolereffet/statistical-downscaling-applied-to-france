# Statistical downscaling applied to France

# Quickstart 

You'll need a CDS [API key](https://cds.climate.copernicus.eu/how-to-api) according to your OS

Follow the CDS documentation, once you have your CDS API key setup in ~/.cdsapirc you need to accept the licenses of each dataset while being logged into CDS. These can be found on each dataset page right at the end:
- [ERA Interim](https://cds.climate.copernicus.eu/datasets/reanalysis-era-interim?tab=d_download)
- [ERA single levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download)
- [ERA Land](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=download)

Once you have all the CDS part set up, create a virtual environment, the project has been tested on **_Python 3.9.7_**
```bash
python3 -m venv downscaling_env
source downscaling_env/bin/activate  # On Windows, use: downscaling_env\Scripts\activate
```

Install requirements and run the download script with the desired options, expect it to be slow as CDS has limited resources for all users requesting data on a daily basis.
```bash
pip install -r requirements.txt
python3 bin/pulling_data.py --dataset "25km" -o "./data" --start_year 2000 --end_year 2001
```
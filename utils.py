from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import os
os.environ['KAGGLE_USERNAME'] = 'itsikedrisalesmen'
os.environ['KAGGLE_KEY'] = '2c3138465ece45b1f9adf1f2d455a0a6'

class Utils:
    @staticmethod
    def load_csv(path):
        """Load a CSV file and return a DataFrame."""
        return pd.read_csv(path)

    @staticmethod
    def download_kaggle_dataset(dataset_url: str, download_path: str = "datasets", unzip: bool = True):
        """
        Downloads a Kaggle dataset using the Kaggle API.

        Parameters:
        - dataset_url: str → e.g., "sebastianwillmann/beverage-sales"
        - download_path: str → folder to download the dataset into
        - unzip: bool → whether to unzip the dataset after downloading
        """
        os.makedirs(download_path, exist_ok=True)

        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(dataset_url, path=download_path, unzip=unzip)

        print(f"✅ Dataset downloaded to: {os.path.abspath(download_path)}")
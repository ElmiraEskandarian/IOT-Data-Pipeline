from pathlib import Path
import logging
from kaggle.api.kaggle_api_extended import KaggleApi

logger = logging.getLogger(__name__)

def download_kaggle_dataset(dataset_url: str, save_path: Path):
    try:
        logger.info(f"Downloading dataset from Kaggle: {dataset_url}")
        dataset_identifier = dataset_url.split("/datasets/")[-1]

        api = KaggleApi()
        api.authenticate()

        api.dataset_download_files(
            dataset_identifier,
            path=save_path.parent,
            unzip=True
        )

        logger.info(f"Dataset downloaded to {save_path.parent}")

    except Exception as e:
        logger.error(f"Error downloading Kaggle dataset: {str(e)}")
        raise
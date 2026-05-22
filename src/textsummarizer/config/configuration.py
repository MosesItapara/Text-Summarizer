from src.textsummarizer.entity import DataIngestionConfig
from src.textsummarizer.utils.common import read_yaml, create_directories
from src.textsummarizer.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from pathlib import Path

CONFIG_FILE_PATH: Path = Path("config/config.yaml")
PARAMS_FILE_PATH: Path = Path("params.yaml")

class ConfigurationManager:
    def __init__(self):
        self.config = read_yaml(CONFIG_FILE_PATH)
        self.params = read_yaml(PARAMS_FILE_PATH)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        data_ingestion_config = self.config.artifacts.data_ingestion
        return DataIngestionConfig(
            root_dir=data_ingestion_config.root_dir,
            ingested_train_dir=data_ingestion_config.ingested_train_dir,
            ingested_test_dir=data_ingestion_config.ingested_test_dir,
            dataset_name=data_ingestion_config.dataset_name,
        )

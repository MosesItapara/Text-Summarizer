from src.textsummarizer.entity import DataIngestionConfig, DataTransformationConfig
from src.textsummarizer.utils.common import read_yaml, create_directories
from src.textsummarizer.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from pathlib import Path
import os

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

    def get_data_transformation_config(self) -> DataTransformationConfig:
        cfg    = self.config.artifacts.data_transformation
        params = self.params.data_transformation
        ing    = self.config.artifacts.data_ingestion
        os.makedirs(cfg.root_dir, exist_ok=True)
        return DataTransformationConfig(
            root_dir=Path(cfg.root_dir),
            ingested_train_dir=Path(ing.ingested_train_dir),
            ingested_test_dir=Path(ing.ingested_test_dir),
            tokenizer_name=cfg.tokenizer_name,
            input_column=params.input_column,
            target_column=params.target_column,
            max_input_length=params.max_input_length,
            max_target_length=params.max_target_length,
         )
    

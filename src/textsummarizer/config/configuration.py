from src.textsummarizer.entity import DataIngestionConfig, DataTransformationConfig, ModelTrainerConfig
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
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        cfg = self.config.artifacts.model_trainer
        trans = self.config.artifacts.data_transformation
        params = self.params.model_trainer
        os.makedirs(cfg.root_dir, exist_ok=True)
        return ModelTrainerConfig(
            root_dir=Path(cfg.root_dir),
            transformed_train_dir=Path(trans.root_dir) / 'transformed_train',
            transformed_test_dir=Path(trans.root_dir) / 'transformed_test',
            model_name=params.model_name,
            num_train_epochs=params.num_train_epochs,
            warmup_steps=params.warmup_steps,
            per_device_train_batch_size=params.per_device_train_batch_size,
            per_device_eval_batch_size=params.per_device_eval_batch_size,
            weight_decay=params.weight_decay,
            logging_steps=params.logging_steps,
            evaluation_strategy=params.evaluation_strategy,
            eval_steps=params.eval_steps,
            save_steps=params.save_steps,
            gradient_accumulation_steps=params.gradient_accumulation_steps
        )
    

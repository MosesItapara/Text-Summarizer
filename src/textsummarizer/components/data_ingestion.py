from datasets import load_dataset
from src.textsummarizer.entity import DataIngestionConfig
from src.textsummarizer.logging import logger

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_and_save(self):
        dataset = load_dataset(self.config.dataset_name)
        dataset["train"].to_csv(self.config.ingested_train_dir)
        dataset["test"].to_csv(self.config.ingested_test_dir)
        print(f"Train: {len(dataset['train'])} records")
        print(f"Test: {len(dataset['test'])} records")
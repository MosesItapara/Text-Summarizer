from datasets import load_dataset
from src.textsummarizer.entity import DataIngestionConfig
from src.textsummarizer.logging import logger

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_and_save(self):
        dataset = load_dataset(self.config.dataset_name)

        # Save as Arrow format — NOT csv
        dataset['train'].save_to_disk(str(self.config.ingested_train_dir))
        dataset['test'].save_to_disk(str(self.config.ingested_test_dir))

        logger.info(f"Train: {len(dataset['train'])} | Test: {len(dataset['test'])}")
        return dataset
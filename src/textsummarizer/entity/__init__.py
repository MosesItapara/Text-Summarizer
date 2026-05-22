from dataclasses import dataclass

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: str
    ingested_train_dir: str
    ingested_test_dir: str
    dataset_name: str


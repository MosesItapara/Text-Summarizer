from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    ingested_train_dir: Path
    ingested_test_dir: Path
    dataset_name: str

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    ingested_train_dir: Path
    ingested_test_dir: Path
    tokenizer_name: str
    input_column: str
    target_column: str
    max_input_length: int
    max_target_length: int




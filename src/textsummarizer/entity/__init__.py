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

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: str
    model_name: str
    num_train_epochs: int
    warmup_steps: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    weight_decay: float
    logging_steps: int
    evaluation_strategy: str
    eval_steps: int
    save_steps: int
    gradient_accumulation_steps: int



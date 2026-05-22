from src.textsummarizer.logging import logger
from src.textsummarizer.pipeline.stage_1_data_ingestion_pipeline import DataIngestionPipeline
from src.textsummarizer.pipeline.stage_2_data_transformation_pipeline import DataTransformationPipeline
from src.textsummarizer.pipeline.stage_3_model_trainer_pipeline import ModelTrainerTrainingPipeline


STAGE_NAME = "Data Ingestion Stage"

if __name__ == "__main__":
    logger.info(f"Starting {STAGE_NAME}...")
    pipeline = DataIngestionPipeline()
    pipeline.initiate_data_ingestion()

STAGE_NAME = "Data Transformation Stage"

if __name__ == "__main__":
    logger.info(f"Starting {STAGE_NAME}...")
    pipeline = DataTransformationPipeline()
    pipeline.initiate_data_transformation()

STAGE_NAME = "Model Trainer Stage"

if __name__ == "__main__":
    logger.info(f"Starting {STAGE_NAME}...")
    pipeline = ModelTrainerTrainingPipeline()
    pipeline.initiate_model_trainer()
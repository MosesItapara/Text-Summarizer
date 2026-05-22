from src.textsummarizer.logging import logger
from src.textsummarizer.pipeline.stage_1_data_ingestion_pipeline import DataIngestionPipeline

STAGE_NAME = "Data Ingestion Stage"

if __name__ == "__main__":
    logger.info(f"Starting {STAGE_NAME}...")
    pipeline = DataIngestionPipeline()
    pipeline.initiate_data_ingestion()
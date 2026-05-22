from src.textsummarizer.config.configuration import ConfigurationManager
from src.textsummarizer.components.data_ingestion import DataIngestion
from src.textsummarizer.logging import logger

class DataIngestionPipeline:
    def __init__(self):
        pass

    def initiate_data_ingestion(self):
        try:
            config_manager = ConfigurationManager()
            data_ingestion_config = config_manager.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)
            data_ingestion.download_and_save()
        except Exception as e:
            logger.exception(f"Error in data ingestion: {e}")
            raise e
from src.textsummarizer.config.configuration import ConfigurationManager
from src.textsummarizer.components.data_transformation import DataTransformation
from src.textsummarizer.logging import logger

class DataTransformationPipeline:
    def __init__(self):
        pass

    def initiate_data_transformation(self):
        try:
            config_manager = ConfigurationManager()
            transform_config = config_manager.get_data_transformation_config()
            transformation = DataTransformation(config=transform_config)
            train_ds, test_ds = transformation.transform()
        except Exception as e:
            raise e
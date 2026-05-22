from src.textsummarizer.config.configuration import ConfigurationManager
from src.textsummarizer.components.model_evaluation import ModelEvaluation
from src.textsummarizer.logging import logger

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def initiate_model_trainer(self):
        config_manager = ConfigurationManager()
        model_evaluation_config = config_manager.get_data_model_evaluation_config()
        model_evaluation_config = ModelEvaluation(config=model_evaluation_config)
        model_evaluation_config.evaluate()
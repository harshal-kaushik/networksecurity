from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.model_trainer import ModelTrainer
from networksecurity.entity.artifact_entity import DataIngestionArtifact
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataIngestionConfig
import sys
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.entity.config_entity import TrainingPipelineConfig,DataIngestionConfig,DataValidationConfig,DataTransformationConfig,ModelTrainerConfig
from networksecurity.components.data_validation import DataValidation
if __name__ == '__main__':
    try:
        training_pipline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipline_config=training_pipline_config)
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        logging.info("Data Ingestion Started")
        data_ingestion_artifact=data_ingestion.initate_data_ingestion()
        logging.info("Data ingestion completed")
        print(data_ingestion_artifact)
        data_validation_config=DataValidationConfig(training_pipline_config=training_pipline_config)

        data_validation = DataValidation(
            data_ingestion_artifact=data_ingestion_artifact,
            data_validation_config=data_validation_config
        )
        logging.info("Data Validation Started")
        data_validation_artifact=data_validation.initate_data_validation()
        logging.info("Data Validation completed")
        data_transformation_config=DataTransformationConfig(training_pipeline_config=training_pipline_config)
        data_transformation = DataTransformation(
            data_transformation_config=data_transformation_config,
            data_validation_artifact=data_validation_artifact,
        )
        logging.info("Data Transformation Started")
        data_transformation_artifact=data_transformation.initate_data_transformation()
        logging.info("Data Transformation completed")
        print(data_transformation_artifact)

        logging.info("Model Training Started")
        model_trainer_config = ModelTrainerConfig(training_pipeline_config=training_pipline_config)
        model_trainer = ModelTrainer(model_trainer_config=model_trainer_config,data_transformation_artifact=data_transformation_artifact)
        model_trainer_artifact = model_trainer.initate_model_trainer()

        logging.info("Model Training Completed")


    except Exception as e:
        raise NetworkSecurityException(e,sys)



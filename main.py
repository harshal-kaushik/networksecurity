from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.entity.artifact_entity import DataIngestionArtifact
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataIngestionConfig
import sys
from networksecurity.entity.config_entity import TrainingPipelineConfig

if __name__ == '__main__':
    try:
        training_pipline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipline_config=training_pipline_config)
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        logging.info("Data Ingestion Started")
        data_ingestion_artifict=data_ingestion.initate_data_ingestion()
        print(data_ingestion_artifict)
    except Exception as e:
        raise NetworkSecurityException(e,sys)



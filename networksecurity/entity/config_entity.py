from datetime import datetime
import os
from networksecurity.constant import training_pipline

print(training_pipline.PIPLINE_NAME)
print(training_pipline.ARTIFACT_DIR)

class TrainingPipelineConfig:
    def __init__(self):
        self.timestamp = datetime.now()
        self.pipeline_name = training_pipline.PIPLINE_NAME
        self.artifact_dir = training_pipline.ARTIFACT_DIR
        self.artifact_dir_with_timestamp = os.path.join(self.artifact_dir, str(self.timestamp))
class DataIngestionConfig:
    def __init__(self,training_pipline_config:TrainingPipelineConfig):
        self.data_ingestion_dir:str=os.path.join(
            training_pipline_config.artifact_dir,training_pipline.data_ingestion_dir_name,
        )
        self.feature_store_file_path = os.path.join(
            self.data_ingestion_dir,
            training_pipline.data_ingestion_feature_store_dir,
            training_pipline.FILE_NAME
        )
        self.train_file_path = os.path.join(
            self.data_ingestion_dir,
            training_pipline.data_ingestion_ingested_dir,
            training_pipline.train_file_name
        )
        self.test_file_path = os.path.join(
            self.data_ingestion_dir,
            training_pipline.data_ingestion_ingested_dir,
            training_pipline.test_file_name
        )
        self.train_test_split_ratio: float = training_pipline.data_ingestion_train_test_split_ratio
        self.collection_name : str= training_pipline.DATA_INGESTION_COLLECTION_NAME
        self.database_name : str=training_pipline.data_ingestion_database_name
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

class DataValidationConfig:
    def __init__(self,training_pipline_config:TrainingPipelineConfig):
        self.data_validation_dir:str=os.path.join(training_pipline_config.artifact_dir,training_pipline.DATA_VALIDATION_DIR_NAME)
        self.valid_data_dir: str = os.path.join(self.data_validation_dir,
                                                     training_pipline.VALID_DIR_NAME)
        self.invalid_data_dir: str = os.path.join(self.data_validation_dir,
                                                training_pipline.INVALID_DIR_NAME)
        self.invalid_data_dir: str = os.path.join(self.data_validation_dir,
                                                  training_pipline.INVALID_DIR_NAME)
        self.valid_train_file_path:str = os.path.join(self.valid_data_dir,training_pipline.VALID_DIR_NAME)
        self.valid_test_file_path:str = os.path.join(self.valid_data_dir,training_pipline.VALID_DIR_NAME)

        self.invalid_train_file_path:str = os.path.join(self.valid_data_dir, training_pipline.INVALID_DIR_NAME)
        self.invalid_test_file_path:str = os.path.join(self.valid_data_dir, training_pipline.INVALID_DIR_NAME)

        self.drift_report_file_path:str = os.path.join(self.data_validation_dir,training_pipline.DATA_DRIFT_REPORT_DIR_NAME,training_pipline.DATA_DRIFT_REPORT_FILE_NAME
                                                       )


class DataTransformationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_transformation_dir: str = os.path.join( training_pipeline_config.artifact_dir,training_pipline.DATA_TRANSFORMATION_DIR_NAME )
        self.transformed_train_file_path: str = os.path.join( self.data_transformation_dir,training_pipline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            training_pipline.train_file_name.replace("csv", "npy"),)
        self.transformed_test_file_path: str = os.path.join(self.data_transformation_dir,  training_pipline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            training_pipline.test_file_name.replace("csv", "npy"), )
        self.transformed_object_file_path: str = os.path.join( self.data_transformation_dir, training_pipline.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
            training_pipline.PREPROCESSING_OBJECT_FILE_NAME)
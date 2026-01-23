import os
import sys
import numpy as np
import pandas as pd

TARGET_COLUMN = 'CLASS_LABEL'
PIPLINE_NAME : str = 'Phising Data End-to-End Ml ops project'
ARTIFACT_DIR : str = 'Artifacts'
FILE_NAME : str='Phishing_Legitimate_full.csv'


SAVED_MODEL_DIR= os.path.join("saved_models")
model_file_name = 'model.pkl'


train_file_name : str='train.csv'
test_file_name : str='test.csv'

DATA_INGESTION_COLLECTION_NAME :str = "Network_Data"
data_ingestion_database_name : str='project_data'
data_ingestion_dir_name : str='data_ingestion'
data_ingestion_feature_store_dir:str='feature_store'
data_ingestion_ingested_dir:str='ingested'
data_ingestion_train_test_split_ratio : float=0.2

schema_file_path = os.path.join('data_schema','schema.yaml')
"""
Data Validation Related Constants 
"""
DATA_VALIDATION_DIR_NAME :str= "data_validation"
VALID_DIR_NAME :str= "validated"
INVALID_DIR_NAME :str ="invalid"
DATA_DRIFT_REPORT_DIR_NAME :str= "data_drift_report"
DATA_DRIFT_REPORT_FILE_NAME :str= "report.yaml"
PREPROCESSING_OBJECT_FILE_NAME:str="preprocessing.pkl"


"""
Data Transformation related constant start with DATA_TRANSFORMATION VAR NAME
"""

DATA_TRANSFORMATION_DIR_NAME : str='data_transformation'
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR : str='transformation'
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR : str='transformed_object'

## knn imputer to replace nan values
DATA_TRANSFORMATION_IMPUTER_PARAMS : dict = {
    "missing_values": np.nan,
    "n_neighbors":3,
    "weights":'uniform',
}

Data_Transformation_Train_File_Path:str='train.npy'
Data_Transformation_Test_File_Path:str='test.npy'

"""
Model Trainer realted constant
"""
model_trainer_dir_name:str="model_trainer"
model_trainer_trained_model_dir:str="trained_model"
model_trainer_trained_model_name:str="model.pkl"
model_trainer_expected_score :float=0.6
model_trainer_over_fitting_under_fitting_threshold : float=0.05


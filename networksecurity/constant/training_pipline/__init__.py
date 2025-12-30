import os
import sys
import numpy as np
import pandas as pd

TARGET_COLUMN = 'target'
PIPLINE_NAME : str = 'Phising Data End-to-End Ml ops project'
ARTIFACT_DIR : str = 'Artifacts'
FILE_NAME : str='Phishing_Legitimate_full.csv'

train_file_name : str='train.csv'
test_file_name : str='test.csv'

DATA_INGESTION_COLLECTION_NAME :str = "Network_Data"
data_ingestion_database_name : str='project_data'
data_ingestion_dir_name : str='data_ingestion'
data_ingestion_feature_store_dir:str='feature_store'
data_ingestion_ingested_dir:str='ingested'
data_ingestion_train_test_split_ratio : float=0.2
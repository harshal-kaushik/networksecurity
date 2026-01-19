import pandas as pd

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging


## configuration for data ingestion config

from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact
import numpy as np
import os
import sys
import pymongo
from typing import List
from sklearn.model_selection import train_test_split

from dotenv import load_dotenv
load_dotenv()

MONGO_URI = os.getenv('MONGO_DB_URL')

class DataIngestion:
    def __init__(self,data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(e,sys)

    def export_collection_as_dataframe(self):
        try:
            databasename=self.data_ingestion_config.database_name
            collectionname=self.data_ingestion_config.collection_name
            self.mongo_client = pymongo.MongoClient(MONGO_URI)
            collection = self.mongo_client[databasename][collectionname]

            df=pd.DataFrame(list(collection.find()))
            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"],axis=1)

            df.replace({"na":np.nan},inplace=True)
            return df
        except Exception as e:
            raise NetworkSecurityException(e,sys)
    def initate_data_ingestion(self):

        try:
            dataframe=self.export_collection_as_dataframe()
            dataframe=self.export_data_to_feature_store(dataframe)
            train_file_path,test_file_path=self.split_data_as_train_test(dataframe)

            data_ingestion_artifact = DataIngestionArtifact(
                train_file_path=train_file_path,
                test_file_path=test_file_path,
            )
            return data_ingestion_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)

    def export_data_to_feature_store(self, df):
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            os.makedirs(os.path.dirname(feature_store_file_path), exist_ok=True)
            df.to_csv(feature_store_file_path, index=False, header=True)
            return df
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def split_data_as_train_test(self, df):
        try:
            train_set, test_set = train_test_split(
                df,
                test_size=self.data_ingestion_config.train_test_split_ratio,train_size=0.8,
                random_state=42
            )
            logging.info("Performed train_test_split.")
            train_file_path = self.data_ingestion_config.train_file_path
            test_file_path = self.data_ingestion_config.test_file_path
            os.makedirs(os.path.dirname(train_file_path), exist_ok=True)
            train_set.to_csv(train_file_path, index=False, header=True)
            test_set.to_csv(test_file_path, index=False, header=True)
            return train_file_path, test_file_path
        except Exception as e:
            raise NetworkSecurityException(e, sys)
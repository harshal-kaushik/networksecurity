import os
import sys

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from networksecurity.constant.training_pipline import TARGET_COLUMN
from networksecurity.constant.training_pipline import DATA_TRANSFORMATION_IMPUTER_PARAMS

from networksecurity.entity.artifact_entity import DataTransformationArtifact,DataValidationArtifact
from networksecurity.logging.logger import logging
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.utils.main_utils.utils import save_numpy_array_data,save_object
from networksecurity.entity.config_entity import DataTransformationConfig

# data transformation pipline

class DataTransformation:
    def __init__(self,data_validation_artifact:DataValidationArtifact,
                 data_transformation_config:DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise NetworkSecurityException(e,sys)
    @staticmethod
    def read_data(file_path):
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e,sys)

    def get_data_transformer_object(cls)->Pipeline:

        logging.info("Emtered the get_trandsformer object of transofrmation classs")
        try:
            imputer=KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            processor:Pipeline = Pipeline([('imputer',imputer)])

            return processor

        except Exception as e:
            raise NetworkSecurityException(e,sys)
    def initate_data_transformation(self)->DataTransformationArtifact:
        logging.info("Initializing Data Transformation Artifact")
        try:
            logging.info("Starting Data Transformation")
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df= DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)


            ## training dataframe
            input_feature_train_df= train_df.drop(columns=[TARGET_COLUMN],axis=1)
            target_feature_train_df= train_df[TARGET_COLUMN]

            ## test dataframe
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            preprocessor = self.get_data_transformer_object()

            preprocessed_object = preprocessor.fit(input_feature_train_df)
            transformed_input_train_features=preprocessor.transform(input_feature_train_df)
            transformed_input_test_features=preprocessor.transform(input_feature_test_df)

            train_arr = np.c_[transformed_input_train_features,np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_test_features, np.array(target_feature_test_df)]

            #save numpy array data
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path,array=train_arr,)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path,array=test_arr,)

            save_object(self.data_transformation_config.transformed_object_file_path,preprocessed_object)

            #preparing artifacts
            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path

            )
            return data_transformation_artifact


        except Exception as e:
            raise NetworkSecurityException(e,sys)
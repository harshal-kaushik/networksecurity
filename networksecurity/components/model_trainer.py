import os
import sys
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact

from networksecurity.utils.main_utils.utils import save_object,load_object,load_numpy_array_data,evaluate_model
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score
import mlflow

# importing ml liberaries
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier)
class ModelTrainer():
    def __init__(self,model_trainer_config:ModelTrainerConfig,data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact

        except Exception as e:
            raise NetworkSecurityException(e,sys)
    def track_mlfolow(self,best_model,classificationmartic):
        try:
            with mlflow.start_run():
                f1_score = classificationmartic.f1_score
                precision = classificationmartic.precision_score
                recall = classificationmartic.recall_score

                mlflow.log_metric("f1_score",f1_score)
                mlflow.log_metric("precision",precision)
                mlflow.log_metric("recall",recall)
                mlflow.sklearn.log_model(best_model,"model")
        except Exception as e:
            raise NetworkSecurityException(e,sys)

    def train_model(self,x_train,y_train,x_test,y_test):
        model = {
            "RandomForestClassifier" : RandomForestClassifier(verbose=1),
            "GradientBoostingClassifier" : GradientBoostingClassifier(verbose=1),
            "DecisionTreeClassifier" : DecisionTreeClassifier(),
            "LogisticRegression" : LogisticRegression(verbose=1),
            "AdaBoostClassifier" : AdaBoostClassifier(),
        }

        model_param_grid = {

            "RandomForestClassifier": {
                "n_estimators": [8,16,32,64,128,256],
                # "criterion": ["gini", "entropy", "log_loss"],
                # "max_depth": [None, 5, 10, 20],
                # "min_samples_split": [2, 5, 10],
                # "min_samples_leaf": [1, 2, 4],
                # "max_features": ["sqrt", "log2", None],
                # "bootstrap": [True, False],
                # "class_weight": [None, "balanced"],
                # "random_state": [42]
            },

            "GradientBoostingClassifier": {
                "n_estimators": [8,16,32,64,128,256],
                "learning_rate": [0.01, 0.05, 0.1],
                # "max_depth": [3, 5, 7],
                # "min_samples_split": [2, 5, 10],
                # "min_samples_leaf": [1, 2, 4],
                "subsample": [0.6,0.7,0.8,0.9, 1.0],
                # "max_features": ["sqrt", "log2", None],
                # "random_state": [42]
            },

            "DecisionTreeClassifier": {
                "criterion": ["gini", "entropy", "log_loss"],
                # "max_depth": [None, 5, 10, 20],
                # "min_samples_split": [2, 5, 10],
                # "min_samples_leaf": [1, 2, 4],
                # "max_features": ["sqrt", "log2", None],
                # "class_weight": [None, "balanced"],
                # "random_state": [42]
            },


            "LogisticRegression": {
                "penalty": ["l1", "l2", "elasticnet", None],
                # "C": [0.01, 0.1, 1, 10],
                # "solver": ["lbfgs", "liblinear", "saga"],
                # "max_iter": [100, 200, 500],
                # "class_weight": [None, "balanced"],
                # "l1_ratio": [0.0, 0.5, 1.0],  # only for elasticnet
                # "random_state": [42]
            },

            "AdaBoostClassifier": {
                "n_estimators": [8,16,32,64,128,256],
                "learning_rate": [0.01, 0.05, 0.1, 1.0],
            }

        }

        model_report : dict = evaluate_model(x_train= x_train,y_train= y_train,x_test=x_test,y_test=y_test,models=model,param=model_param_grid)

        # best model score from the dict
        best_model_report = max(sorted(model_report.values()))

        # best model score name from the dict

        best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_report)]

        best_model = model[best_model_name]
        y_train_pred = best_model.predict(x_train)

        classification_train_metric=get_classification_score(y_true=y_train,y_pred=y_train_pred)

        ## Track the mlflow

        self.track_mlfolow(best_model,classification_train_metric)




        y_test_pred = best_model.predict(x_test)
        classification_test_metric = get_classification_score(y_true=y_test,y_pred=y_test_pred)
        self.track_mlfolow(best_model, classification_test_metric)
        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path, exist_ok=True)

        Network_Model = NetworkModel(preprocessor=preprocessor,model=best_model)
        save_object(self.model_trainer_config.trained_model_file_path,obj=Network_Model)

        # Model Trainer Artifact

        ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                             train_metric_artifact=classification_train_metric,
                             test_metric_artifact=classification_test_metric,)

        logging.info("Model trained successfully")
        logging.info(f"Model trainer artifact:{ModelTrainerArtifact}")

        return ModelTrainerArtifact




    def initate_model_trainer(self)->ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            # loading training array and test array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train,y_train,x_test,y_test = (
                train_arr[:,:-1],train_arr[:,-1],test_arr[:,:-1],test_arr[:,-1]

            )

            model_trainer_artifact = self.train_model(x_train,y_train,x_test,y_test)

            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e,sys)
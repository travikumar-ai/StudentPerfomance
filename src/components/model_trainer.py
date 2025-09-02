import os
import sys
from dataclasses import dataclass

from src.utills import logger
import logging
from src.utills.exception import CustomException

import mlflow
from mlflow.models import infer_signature

#--- remote ml flow-------
import dagshub

#-----models-----
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor,
                              AdaBoostRegressor,
                              GradientBoostingRegressor)

from xgboost import XGBRegressor

from catboost import CatBoostRegressor


from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error
from sklearn.metrics import r2_score

#-------save the model----
from src.utills.utilities import save_object, evaluate_models

dagshub.init(repo_owner='travikumar3456',
             repo_name='StudentPerfomance', mlflow=True)


@dataclass
class ModelTrainingConfig:
    trained_model_file_path = r'src/artifacts'
    

class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainingConfig()
        
    def intiate_model_trainer(self, train_array, test_array):
        
        try:
            logging.info("Split training and test input data")
            
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            
            models = {
                'LinearRegression': LinearRegression() ,
                'KNeighborsRegressor': KNeighborsRegressor(),
                'DecisionTreeRegressor': DecisionTreeRegressor() ,
                'GradientBoostingRegressor': GradientBoostingRegressor() ,
                'AdaBoostRegressor': AdaBoostRegressor(),
                'RandomForestRegressor': RandomForestRegressor(),
                'XGBRegressor': XGBRegressor(),
                'CatBoostRegressor': CatBoostRegressor(),
                }
            
            params = {
                        'LinearRegression':{},
                        'KNeighborsRegressor': {
                                                    'n_neighbors':[3,5,7], 
                                                    'weights':['uniform', 'distance'],
                                                    'algorithm': ['ball_tree', 'kd_tree', 'brute']
                                                    },
                        'DecisionTreeRegressor': {
                                                    'criterion':['squared_error',
                                                                    'friedman_mse', 'absolute_error']
                                                    
                                                                            },
                        'GradientBoostingRegressor': {
                                                        # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                                                        'learning_rate': [.1, .01, .05, .001],
                                                        'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                                                        'criterion':['squared_error', 'friedman_mse'],
                                                        # 'max_features':['auto','sqrt','log2'],
                                                        'n_estimators': [8, 16, 32, 64, 128, 256]
                                                        },
                        'AdaBoostRegressor': {
                                                    'learning_rate': [.1, .01, 0.5, .001],
                                                    # 'loss':['linear','square','exponential'],
                                                    'n_estimators': [8, 16, 32, 64, 128, 256]
                                                },
                        'RandomForestRegressor': {
                                                    'criterion': ['squared_error',
                                                                'friedman_mse', 'absolute_error']

                                                }, 
                        'XGBRegressor':  {
                                                'learning_rate': [.1, .01, .05, .001],
                                                'n_estimators': [8, 16, 32, 64, 128, 256]
                                            }, 
                        'CatBoostRegressor':  {
                                                    'depth': [6, 8, 10],
                                                    'iterations': [30, 50, 100]
                                                },
                }
            model_train_report, model_test_report, models_best_params = evaluate_models(X_train, 
                                                y_train,
                                                X_test,
                                                y_test,
                                                models,
                                                params)
            
            # To get best model score from dict
            best_model_score = max(sorted(model_test_report.values()))
            
            # To get best model name from dict

            best_model_name = list(model_test_report.keys())[
                list(model_test_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            best_model.set_params(**models_best_params[best_model_name])
            
            # mlflow.set_tracking_uri('http://localhost:5000')
            mlflow.autolog()
            
            mlflow.set_experiment('Student Performance Prediction')
            with mlflow.start_run(run_name='Student Performance Predictions'):
                mlflow.log_params(models_best_params[best_model_name])
                
                predicted = best_model.predict(X_test)

                r2_square = r2_score(y_test, predicted)
                mse = mean_squared_error(y_test, predicted)
                rmse = root_mean_squared_error(y_test, predicted)
                
                
                mlflow.log_metric('MSE', mse)
                mlflow.log_metric('RMSE', rmse)
                mlflow.log_metric('R2 Score', r2_square)
                
                mlflow.set_tag(
                    'Student Performance Prediction', 'v1.0'
                )
                
                signature = infer_signature(X_train, best_model.predict(X_train))
                
                
                
                mlflow.sklearn.log_model(
                    sk_model= best_model,
                    artifact_path='model',
                    signature= signature,
                    input_example=X_train,
                    registered_model_name=f'{best_model_name}'
                )
                
                
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(
                f"Best found model on both training and testing dataset")

            save_object(
                file_path=os.path.join(self.model_trainer_config.trained_model_file_path ,
                                       best_model_name + '.pkl'),
                obj=best_model
            )
            
            
            
            
            
            
            return r2_square, best_model_name, models_best_params[best_model_name]
        except Exception as e:
            raise CustomException(e)
            

import os
import sys

import pickle

from sklearn.base import r2_score
from sklearn.model_selection import GridSearchCV

from src.utills.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e)
    

def evaluate_models(X_train, y_train, X_test, y_test, models:dict, param):
    
    try:
        
        train_report = {}
        test_report = {}
        
        for model_name, model in models.items():
            para = param[model_name]
            
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)
            
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            
            y_train_pred = model.predict(X_train)
            
            y_test_pred = model.predict(X_test)
            
            train_model_accuracy = r2_score(y_train, y_train_pred)
            test_model_accuracy = r2_score(y_test, y_test_pred)
            
            train_report[model_name] = train_model_accuracy
            test_report[model_name] = test_model_accuracy
            
        return train_report, test_report
      
    except Exception as e:
        raise CustomException(e)
    

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e:
        raise CustomException(e)
 
import os
import sys
import logging

from requests import head

from src.utills import logger
from src.utills.exception import CustomException

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

RAW_DATA_CSV_FILE = 'data/data.csv'


@dataclass
class DataIngestionConfig:
    raw_data = os.path.join('src/artifacts', 'raw.csv')
    train_data = os.path.join('src/artifacts', 'train.csv')
    test_data = os.path.join('src/artifacts', 'test.csv')
    

class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()
        
    def intiate_data_ingestion(self):
        logging.info('Entered into the data ingestion method')
        
        try:
            data = pd.read_csv(RAW_DATA_CSV_FILE)
            logging.info('Reading Raw data completed')
            
            os.makedirs(
                os.path.dirname(
                    self.ingestion_config.raw_data
                ), exist_ok=True
            )
            os.makedirs(
                os.path.dirname(
                    self.ingestion_config.train_data
                ), exist_ok=True
            )
            os.makedirs(
                os.path.dirname(
                    self.ingestion_config.test_data
                ), exist_ok=True
            )
            
            data.to_csv(self.ingestion_config.raw_data, 
                        index=False, 
                        header=True)
            
            logging.info('Splitting the data into train and test data sets')
            
            train_data, test_data = train_test_split(data, 
                                                     test_size=0.2, 
                                                     random_state=42)
            
            train_data.to_csv(self.ingestion_config.train_data,
                        index=False,
                        header=True)
            test_data.to_csv(self.ingestion_config.test_data,
                        index=False,
                        header=True)
            
            logging.info('Splitting and saving the data finished')
            
            return (
                self.ingestion_config.train_data,
                self.ingestion_config.test_data
            )
        except Exception as e:
            raise CustomException(e)

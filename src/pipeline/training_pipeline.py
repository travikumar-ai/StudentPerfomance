
from src.components import data_transformation
from src.components.data_ingestion import DataIngestion
from src.components import data_transformation
from src.components.model_trainer import ModelTrainer








def run_training():
    data_ingestion = DataIngestion()
    train_data_path, test_data_path = data_ingestion.intiate_data_ingestion()
    # print(train_data_path, test_data_path, sep='\n')
    
    data_trans = data_transformation.DataTransformation()
    train_arr, test_arr, preprocessor_path = data_trans.intiate_data_transformation(train_data_path, test_data_path)
    print(preprocessor_path)
    
    model_trainer = ModelTrainer()
    
    r2_square, best_model_name, model_params = model_trainer.intiate_model_trainer(
        train_arr, test_arr
        )
    
    print(f"R2 Score:  {r2_square}")
    print(f"Best Model: {best_model_name}")
    print(f"Model Parameters: {model_params}")
    

if __name__=="__main__":
    run_training()
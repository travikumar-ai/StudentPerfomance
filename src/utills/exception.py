import logging
import sys
from src.utills import logger


def error_msg_details(error:Exception) -> str:
    
    _, _, exc_tb = sys.exc_info()
    
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_no = exc_tb.tb_lineno
    
    error_msg = ( f"Error occured in python script name [{file_name}]"
                 f"line number [{line_no}] error msg [{str(error)}]"
                 )
    
    return error_msg

class CustomException(Exception):
    def __init__(self, error) -> None:
        self.error_message = error_msg_details(error)
        super().__init__(self.error_message)
        

if __name__=="__main__":
    logging.info('Starting a test run that will handle an exception')
    
    try:
        a = 1/0
    except Exception as e:
        logging.error('A non-critical error was caught and handled')
        custom_ex = CustomException(e)
        logging.error(f"Error Details: {custom_ex.error_message}")
        
    logging.info('Test run finished.')
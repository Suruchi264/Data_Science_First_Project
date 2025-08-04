import logging
import os
import logging
from datetime import datetime
import sys

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path=os.path.join(os.getcwd(),"logs",LOG_FILE)
os.makedirs(logs_path,exist_ok=True)

LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,


)

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message

    def __str__(self):
        return str(self.error_message)

if __name__=="__main__":
    try: 
        a = 1 / 0
    except Exception as e:
        logging.info("Division by zero error")
        raise CustomException(e,sys)

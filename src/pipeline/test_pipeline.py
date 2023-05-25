import src
import pandas as pd

from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = ('artifacts\model.pkl')
            preprocessor_path = ('artifacts\preprocessor.pkl')
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)

            data_Scaled = preprocessor.transform(features)

            preds = model.predict(data_scaled)

            return preds 
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,gender: str,race_ethinicity:str,parental_level_of_eduction,lunch: str,test_preparation_course: str,reading_score: int,writing_score: int):
        self.gender = gender
        self.race_ethinicity = race_ethinicity
        self.parental_level_of_eduction = parental_level_of_eduction
        self.lunch = lunch,
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                'gender' : [gender],
                'race_ethinicity' : [race_ethinicity],
                'parental_level_of_eduction' : [parental_level_of_eduction],
                'lunch' :[lunch],
                'test_preparation_course' : [test_preparation_course],
                'reading_score' : [reading_score],
                'writing_score' : [writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)

import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj

'''
@dataclass: It is used to automatically generates the attributes. The dataclass decorator automatically generates __init__, __repr__, __eq__, and other methods based on the class attributes. This reduces boiler plate code and makes it easier to define and work with data-oriented classes.
'''

@dataclass
class datatransformationConfig:
    preprocessor_obj_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = datatransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            numeric_cols = ['writing score', 'reading score']
            cat_cols = ['gender','race/ethnicity','parental level of education','lunch','test preparation course']

            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaling',StandardScaler())
                ]
            )

            logging.info('Numerical Columns preprocessing completed'+'\n'+f'Numrical Columns: {numeric_cols}')

            cat_pipeline = Pipeline(
                steps =[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('encoding',OneHotEncoder()),
                    ('scaling', StandardScaler(with_mean=False))
                ]
            )

            logging.info('Categorical Columns preprocessing completed'+'\n'+f'Categorical Columns: {cat_cols}')

            preprocessor = ColumnTransformer(
                [
                    ('numeric_pipeline', num_pipeline, numeric_cols),
                    ('cat_pipeline', cat_pipeline, cat_cols)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read the train and test data successfully')

            logging.info('Obtaining preprocessor object')

            preprocessor_obj = self.get_data_transformer_object()

            target_col = 'math score'
            numeric_cols = ['writing score', 'reading score']

            input_feature_train_df = train_df.drop(columns=[target_col], axis = 1)
            target_feature_train_df = train_df[target_col]

            input_feature_test_df = test_df.drop(columns=[target_col], axis = 1)
            target_feature_test_df = test_df[target_col]

            logging.info("Applying preprocessor object on train and test data")

            input_feature_train_array = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_array = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_array, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_array, np.array(target_feature_test_df)
            ]
            logging.info('Saved Preprocessing Objects')

            save_obj(
                file_path = self.data_transformation_config.preprocessor_obj_path,
                obj = preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_path
            )
        except Exception as e:
            raise CustomException(e,sys)

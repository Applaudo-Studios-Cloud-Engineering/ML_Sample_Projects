import pandas as pd
from .nodes import *


def create_ML_pipeline(path_to_csv: str, package_name: str, module_name: str, algorithm_name: str, cv: int,
                       test_size: float, random_state: int):
    df = pd.read_csv(path_to_csv)

    X = df.drop(['Loan_Status'], axis=1)
    y = df['Loan_Status']

    return train_test_and_evaluate_model(package_name, module_name, algorithm_name, cv, X, y, test_size, random_state)

import pandas as pd
from .nodes import *


def create_preprocessing_pipeline(path_to_csv: str, path_to_save: str):
    df = pd.read_csv(path_to_csv)

    df['Credit_History'], null_count = fill_empty_values(df['Credit_History'], [1.0, 0.0], [0.84, 0.16], float)
    print("Null values found in feature Credit_History after fill process: ", null_count)

    df['Self_Employed'], null_count = fill_empty_values(df['Self_Employed'], ['No', 'Yes'], [0.81, 0.19], str)
    print("Null values found in feature Self_Employed after fill process: ", null_count)

    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
    print("Null values found in feature LoanAmount after fill process: ", null_count)

    df['Dependents'] = df['Dependents'].replace("3+", 3)  # Replace "3+" with 3
    df['Dependents'], null_count = fill_empty_values(df['Dependents'], [0, 1, 2, 3], [0.59, 0.17, 0.16, 0.08], int)
    print("Null values found in feature Dependents after fill process: ", null_count)

    df['Loan_Amount_Term'], null_count = fill_empty_values(df['Loan_Amount_Term'],
                                                           [360.0, 180.0, 480.0, 300.0, 240.0, 84.0, 120.0, 60.0, 36.0,
                                                            12.0],
                                                           [0.83, 0.07, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.005,
                                                            0.005], float)
    print("Null values found in feature Loan_Amount_Term after fill process: ", null_count)

    df['Gender'], null_count = fill_empty_values(df['Gender'], ['Male', 'Female'], [0.81, 0.19], str)
    print("Null values found in feature Gender after fill process: ", null_count)

    df['Married'], null_count = fill_empty_values(df['Married'], ['No', 'Yes'], [0.65, 0.35], str)
    print("Null values found in feature Married after fill process: ", null_count)

    df['ApplicantIncome'] = df['ApplicantIncome'].astype(float)
    df['CoapplicantIncome'] = df['CoapplicantIncome'].astype(float)

    df.to_csv(path_to_save, index=False)

    return df
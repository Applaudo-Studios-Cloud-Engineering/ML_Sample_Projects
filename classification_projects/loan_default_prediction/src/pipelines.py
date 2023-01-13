import pandas as pd

from nodes import fill_empty_values


def preprocessing_pipeline():
    df = pd.read_csv('')

    df['Credit_History'], null_count = fill_empty_values(df['Credit_History'], [1.0, 0.0], [0.84, 0.16], 'float')
    df['Self_Employed'], null_count = fill_empty_values(df['Self_Employed'], ['No', 'Yes'], [0.81, 0.19], 'str')
    df['Loan_Amount'] = df['Loan_Amount'].fillna(df['LoanAmount'].median())
    df['Dependents'], null_count = fill_empty_values(df['Dependents'], ['0', '1', '2', '3+'], [0.59, 0.17, 0.16, 0.08], 'str')
    df['Loan_Amount_Term'], null_count = fill_empty_values(df['Loan_Amount_Term'],
                                               [360.0, 180.0, 480.0, 300.0, 240.0, 84.0, 120.0, 60.0, 36.0, 12.0],
                                               [0.83, 0.07, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.005, 0.005], 'float')
    df['Gender'], null_count = fill_empty_values(df['Gender'], ['Male', 'Female'], [0.81, 0.19], 'str')


import pandas as pd
from .nodes import *


def create_feature_engineering_pipeline(path_to_csv, path_to_save: str,
                                        apply_distribution_transformations: bool = True):
    df = pd.read_csv(path_to_csv)

    df['Gender'], null_count = encode_feature_values(df['Gender'], {'Male': 0, 'Female': 1})
    print("Null values found in feature Gender after encoding process: ", null_count)

    df['Married'], null_count = encode_feature_values(df['Married'], {'No': 0, 'Yes': 1})
    print("Null values found in feature Married after encoding process: ", null_count)

    df['Education'], null_count = encode_feature_values(df['Education'], {'Not Graduate': 0, 'Graduate': 1})
    print("Null values found in feature Education after encoding process: ", null_count)

    df['Property_Area'], null_count = encode_feature_values(df['Property_Area'],
                                                            {'Urban': 0, 'Semiurban': 1, 'Rural': 2})
    print("Null values found in feature Property_Area after encoding process: ", null_count)

    df['Loan_Status'], null_count = encode_feature_values(df['Loan_Status'], {'N': 0, 'Y': 1})
    print("Null values found in feature Loan_Status after encoding process: ", null_count)

    df['Self_Employed'], null_count = encode_feature_values(df['Self_Employed'], {'No': 0, 'Yes': 1})
    print("Null values found in feature Self_Employed after encoding process: ", null_count)

    df['Total_Income'], null_count = sum_features([df['ApplicantIncome'], df['CoapplicantIncome']], 0)
    null_count = df['Total_Income'].isnull().sum()
    print("Null values found in feature Total_Income after sum process: ", null_count)

    if apply_distribution_transformations:
        df['Total_Income'], null_count = apply_distribution_transformation(df['Total_Income'])
        print("Null values found in feature Total_Income after distribution transformation process: ", null_count)

        df['LoanAmount'], null_count = apply_distribution_transformation(df['LoanAmount'])
        print("Null values found in feature LoanAmount after distribution transformation process: ", null_count)

    df.drop(['Loan_ID'], axis=1, inplace=True)

    df.to_csv(path_to_save, index=False)

    return df

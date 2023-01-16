import pandas as pd

from nodes import fill_empty_values, encode_feature_values, apply_distribution_transformation, \
    train_test_and_evaluate_model


def create_preprocessing_pipeline(path_to_csv: str, path_to_save: str):
    df = pd.read_csv(path_to_csv)

    df['Credit_History'], null_count = fill_empty_values(df['Credit_History'], [1.0, 0.0], [0.84, 0.16], 'float')
    df['Self_Employed'], null_count = fill_empty_values(df['Self_Employed'], ['No', 'Yes'], [0.81, 0.19], 'str')
    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
    df['Dependents'], null_count = fill_empty_values(df['Dependents'], ['0', '1', '2', '3+'], [0.59, 0.17, 0.16, 0.08],
                                                     'str')
    df['Loan_Amount_Term'], null_count = fill_empty_values(df['Loan_Amount_Term'],
                                                           [360.0, 180.0, 480.0, 300.0, 240.0, 84.0, 120.0, 60.0, 36.0,
                                                            12.0],
                                                           [0.83, 0.07, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.005,
                                                            0.005], 'float')
    df['Gender'], null_count = fill_empty_values(df['Gender'], ['Male', 'Female'], [0.81, 0.19], 'str')
    df['Married'], null_count = fill_empty_values(df['Married'], ['No', 'Yes'], [0.65, 0.35])

    df.to_csv(path_to_save, index=False)

    return df


def create_feature_engineering_pipeline(path_to_csv, path_to_save: str,
                                        apply_distribution_transformations: bool = True):
    df = pd.read_csv(path_to_csv)

    df['Gender'], null_count = encode_feature_values(df['Gender'], {'Male': 0, 'Female': 1})
    df['Married'], null_count = encode_feature_values(df['Married'], {'No': 0, 'Yes': 1})
    df['Education'], null_count = encode_feature_values(df['Education'], {'Not Graduate': 0, 'Graduate': 1})
    df['Property_Area'], null_count = encode_feature_values(df['Property_Area'],
                                                            {'Urban': 0, 'Semiurban': 1, 'Rural': 2})
    df['Loan_Status'], null_count = encode_feature_values(df['Loan_Amount'], {'N': 0, 'Y': 1})
    df['Self_Employed'], null_count = encode_feature_values(df['Self_Employed'], {'No': 0, 'Yes': 1})

    df['Total_Income'] = df['ApplicationIncome'] + df['CoapplicantIncome']

    if apply_distribution_transformations:
        df['Total_Income'] = apply_distribution_transformation(df['Total_Income'])
        df['LoanAmount'] = apply_distribution_transformation(df['LoanAmount'])

    df.drop(['Loan_ID'], axis=1, inplace=True)

    df.to_csv(path_to_save, index=False)

    return df


def create_ML_pipeline(path_to_csv: str):
    df = pd.read_csv(path_to_csv)

    X = df.drop(['Loan_Status'], axis=1)
    y = df['Loan_Status']

    X_train, X_test, y_train, y_test, model, acc, conf_matrix, cross_val \
        = train_test_and_evaluate_model('sklearn', 'linear_model', 'LogisticRegression', 5, X, y)

    X_train, X_test, y_train, y_test, model, acc, conf_matrix, cross_val \
        = train_test_and_evaluate_model('sklearn', 'tree', 'DecisionTreeClassifier', 5, X, y)

    X_train, X_test, y_train, y_test, model, acc, conf_matrix, cross_val \
        = train_test_and_evaluate_model('sklearn', 'ensemble', 'RandomForestClassifier', 5, X, y)

    X_train, X_test, y_train, y_test, model, acc, conf_matrix, cross_val \
        = train_test_and_evaluate_model('xgboost', None, 'XGBClassifier', 5, X, y)


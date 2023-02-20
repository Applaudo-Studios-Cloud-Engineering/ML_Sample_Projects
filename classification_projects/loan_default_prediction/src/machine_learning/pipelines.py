import pandas as pd
from .nodes import *

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

    print('Model trained successfully, acc: ', acc)

    return model, acc

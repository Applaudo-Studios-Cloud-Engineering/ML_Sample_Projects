from typing import List, Any, Dict

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report


def train_test_and_evaluate_model(ML_lib: str, package_name: str | None, algorithm_name: str, cv_split: int,
                                  X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 123) \
        -> [pd.DataFrame,
            pd.DataFrame,
            pd.Series,
            pd.Series, Any,
            float, Any,
            List[float]]:

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size, random_state)

    if ML_lib == 'xgboost':
        algorithm = getattr(ML_lib, algorithm_name)
    else:
        algorithm = getattr(ML_lib, f'{package_name}.{algorithm_name}')

    model = algorithm()
    model.fit(X_train, y_train)

    acc = model.evaluate(X_test, y_test)

    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_pred, y_pred)

    cross_val = cross_val_score(model, X, y, cv=cv_split)

    return [X_train, X_test, y_train, y_test, model, acc, conf_matrix, cross_val]


def tune_logistic_regression(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 123,
                             cv_split: int = 5, hyper_params: List[Any] = None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size, random_state)

    model = LogisticRegression(hyper_params)

    model.fit(X_train, y_train)

    acc = model.evaluate(X_test, y_test)

    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_pred, y_pred)

    cross_val = cross_val_score(model, X, y, cv=cv_split)

    return [X_train, X_test, y_train, y_test, model, acc, conf_matrix, cross_val]

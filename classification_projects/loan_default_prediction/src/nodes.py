from typing import List, Any, Dict

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report


def remove_symbols(col: pd.Series, symbol: str, replace_symbol: Any = None) -> [pd.Series, int]:
    res_col = col.str.replace(symbol, replace_symbol)
    null_count = res_col.isnull().sum()

    return [res_col, null_count]


def fill_empty_values(col: pd.Series, fill_values: List[Any], probabilities: List[float], type_inference: str) -> \
        [pd.Series, int]:
    res_col = col.fillna(np.random.choice(fill_values, probabilities)).astype(type_inference)
    null_count = res_col.isnull().sum()

    return [res_col, null_count]


def encode_feature_values(col: pd.Series, encode_dict: Dict[str, int]) -> [pd.Series, int]:
    res_col = col.map(encode_dict)
    null_count = res_col.isnull().sum()

    return [res_col, null_count]


def sum_features(feat_to_sum: List[pd.Series], fill_val: Any = None) -> [pd.Series, int]:
    res_col = pd.Series([])

    for feat in feat_to_sum:
        res_col.add(feat, fill_value=fill_val)

    null_count = res_col.isnull().sum()

    return [res_col, null_count]


def apply_distribution_transformation(col: pd.Series, transformation_to_apply: str = 'log') -> [pd.Series, int]:
    if transformation_to_apply in ['log', 'sqrt', 'reciprocal']:
        transformation = getattr(np, transformation_to_apply)

        res_col = transformation(col)
        null_count = res_col.isnull().sum()

        return [res_col, null_count]


def train_test_and_evaluate_model(ML_lib: str, package_name: str | None, algorithm_name: str, cv_split: int,
                                  X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 123) \
        -> [pd.DataFrame,
            pd.DataFrame,
            pd.Series,
            pd.Series, Any,
            float, Any,
            List[float]]:

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size, random_state)

    if ML_lib is 'xgboost':
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

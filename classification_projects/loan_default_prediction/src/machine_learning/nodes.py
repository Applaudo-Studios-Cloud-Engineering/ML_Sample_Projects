import importlib
from typing import Any, List

from typing import Union

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import mlflow
from mlflow.models.signature import infer_signature

def train_test_and_evaluate_model(ML_lib: str, package_name: Union[str, None], algorithm_name: str, cv_split: int,
                                  X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 123) \
        -> [pd.DataFrame,
            pd.DataFrame,
            pd.Series,
            pd.Series, Any,
            float, Any,
            List[float]]:
 #   mlflow.set_tracking_uri("http://my-mlflow.mlflow.svc.cluster.local:80")
  #  mlflow.set_registry_uri("http://my-mlflow.mlflow.svc.cluster.local:80")
   # mlflow.set_experiment(experiment_name="train-example")        
   
    #mlflow.sklearn.autolog(
     #   registered_model_name="sklearn-model-tests",
      #  log_input_examples=True
   # )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(test_size), random_state=int(random_state))

    ML_lib = importlib.import_module("sklearn.linear_model")

    if ML_lib == 'xgboost':
        algorithm = getattr(ML_lib, algorithm_name)
    else:
        algorithm = getattr(ML_lib, "LogisticRegression")

    model = algorithm(max_iter=1000)
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)

    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)

    cross_val = cross_val_score(model, X, y, cv=cv_split)

    # signature = infer_signature(x, y)
    # print(signature)
    # mlflow.log_metric("accuracy", accuracy)
    # mlflow.sklearn.log_model(
    # model,
    # "sklearn-model-tests",
    # registered_model_name="sklearn-model-tests",
    # signature=signature
    # )
    run_id = mlflow.last_active_run().info.run_id

    print("Logged data and model in run {}".format(run_id))

    return [X_train, X_test, y_train, y_test, model, acc, conf_matrix, cross_val]



def tune_logistic_regression(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 123,
                             cv_split: int = 5, hyper_params: List[Any] = None):
    
    print(f"test_size: {test_size}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state=random_state)

    model = LogisticRegression(hyper_params)

    model.fit(X_train, y_train)

    acc = model.evaluate(X_test, y_test)

    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_pred, y_pred)

    cross_val = cross_val_score(model, X, y, cv=cv_split)

    return [X_train, X_test, y_train, y_test, model, acc, conf_matrix, cross_val]

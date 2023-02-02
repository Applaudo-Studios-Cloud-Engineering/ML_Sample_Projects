from typing import Any, List
import pandas as pd
import numpy as np


def remove_symbols(col: pd.Series, symbol: str, replace_symbol: Any = None) -> [pd.Series, int]:
    res_col = col.str.replace(symbol, replace_symbol)
    null_count = res_col.isnull().sum()

    return [res_col, null_count]


def fill_empty_values(col: pd.Series, fill_values: List[Any], probabilities: List[float], type_inference: Any) -> \
        [pd.Series, int]:
    res_col = col.fillna(np.random.choice(fill_values, p=probabilities)).astype(type_inference)
    null_count = res_col.isnull().sum()

    return [res_col, null_count]
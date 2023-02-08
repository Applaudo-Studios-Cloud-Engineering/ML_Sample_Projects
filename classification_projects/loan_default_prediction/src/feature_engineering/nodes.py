from typing import Dict, List, Any
import pandas as pd
import numpy as np


def encode_feature_values(col: pd.Series, encode_dict: Dict[str, int]) -> [pd.Series, int]:
    res_col = col.map(encode_dict)
    null_count = res_col.isnull().sum()

    return [res_col, null_count]


def sum_features(feat_to_sum: List[pd.Series], fill_val: Any = None) -> [pd.Series, int]:
    res_col = feat_to_sum[0].copy()

    for i in range(1, len(feat_to_sum)):
        res_col = res_col.add(feat_to_sum[i], fill_value=fill_val)

    null_count = res_col.isnull().sum()

    return [res_col, null_count]


def apply_distribution_transformation(col: pd.Series, transformation_to_apply: str = 'log') -> [pd.Series, int]:
    if transformation_to_apply in ['log', 'sqrt', 'reciprocal']:
        transformation = getattr(np, transformation_to_apply)

        res_col = transformation(col)
        null_count = res_col.isnull().sum()

        return [res_col, null_count]

import pandas as pd
import numpy as np

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from feature_engineering import (encode_feature_values, sum_features, apply_distribution_transformation)


def test_encode_feature_values():
    encode_dict = {'a': 1, 'b': 2, 'c': 3}
    data = pd.Series(['a', 'b', 'c', np.nan])
    expected_result = pd.Series([1, 2, 3, np.nan])
    result, null_count = encode_feature_values(data, encode_dict)

    assert expected_result.equals(result)
    assert null_count == 1


def test_sum_features():
    data1 = pd.Series([1, 2, 3, np.nan])
    data2 = pd.Series([4, np.nan, 6, 7])
    expected_result = pd.Series([5, np.nan, 9, np.nan])
    result, null_count = sum_features([data1, data2], np.nan)
    print(expected_result)
    assert expected_result.equals(result)
    assert null_count == 2


def test_apply_distribution_transformation():
    data = pd.Series([1, 4, 9, np.nan])
    expected_result = pd.Series([1, 2, 3, np.nan])
    result, null_count = apply_distribution_transformation(data, 'sqrt')

    assert expected_result.equals(result)
    assert null_count == 1

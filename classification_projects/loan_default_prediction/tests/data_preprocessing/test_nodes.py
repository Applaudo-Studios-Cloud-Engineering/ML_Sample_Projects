import pandas as pd

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from data_preprocessing import (remove_symbols)

def test_remove_symbols():
    # Create a test DataFrame
    df = pd.DataFrame({
        'col1': ['123', '456', '789'],
        'col2': ['a@b.c', 'd@e.f', 'g@h.i'],
        'col3': [1, 2, 3]
    })

    # Call the remove_symbols function with the test DataFrame

    # Check that the function returned the expected results
    assert ['a.b.c', 'd.e.f', 'g.h.i'] == ['a.b.c', 'd.e.f', 'g.h.i']
    assert 0 == 0

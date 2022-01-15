import pandas as pd
import numpy as np

from src.data.feature_engineering import get_mean_within_category


def test_get_mean_within_category():

    input_df = pd.DataFrame(
        {
            "group": ["a", "b", "a", "b", "c"],
            "value": [1, 3, 3, 2, 4],
        }
    )
    expected_result = np.array([2, 2.5, 2, 2.5, 4])
    assert (
        get_mean_within_category(input_df, "group", "value").values == expected_result
    ).all()

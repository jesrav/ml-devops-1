import pandas as pd
from src.data.preprocessing import add_churn_target


def test_add_churn_target():

    input_df = pd.DataFrame(
        {
            "Attrition_Flag": [
                "Existing Customer",
                "Existing Customer",
                "Attrited Customer",
            ],
            "dummy": [1, 3, 3],
        }
    )
    expected_result = pd.DataFrame(
        {
            "Attrition_Flag": [
                "Existing Customer",
                "Existing Customer",
                "Attrited Customer",
            ],
            "dummy": [1, 3, 3],
            "Churn": [0, 0, 1],
        }
    )
    pd.testing.assert_frame_equal(add_churn_target(input_df), expected_result)

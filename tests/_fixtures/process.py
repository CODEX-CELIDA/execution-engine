import pandas as pd
import pytest


@pytest.fixture
def empty_dataframe():
    df_dtypes = {
        "person_id": "int64",
        "interval_start": "datetime64[ns, UTC]",
        "interval_end": "datetime64[ns, UTC]",
        "interval_type": "category",
    }

    return pd.DataFrame(columns=df_dtypes.keys())

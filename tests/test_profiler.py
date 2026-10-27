import pandas as pd
import pytest
from tools.data_profiler import profile_dataset

def test_data_profiler_detects_numeric_and_categorical():
    df = pd.DataFrame({
        "num": [1, 2, 3],
        "cat": ["a", "b", "c"],
        "label": [0, 1, 0]
    })
    profile = profile_dataset(df, target= "label")
    assert "numeric" in profile ["feature_types"]
    assert "categorical" in profile ["feature_types"]

def test_profiler_missing():
    df = pd.DataFrame({
        "a": [1, "none", 3],
        "label": [0, 1, 0]
    })
    profile = profile_dataset(df, target= "label")
    assert "a" in profile["missing_pct"]

def test_profiler_imbalance():
    df = pd.DataFrame({
        "x": [1, 2, 3, 4],
        "label": [0, 0, 0, 1]
    })
    profile = profile_dataset(df, target= "label")
    assert profile["imbalance_ratio"] >= 3.0

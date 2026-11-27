"""
Cleaning statistics come from the training rows only (KNOWN_ISSUES #86). No data needed.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src.data_prep import EMPLOYED_PLACEHOLDER, clean_data, engineer_features, fit_cleaning, split_mask


def _frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "SK_ID_CURR": np.arange(n),
        "TARGET": (rng.random(n) < 0.2).astype(int),
        "AMT_INCOME_TOTAL": rng.normal(100.0, 10.0, n),
        "AMT_CREDIT": rng.normal(500.0, 50.0, n),
        "AMT_GOODS_PRICE": rng.normal(450.0, 40.0, n),
        "DAYS_BIRTH": -rng.integers(8000, 20000, n),
        "DAYS_EMPLOYED": -rng.integers(100, 5000, n).astype(float),
        "FLAG_X": rng.integers(0, 2, n),
        "NAME_TYPE": rng.choice(["a", "b"], n, p=[0.8, 0.2]).astype(object),
    })


def test_split_matches_train_baseline():
    df = _frame()
    X_tr, _, _, _ = train_test_split(df.drop(columns="TARGET"), df["TARGET"], test_size=0.2,
                                     stratify=df["TARGET"], random_state=42)
    mask = split_mask(df["TARGET"].values)
    assert mask.sum() == len(X_tr)
    assert set(np.where(mask)[0]) == set(X_tr.index)


def test_test_rows_do_not_change_the_statistics():
    df = _frame()
    train = split_mask(df["TARGET"].values)
    base = fit_cleaning(df[train])

    shifted = df.copy()
    shifted.loc[~train, "AMT_INCOME_TOTAL"] = 1e9          # extreme values in the test rows only
    shifted.loc[~train, "NAME_TYPE"] = "b"
    again = fit_cleaning(shifted[train])
    assert again["medians"]["AMT_INCOME_TOTAL"] == base["medians"]["AMT_INCOME_TOTAL"]
    assert again["bounds"]["AMT_INCOME_TOTAL"] == base["bounds"]["AMT_INCOME_TOTAL"]
    assert again["modes"]["NAME_TYPE"] == "a"

    cleaned = clean_data(shifted, again)
    lo, hi = base["bounds"]["AMT_INCOME_TOTAL"]
    assert cleaned.loc[~train, "AMT_INCOME_TOTAL"].max() == pytest.approx(hi)   # clipped to the training bound


def test_gaps_filled_with_training_values_and_binary_columns_not_clipped():
    df = _frame()
    train = split_mask(df["TARGET"].values)
    df.loc[np.where(~train)[0][:5], ["AMT_CREDIT", "NAME_TYPE"]] = np.nan
    stats = fit_cleaning(df[train])
    out = clean_data(df, stats)
    assert out.isna().sum().sum() == 0
    assert (out.loc[np.where(~train)[0][:5], "AMT_CREDIT"] == stats["medians"]["AMT_CREDIT"]).all()
    assert "FLAG_X" not in stats["bounds"] and "TARGET" not in stats["bounds"]
    assert "SK_ID_CURR" not in stats["bounds"]


def test_employment_placeholder_becomes_flag():
    df = _frame()
    df.loc[0, "DAYS_EMPLOYED"] = EMPLOYED_PLACEHOLDER
    out = clean_data(df, fit_cleaning(df))
    assert out.loc[0, "IS_EMPLOYED_ANOMALY"] == 1 and out.loc[1, "IS_EMPLOYED_ANOMALY"] == 0
    assert out.loc[0, "DAYS_EMPLOYED"] != EMPLOYED_PLACEHOLDER


def test_ratio_gaps_use_training_median():
    df = _frame()
    train = split_mask(df["TARGET"].values)
    test_row = int(np.where(~train)[0][0])
    df.loc[test_row, "AMT_GOODS_PRICE"] = 0.0               # ratio undefined for this row
    out = engineer_features(df, train)
    expected = (df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"])[train].median()
    assert out.loc[test_row, "CREDIT_TO_GOODS_RATIO"] == pytest.approx(expected)
    assert out.loc[test_row, "AGE_YEARS"] == int(-df.loc[test_row, "DAYS_BIRTH"] / 365)

"""
Phase 1: Data loading, cleaning, feature engineering

Fill values (median, most common value) and clipping bounds are learned from
the training rows only, using the same stratified 80/20 split that
src/train_baseline.py makes afterwards, so the test rows never shape them.
Run: python src/data_prep.py --data-path data/home_credit/ --out data/processed/app_train_clean.csv
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path="data/home_credit/"):
    app_train = pd.read_csv(path + "application_train.csv")
    bureau = pd.read_csv(path + "bureau.csv")
    prev_app = pd.read_csv(path + "previous_application.csv")
    return {"app_train": app_train, "bureau": bureau, "prev_app": prev_app}


def aggregate_secondary_tables(app_df, bureau, prev_app):
    # aggregate to one row per SK_ID_CURR before merging, else rows duplicate
    bureau_agg = bureau.groupby("SK_ID_CURR").agg(
        BUREAU_CREDIT_COUNT=("SK_ID_BUREAU", "count"),
        BUREAU_AVG_CREDIT_AMT=("AMT_CREDIT_SUM", "mean"),
        BUREAU_MAX_OVERDUE=("AMT_CREDIT_SUM_OVERDUE", "max"),
    ).reset_index()

    prev_app_agg = prev_app.groupby("SK_ID_CURR").agg(
        PREV_APP_COUNT=("SK_ID_PREV", "count"),
        PREV_APP_AVG_CREDIT=("AMT_CREDIT", "mean"),
        PREV_APP_REFUSED_COUNT=("NAME_CONTRACT_STATUS", lambda s: (s == "Refused").sum()),
    ).reset_index()

    merged = app_df.merge(bureau_agg, on="SK_ID_CURR", how="left")
    merged = merged.merge(prev_app_agg, on="SK_ID_CURR", how="left")

    fill_cols = [
        "BUREAU_CREDIT_COUNT", "BUREAU_AVG_CREDIT_AMT", "BUREAU_MAX_OVERDUE",
        "PREV_APP_COUNT", "PREV_APP_AVG_CREDIT", "PREV_APP_REFUSED_COUNT",
    ]
    merged[fill_cols] = merged[fill_cols].fillna(0)
    return merged


NO_CLIP_COLUMNS = {"SK_ID_CURR", "TARGET", "IS_EMPLOYED_ANOMALY"}
EMPLOYED_PLACEHOLDER = 365243   # DAYS_EMPLOYED code for unemployed/pensioner, not a real value
TEST_SIZE, SPLIT_SEED = 0.2, 42  # the split of src/train_baseline.py


def split_mask(y, test_size=TEST_SIZE, seed=SPLIT_SEED):
    """True for training rows: the same stratified split src/train_baseline.py makes on this row order."""
    idx = np.arange(len(y))
    train_idx, _ = train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)
    mask = np.zeros(len(y), dtype=bool)
    mask[train_idx] = True
    return mask


def _flag_employment(df):
    df = df.copy()
    df["IS_EMPLOYED_ANOMALY"] = (df["DAYS_EMPLOYED"] == EMPLOYED_PLACEHOLDER).astype(int)
    df.loc[df["DAYS_EMPLOYED"] == EMPLOYED_PLACEHOLDER, "DAYS_EMPLOYED"] = np.nan
    return df


def fit_cleaning(train_df):
    """Fill values and clipping bounds, learned from the training rows only."""
    df = _flag_employment(train_df)
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns   # text categories; the timestamp is left alone
    medians = df[num_cols].median()
    filled = df[num_cols].fillna(medians)
    # skip ID/label/binary cols so clipping doesn't corrupt them
    clip_cols = [c for c in num_cols
                 if c not in NO_CLIP_COLUMNS and not set(filled[c].dropna().unique()).issubset({0, 1})]
    bounds = {}
    for col in clip_cols:
        std = filled[col].std()
        if std and std > 0:
            mean = filled[col].mean()
            bounds[col] = (mean - 3 * std, mean + 3 * std)
    modes = df[cat_cols].mode().iloc[0] if len(cat_cols) else pd.Series(dtype=object)
    return {"medians": medians, "modes": modes, "bounds": bounds}


def clean_data(df, stats):
    """Flags the employment placeholder, fills gaps and clips outliers with statistics from fit_cleaning."""
    df = _flag_employment(df)
    num_cols = [c for c in stats["medians"].index if c in df.columns]
    df[num_cols] = df[num_cols].fillna(stats["medians"][num_cols])
    if len(stats["modes"]):
        cat_cols = [c for c in stats["modes"].index if c in df.columns]
        df[cat_cols] = df[cat_cols].fillna(stats["modes"][cat_cols])
    for col, (lo, hi) in stats["bounds"].items():
        df[col] = df[col].clip(lo, hi)
    return df


def engineer_features(df, train_mask):
    df = df.copy()
    df["CREDIT_TO_GOODS_RATIO"] = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"].replace(0, np.nan)
    df["AGE_YEARS"] = (-df["DAYS_BIRTH"] / 365).astype(int)
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    for col in ("CREDIT_TO_GOODS_RATIO", "CREDIT_INCOME_RATIO"):
        df[col] = df[col].fillna(df.loc[train_mask, col].median())
    return df


def simulate_application_timestamps(df, start_date="2025-01-01", end_date="2025-12-31", seed=42):
    # dataset has no real submission date, so simulate one for the dashboard's history view
    rng = np.random.default_rng(seed)
    date_range = pd.date_range(start=start_date, end=end_date, freq="h")
    df = df.copy()
    df["SIMULATED_SUBMITTED_AT"] = rng.choice(date_range, size=len(df))
    return df.sort_values("SIMULATED_SUBMITTED_AT").reset_index(drop=True)


def run_pipeline(data_path, out_path):
    tables = load_data(data_path)
    df = aggregate_secondary_tables(tables["app_train"], tables["bureau"], tables["prev_app"])
    # order the rows first: the train/test split depends on it, and cleaning only uses the training rows
    df = simulate_application_timestamps(df)
    train = split_mask(df["TARGET"].values)
    df = clean_data(df, fit_cleaning(df[train]))
    df = engineer_features(df, train)
    df = df[[c for c in df.columns if c != "SIMULATED_SUBMITTED_AT"] + ["SIMULATED_SUBMITTED_AT"]]
    df.to_csv(out_path, index=False)
    print(f"[data_prep] Wrote {len(df)} rows, {df.shape[1]} columns -> {out_path} "
          f"(cleaning statistics from {int(train.sum())} training rows)")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/home_credit/")
    parser.add_argument("--out", default="data/processed/app_train_clean.csv")
    args = parser.parse_args()
    run_pipeline(args.data_path, args.out)
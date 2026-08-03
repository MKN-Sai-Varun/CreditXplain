"""
Phase 1: Data loading, cleaning, feature engineering
Run: python src/data_prep.py --data-path data/home_credit/ --out data/processed/app_train_clean.csv
"""
import argparse
import numpy as np
import pandas as pd


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


def clean_data(df):
    df = df.copy()
    # 365243 = placeholder for unemployed/pensioner, not a real value
    df["IS_EMPLOYED_ANOMALY"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
    df.loc[df["DAYS_EMPLOYED"] == 365243, "DAYS_EMPLOYED"] = np.nan

    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns

    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    if len(cat_cols) > 0:
        modes = df[cat_cols].mode().iloc[0]
        df[cat_cols] = df[cat_cols].fillna(modes)

    # skip ID/label/binary cols so clipping doesn't corrupt them
    clip_cols = [
        c for c in num_cols
        if c not in NO_CLIP_COLUMNS and not set(df[c].dropna().unique()).issubset({0, 1})
    ]
    for col in clip_cols:
        std = df[col].std()
        if std and std > 0:
            mean = df[col].mean()
            df[col] = df[col].clip(mean - 3 * std, mean + 3 * std)

    return df


def engineer_features(df):
    df = df.copy()
    df["CREDIT_TO_GOODS_RATIO"] = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"].replace(0, np.nan)
    df["AGE_YEARS"] = (-df["DAYS_BIRTH"] / 365).astype(int)
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    df["CREDIT_TO_GOODS_RATIO"] = df["CREDIT_TO_GOODS_RATIO"].fillna(df["CREDIT_TO_GOODS_RATIO"].median())
    df["CREDIT_INCOME_RATIO"] = df["CREDIT_INCOME_RATIO"].fillna(df["CREDIT_INCOME_RATIO"].median())
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
    df = clean_data(df)
    df = engineer_features(df)
    df = simulate_application_timestamps(df)
    df.to_csv(out_path, index=False)
    print(f"[data_prep] Wrote {len(df)} rows, {df.shape[1]} columns -> {out_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/home_credit/")
    parser.add_argument("--out", default="data/processed/app_train_clean.csv")
    args = parser.parse_args()
    run_pipeline(args.data_path, args.out)
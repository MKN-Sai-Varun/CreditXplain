import os
import numpy as np
import pandas as pd


TRAIN_X = "data/processed/X_train.csv"
TEST_X = "data/processed/X_test.csv"

OUTPUT_DIR = "data/processed"

TRAIN_OUT = os.path.join(
    OUTPUT_DIR,
    "X_train_fe_v2.csv"
)

TEST_OUT = os.path.join(
    OUTPUT_DIR,
    "X_test_fe_v2.csv"
)


def safe_divide(a, b):
    return a / (b.abs() + 1e-6)


def add_features(df):

    df = df.copy()

    # =========================================================
    # 1. EXTERNAL SOURCE FEATURES
    # =========================================================

    ext_cols = [
        c for c in [
            "EXT_SOURCE_1",
            "EXT_SOURCE_2",
            "EXT_SOURCE_3"
        ]
        if c in df.columns
    ]

    if len(ext_cols) >= 2:

        df["EXT_SOURCE_MEAN"] = df[ext_cols].mean(axis=1)
        df["EXT_SOURCE_MIN"] = df[ext_cols].min(axis=1)
        df["EXT_SOURCE_MAX"] = df[ext_cols].max(axis=1)
        df["EXT_SOURCE_STD"] = df[ext_cols].std(axis=1)
        df["EXT_SOURCE_MEDIAN"] = df[ext_cols].median(axis=1)

        df["EXT_SOURCE_RANGE"] = (
            df["EXT_SOURCE_MAX"] -
            df["EXT_SOURCE_MIN"]
        )

        df["EXT_SOURCE_COUNT"] = (
            df[ext_cols].notna().sum(axis=1)
        )

    if "EXT_SOURCE_1" in df.columns and "EXT_SOURCE_2" in df.columns:
        df["EXT_SOURCE_1_2_DIFF"] = (
            df["EXT_SOURCE_1"] -
            df["EXT_SOURCE_2"]
        )

        df["EXT_SOURCE_1_2_PRODUCT"] = (
            df["EXT_SOURCE_1"] *
            df["EXT_SOURCE_2"]
        )

    if "EXT_SOURCE_1" in df.columns and "EXT_SOURCE_3" in df.columns:
        df["EXT_SOURCE_1_3_DIFF"] = (
            df["EXT_SOURCE_1"] -
            df["EXT_SOURCE_3"]
        )

        df["EXT_SOURCE_1_3_PRODUCT"] = (
            df["EXT_SOURCE_1"] *
            df["EXT_SOURCE_3"]
        )

    if "EXT_SOURCE_2" in df.columns and "EXT_SOURCE_3" in df.columns:
        df["EXT_SOURCE_2_3_DIFF"] = (
            df["EXT_SOURCE_2"] -
            df["EXT_SOURCE_3"]
        )

        df["EXT_SOURCE_2_3_PRODUCT"] = (
            df["EXT_SOURCE_2"] *
            df["EXT_SOURCE_3"]
        )


    # =========================================================
    # 2. FINANCIAL BURDEN FEATURES
    # =========================================================

    if "AMT_ANNUITY" in df.columns and "AMT_INCOME_TOTAL" in df.columns:

        df["ANNUITY_INCOME_RATIO"] = safe_divide(
            df["AMT_ANNUITY"],
            df["AMT_INCOME_TOTAL"]
        )

    if "AMT_CREDIT" in df.columns and "AMT_INCOME_TOTAL" in df.columns:

        df["CREDIT_INCOME_RATIO_NEW"] = safe_divide(
            df["AMT_CREDIT"],
            df["AMT_INCOME_TOTAL"]
        )

    if "AMT_GOODS_PRICE" in df.columns and "AMT_INCOME_TOTAL" in df.columns:

        df["GOODS_INCOME_RATIO"] = safe_divide(
            df["AMT_GOODS_PRICE"],
            df["AMT_INCOME_TOTAL"]
        )

    if "AMT_CREDIT" in df.columns and "AMT_ANNUITY" in df.columns:

        df["CREDIT_ANNUITY_RATIO"] = safe_divide(
            df["AMT_CREDIT"],
            df["AMT_ANNUITY"]
        )

    if "AMT_GOODS_PRICE" in df.columns and "AMT_CREDIT" in df.columns:

        df["GOODS_CREDIT_RATIO"] = safe_divide(
            df["AMT_GOODS_PRICE"],
            df["AMT_CREDIT"]
        )

        df["CREDIT_GOODS_GAP"] = (
            df["AMT_CREDIT"] -
            df["AMT_GOODS_PRICE"]
        )

    if (
        "AMT_INCOME_TOTAL" in df.columns
        and "CNT_FAM_MEMBERS" in df.columns
    ):

        df["INCOME_PER_FAMILY_MEMBER"] = safe_divide(
            df["AMT_INCOME_TOTAL"],
            df["CNT_FAM_MEMBERS"]
        )

    if (
        "AMT_CREDIT" in df.columns
        and "CNT_FAM_MEMBERS" in df.columns
    ):

        df["CREDIT_PER_FAMILY_MEMBER"] = safe_divide(
            df["AMT_CREDIT"],
            df["CNT_FAM_MEMBERS"]
        )


    # =========================================================
    # 3. AGE / EMPLOYMENT FEATURES
    # =========================================================

    if "AGE_YEARS" in df.columns:

        df["AGE_SQUARED"] = (
            df["AGE_YEARS"] ** 2
        )

        df["AGE_CUBED"] = (
            df["AGE_YEARS"] ** 3
        )

    if (
        "DAYS_EMPLOYED" in df.columns
        and "AGE_YEARS" in df.columns
    ):

        employment_years = (
            df["DAYS_EMPLOYED"].abs() / 365.25
        )

        df["EMPLOYMENT_YEARS"] = employment_years

        df["EMPLOYMENT_AGE_RATIO"] = safe_divide(
            employment_years,
            df["AGE_YEARS"]
        )

        df["AGE_MINUS_EMPLOYMENT"] = (
            df["AGE_YEARS"] -
            employment_years
        )

    if (
        "DAYS_REGISTRATION" in df.columns
        and "AGE_YEARS" in df.columns
    ):

        registration_years = (
            df["DAYS_REGISTRATION"].abs() / 365.25
        )

        df["REGISTRATION_AGE_RATIO"] = safe_divide(
            registration_years,
            df["AGE_YEARS"]
        )


    # =========================================================
    # 4. CREDIT / INCOME INTERACTIONS
    # =========================================================

    if (
        "CREDIT_INCOME_RATIO" in df.columns
        and "EXT_SOURCE_MEAN" in df.columns
    ):

        df["CREDIT_BURDEN_EXT_RISK"] = (
            df["CREDIT_INCOME_RATIO"] *
            (1.0 - df["EXT_SOURCE_MEAN"])
        )

    if (
        "AMT_ANNUITY" in df.columns
        and "EXT_SOURCE_MEAN" in df.columns
    ):

        df["ANNUITY_EXT_RISK"] = (
            df["AMT_ANNUITY"] *
            (1.0 - df["EXT_SOURCE_MEAN"])
        )

    if (
        "CREDIT_TO_GOODS_RATIO" in df.columns
        and "EXT_SOURCE_MEAN" in df.columns
    ):

        df["CREDIT_GOODS_EXT_RISK"] = (
            df["CREDIT_TO_GOODS_RATIO"] *
            (1.0 - df["EXT_SOURCE_MEAN"])
        )


    # =========================================================
    # 5. BUREAU FEATURES
    # =========================================================

    if (
        "BUREAU_AVG_CREDIT_AMT" in df.columns
        and "AMT_CREDIT" in df.columns
    ):

        df["BUREAU_CURRENT_CREDIT_RATIO"] = safe_divide(
            df["BUREAU_AVG_CREDIT_AMT"],
            df["AMT_CREDIT"]
        )

    if (
        "BUREAU_CREDIT_COUNT" in df.columns
        and "AMT_INCOME_TOTAL" in df.columns
    ):

        df["BUREAU_COUNT_INCOME"] = (
            df["BUREAU_CREDIT_COUNT"] *
            safe_divide(
                df["AMT_CREDIT"],
                df["AMT_INCOME_TOTAL"]
            )
        )

    if (
        "BUREAU_MAX_OVERDUE" in df.columns
        and "AMT_CREDIT" in df.columns
    ):

        df["BUREAU_OVERDUE_CREDIT_RATIO"] = safe_divide(
            df["BUREAU_MAX_OVERDUE"],
            df["AMT_CREDIT"]
        )


    # =========================================================
    # 6. PREVIOUS APPLICATION FEATURES
    # =========================================================

    if (
        "PREV_APP_REFUSED_COUNT" in df.columns
        and "PREV_APP_COUNT" in df.columns
    ):

        df["PREV_APP_REFUSAL_RATE"] = safe_divide(
            df["PREV_APP_REFUSED_COUNT"],
            df["PREV_APP_COUNT"]
        )

        df["PREV_APP_APPROVAL_COUNT"] = (
            df["PREV_APP_COUNT"] -
            df["PREV_APP_REFUSED_COUNT"]
        )

    if (
        "PREV_APP_AVG_CREDIT" in df.columns
        and "AMT_CREDIT" in df.columns
    ):

        df["PREV_CURRENT_CREDIT_RATIO"] = safe_divide(
            df["PREV_APP_AVG_CREDIT"],
            df["AMT_CREDIT"]
        )

        df["PREV_CURRENT_CREDIT_GAP"] = (
            df["PREV_APP_AVG_CREDIT"] -
            df["AMT_CREDIT"]
        )

    if (
        "PREV_APP_COUNT" in df.columns
        and "AMT_INCOME_TOTAL" in df.columns
    ):

        df["PREV_APP_INCOME_INTENSITY"] = (
            df["PREV_APP_COUNT"] *
            safe_divide(
                df["AMT_CREDIT"],
                df["AMT_INCOME_TOTAL"]
            )
        )


    # =========================================================
    # 7. SOCIAL CIRCLE FEATURES
    # =========================================================

    if (
        "DEF_30_CNT_SOCIAL_CIRCLE" in df.columns
        and "OBS_30_CNT_SOCIAL_CIRCLE" in df.columns
    ):

        df["SOCIAL_DEF_30_RATE"] = safe_divide(
            df["DEF_30_CNT_SOCIAL_CIRCLE"],
            df["OBS_30_CNT_SOCIAL_CIRCLE"]
        )

    if (
        "DEF_60_CNT_SOCIAL_CIRCLE" in df.columns
        and "OBS_60_CNT_SOCIAL_CIRCLE" in df.columns
    ):

        df["SOCIAL_DEF_60_RATE"] = safe_divide(
            df["DEF_60_CNT_SOCIAL_CIRCLE"],
            df["OBS_60_CNT_SOCIAL_CIRCLE"]
        )

    if (
        "OBS_30_CNT_SOCIAL_CIRCLE" in df.columns
        and "OBS_60_CNT_SOCIAL_CIRCLE" in df.columns
    ):

        df["SOCIAL_TOTAL_OBS"] = (
            df["OBS_30_CNT_SOCIAL_CIRCLE"] +
            df["OBS_60_CNT_SOCIAL_CIRCLE"]
        )

    if (
        "DEF_30_CNT_SOCIAL_CIRCLE" in df.columns
        and "DEF_60_CNT_SOCIAL_CIRCLE" in df.columns
    ):

        df["SOCIAL_TOTAL_DEF"] = (
            df["DEF_30_CNT_SOCIAL_CIRCLE"] +
            df["DEF_60_CNT_SOCIAL_CIRCLE"]
        )


    # =========================================================
    # 8. CREDIT BUREAU REQUEST FEATURES
    # =========================================================

    bureau_req_cols = [
        c for c in [
            "AMT_REQ_CREDIT_BUREAU_HOUR",
            "AMT_REQ_CREDIT_BUREAU_DAY",
            "AMT_REQ_CREDIT_BUREAU_WEEK",
            "AMT_REQ_CREDIT_BUREAU_MON",
            "AMT_REQ_CREDIT_BUREAU_QRT",
            "AMT_REQ_CREDIT_BUREAU_YEAR"
        ]
        if c in df.columns
    ]

    if bureau_req_cols:

        df["BUREAU_REQUEST_TOTAL"] = (
            df[bureau_req_cols].sum(axis=1)
        )

        df["BUREAU_REQUEST_MEAN"] = (
            df[bureau_req_cols].mean(axis=1)
        )

        df["BUREAU_REQUEST_MAX"] = (
            df[bureau_req_cols].max(axis=1)
        )

        df["BUREAU_REQUEST_NONZERO"] = (
            (df[bureau_req_cols] > 0)
            .sum(axis=1)
        )


    # =========================================================
    # 9. DOCUMENT FEATURES
    # =========================================================

    document_cols = [
        c for c in df.columns
        if c.startswith("FLAG_DOCUMENT_")
    ]

    if document_cols:

        df["DOCUMENT_COUNT"] = (
            df[document_cols].sum(axis=1)
        )

        df["DOCUMENT_ANY"] = (
            df["DOCUMENT_COUNT"] > 0
        ).astype(int)


    # =========================================================
    # 10. PHONE / CONTACT FEATURES
    # =========================================================

    contact_cols = [
        c for c in [
            "FLAG_MOBIL",
            "FLAG_WORK_PHONE",
            "FLAG_CONT_MOBILE",
            "FLAG_PHONE",
            "FLAG_EMAIL"
        ]
        if c in df.columns
    ]

    if contact_cols:

        df["CONTACT_METHOD_COUNT"] = (
            df[contact_cols].sum(axis=1)
        )


    # =========================================================
    # 11. REGION RISK FEATURES
    # =========================================================

    if (
        "REGION_RATING_CLIENT" in df.columns
        and "REGION_RATING_CLIENT_W_CITY" in df.columns
    ):

        df["REGION_RATING_AVG"] = (
            df[
                [
                    "REGION_RATING_CLIENT",
                    "REGION_RATING_CLIENT_W_CITY"
                ]
            ].mean(axis=1)
        )

        df["REGION_RATING_DIFF"] = (
            df["REGION_RATING_CLIENT"] -
            df["REGION_RATING_CLIENT_W_CITY"]
        )


    # =========================================================
    # 12. REGION / CITY DISCREPANCY
    # =========================================================

    region_mismatch_cols = [
        c for c in [
            "REG_REGION_NOT_LIVE_REGION",
            "REG_REGION_NOT_WORK_REGION",
            "LIVE_REGION_NOT_WORK_REGION",
            "REG_CITY_NOT_LIVE_CITY",
            "REG_CITY_NOT_WORK_CITY",
            "LIVE_CITY_NOT_WORK_CITY"
        ]
        if c in df.columns
    ]

    if region_mismatch_cols:

        df["TOTAL_REGION_MISMATCH"] = (
            df[region_mismatch_cols].sum(axis=1)
        )


    # =========================================================
    # 13. PROPERTY QUALITY AGGREGATES
    # =========================================================

    property_avg_cols = [
        c for c in df.columns
        if c.endswith("_AVG")
    ]

    property_mode_cols = [
        c for c in df.columns
        if c.endswith("_MODE")
    ]

    property_medi_cols = [
        c for c in df.columns
        if c.endswith("_MEDI")
    ]

    if property_avg_cols:

        df["PROPERTY_AVG_MEAN"] = (
            df[property_avg_cols].mean(axis=1)
        )

        df["PROPERTY_AVG_STD"] = (
            df[property_avg_cols].std(axis=1)
        )

        df["PROPERTY_AVG_MISSING"] = (
            df[property_avg_cols].isna().sum(axis=1)
        )

    if property_mode_cols:

        df["PROPERTY_MODE_MEAN"] = (
            df[property_mode_cols].mean(axis=1)
        )

    if property_medi_cols:

        df["PROPERTY_MEDI_MEAN"] = (
            df[property_medi_cols].mean(axis=1)
        )


    # =========================================================
    # 14. LOG TRANSFORMS
    # =========================================================

    amount_cols = [
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
        "BUREAU_AVG_CREDIT_AMT",
        "BUREAU_MAX_OVERDUE",
        "PREV_APP_AVG_CREDIT"
    ]

    for col in amount_cols:

        if col in df.columns:

            df[f"{col}_LOG"] = np.log1p(
                df[col].clip(lower=0)
            )


    # =========================================================
    # 15. MISSINGNESS FEATURES
    # =========================================================

    important_cols = [
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
        "OWN_CAR_AGE",
        "BUREAU_CREDIT_COUNT",
        "BUREAU_AVG_CREDIT_AMT",
        "BUREAU_MAX_OVERDUE",
        "PREV_APP_COUNT",
        "PREV_APP_AVG_CREDIT"
    ]

    available_important = [
        c for c in important_cols
        if c in df.columns
    ]

    if available_important:

        df["IMPORTANT_FEATURE_MISSING_COUNT"] = (
            df[available_important].isna().sum(axis=1)
        )


    # =========================================================
    # CLEANUP
    # =========================================================

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return df


# =============================================================
# LOAD
# =============================================================

print("=" * 80)
print("FEATURE ENGINEERING V2")
print("=" * 80)

X_train = pd.read_csv(TRAIN_X)
X_test = pd.read_csv(TEST_X)

print(f"\nOriginal train shape: {X_train.shape}")
print(f"Original test shape : {X_test.shape}")


# =============================================================
# CREATE FEATURES
# =============================================================

X_train_fe = add_features(X_train)
X_test_fe = add_features(X_test)


# =============================================================
# ALIGN COLUMNS
# =============================================================

X_test_fe = X_test_fe.reindex(
    columns=X_train_fe.columns
)


# =============================================================
# SAVE
# =============================================================

X_train_fe.to_csv(
    TRAIN_OUT,
    index=False
)

X_test_fe.to_csv(
    TEST_OUT,
    index=False
)


# =============================================================
# SUMMARY
# =============================================================

new_features = [
    c for c in X_train_fe.columns
    if c not in X_train.columns
]

print("\n" + "=" * 80)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 80)

print(f"\nOriginal features : {X_train.shape[1]}")
print(f"New features      : {len(new_features)}")
print(f"Final features    : {X_train_fe.shape[1]}")

print("\nNew features:")

for feature in new_features:
    print(feature)

print("\nFinal train shape:", X_train_fe.shape)
print("Final test shape :", X_test_fe.shape)

print("\nSaved:")
print(TRAIN_OUT)
print(TEST_OUT)
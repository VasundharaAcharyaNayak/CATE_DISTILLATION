from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================
# PIMA variables where zero is treated as missing
# =============================================================

PIMA_ZERO_AS_MISSING = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]


# =============================================================
# Load observational dataset
# =============================================================

def load_observational_data(config):

    data_path = Path(
        config["experiment"]["data_path"]
    )

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    return pd.read_csv(data_path)


# =============================================================
# Prepare PIMA analytic cohort
# =============================================================

def prepare_pima_cohort(df, config):

    df = df.copy()

    # ---------------------------------------------------------
    # 1. Treat conventional impossible-zero measurements
    #    as missing
    # ---------------------------------------------------------

    for col in PIMA_ZERO_AS_MISSING:

        if col in df.columns:

            df[col] = df[col].replace(
                0,
                np.nan,
            )

    # ---------------------------------------------------------
    # 2. Read treatment and outcome definitions from config
    # ---------------------------------------------------------

    treatment = config[
        "experiment"
    ][
        "treatment"
    ]

    outcome = config[
        "experiment"
    ][
        "outcome"
    ]

    treatment_col = treatment[
        "column"
    ]

    source_col = treatment[
        "source_column"
    ]

    threshold = float(
        treatment[
            "threshold"
        ]
    )

    outcome_col = outcome[
        "column"
    ]

    # ---------------------------------------------------------
    # 3. Construct adverse-state treatment indicator
    # ---------------------------------------------------------

    df[
        treatment_col
    ] = np.where(

        df[
            source_col
        ].notna(),

        (
            df[
                source_col
            ]
            >= threshold
        ).astype(int),

        np.nan,
    )

    # ---------------------------------------------------------
    # 4. Collect required covariates
    # ---------------------------------------------------------

    covariates = config[
        "covariates"
    ]

    required_covariates = list(
        dict.fromkeys(
            covariates[
                "adjustment"
            ]
            + covariates[
                "effect_modifiers"
            ]
            + covariates[
                "clustering"
            ]
        )
    )

    required_columns = (
        required_covariates
        + [
            outcome_col,
            treatment_col,
        ]
    )

    # ---------------------------------------------------------
    # 5. Policy-specific complete-case cohort
    # ---------------------------------------------------------

    analysis_df = (
        df
        .dropna(
            subset=required_columns
        )
        .copy()
    )

    analysis_df[
        treatment_col
    ] = (
        analysis_df[
            treatment_col
        ]
        .astype(int)
    )

    return analysis_df


# =============================================================
# Prepare NHANES smoking-history analytic cohort
# =============================================================

def prepare_smoking_cohort(df, config):
   

    df = df.copy()

    # ---------------------------------------------------------
    # 1. Treatment and outcome columns
    # ---------------------------------------------------------

    treatment_col = config[
        "experiment"
    ][
        "treatment"
    ][
        "column"
    ]

    outcome_col = config[
        "experiment"
    ][
        "outcome"
    ][
        "column"
    ]

    # ---------------------------------------------------------
    # 2. Required smoking covariates
    # ---------------------------------------------------------

    covariates = config[
        "covariates"
    ]

    required_covariates = list(
        dict.fromkeys(
            covariates[
                "adjustment"
            ]
            + covariates[
                "effect_modifiers"
            ]
            + covariates.get(
                "clustering",
                [],
            )
        )
    )

    required_columns = (
        required_covariates
        + [
            outcome_col,
            treatment_col,
        ]
    )

    # ---------------------------------------------------------
    # 3. Check required columns exist
    # ---------------------------------------------------------

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Smoking dataset is missing required columns: "
            f"{missing_columns}"
        )

    # ---------------------------------------------------------
    # 4. Complete-case analytic cohort
    # ---------------------------------------------------------

    analysis_df = (
        df
        .dropna(
            subset=required_columns
        )
        .copy()
    )

    # ---------------------------------------------------------
    # 5. Ensure binary treatment/outcome coding
    # ---------------------------------------------------------

    analysis_df[
        treatment_col
    ] = (
        analysis_df[
            treatment_col
        ]
        .astype(int)
    )

    analysis_df[
        outcome_col
    ] = (
        analysis_df[
            outcome_col
        ]
        .astype(int)
    )

    return analysis_df

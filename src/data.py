from pathlib import Path

import numpy as np
import pandas as pd


def load_observational_data(config):
    """
    Load an observational dataset using its YAML configuration.
    """
    data_path = Path(config["experiment"]["data_path"])

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    return pd.read_csv(data_path)


def prepare_pima_cohort(df, config):
    """
    Construct a PIMA policy-specific complete-case cohort.

    The treatment indicator is created from the source variable and
    threshold specified in the experiment YAML file.
    """
    df = df.copy()

    treatment = config["experiment"]["treatment"]
    outcome = config["experiment"]["outcome"]

    treatment_col = treatment["column"]
    source_col = treatment["source_column"]
    threshold = treatment["threshold"]
    outcome_col = outcome["column"]

    # Construct adverse-state indicator.
    df[treatment_col] = np.where(
        df[source_col].notna(),
        (df[source_col] >= threshold).astype(int),
        np.nan,
    )

    # Collect all covariates required by this experiment.
    covariates = config["covariates"]

    required_covariates = list(
        dict.fromkeys(
            covariates["adjustment"]
            + covariates["effect_modifiers"]
            + covariates["clustering"]
        )
    )

    required_columns = (
        required_covariates
        + [outcome_col, treatment_col]
    )

    # Policy-specific complete-case cohort.
    analysis_df = (
        df
        .dropna(subset=required_columns)
        .copy()
    )

    analysis_df[treatment_col] = (
        analysis_df[treatment_col].astype(int)
    )

    return analysis_df

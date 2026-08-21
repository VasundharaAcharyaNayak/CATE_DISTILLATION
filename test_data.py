from src.config import load_config
from src.data import (
    load_observational_data,
    prepare_pima_cohort,
)


config = load_config("config/glucose.yaml")

df = load_observational_data(config)

analysis_df = prepare_pima_cohort(
    df,
    config,
)

print("Experiment:", config["experiment"]["id"])
print("Analytic N:", len(analysis_df))

treatment_col = config["experiment"]["treatment"]["column"]

print(
    "Eligible T=1:",
    int(analysis_df[treatment_col].sum()),
)

print(
    "Adjustment variables:",
    config["covariates"]["adjustment"],
)

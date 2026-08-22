from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from experiments.run_bmi import run_bmi
from experiments.run_glucose import run_glucose
from experiments.run_smoking import run_smoking

from src.dr_evaluator import (
    dr_selective_shift_values_binary_y,
)

from sensitivity.outcome_nuisance_histgb import (
    dr_selective_shift_values_histgb_outcome,
)


# =============================================================
# Experiment registry
# =============================================================

EXPERIMENT_RUNNERS = {
    "bmi": run_bmi,
    "glucose": run_glucose,
    "smoking": run_smoking,
}


EXPECTED_POLICIES = [
    "K-means",
    "Causal Tree",
    "CF teacher",
    "CF DT student",
    "CF RF student",
    "BCF teacher",
    "BCF DT student",
    "BCF RF student",
]


# =============================================================
# One experiment
# =============================================================

def run_histgb_sensitivity(
    experiment,
    *,
    split_seed=None,
    verbose=True,
):
   

    experiment = str(
        experiment
    ).lower()

    if experiment not in EXPERIMENT_RUNNERS:

        raise ValueError(
            f"Unknown experiment {experiment!r}. "
            f"Choose from {list(EXPERIMENT_RUNNERS)}."
        )

    runner = EXPERIMENT_RUNNERS[
        experiment
    ]

    # =========================================================
    # 1. Reconstruct the primary experiment
    #
    # No bootstrap inference is needed here.
    # =========================================================

    base = runner(
        split_seed=split_seed,
        run_inference=False,
        verbose=False,
    )

    config = base[
        "config"
    ]

    # =========================================================
    # 2. Freeze exactly the eight learned policies
    #
    # Do not include No shift / Shift all.
    # =========================================================

    learned = base[
        "learned_policies"
    ]

    missing = [
        name
        for name
        in EXPECTED_POLICIES
        if name not in learned
    ]

    if missing:

        raise RuntimeError(
            "Missing frozen policies: "
            f"{missing}"
        )

    policies = {

        name:
            np.asarray(
                learned[
                    name
                ],
                dtype=float,
            ).reshape(-1)

        for name
        in EXPECTED_POLICIES
    }

    Z = np.asarray(
        base[
            "Z_evaluation"
        ],
        dtype=float,
    )

    T = np.asarray(
        base[
            "T_evaluation"
        ],
        dtype=int,
    ).reshape(-1)

    Y = np.asarray(
        base[
            "Y_evaluation"
        ],
        dtype=float,
    ).reshape(-1)

    dr_folds = int(
        config[
            "evaluation"
        ][
            "dr_folds"
        ]
    )

    trim = float(
        config[
            "evaluation"
        ][
            "overlap_trim"
        ]
    )

    crossfit_seed = int(
        config[
            "causal_forest"
        ][
            "seed"
        ]
    )

    # =========================================================
    # 3. PRIMARY evaluator
    #
    # e(Z)   = logistic
    # mu0(Z) = logistic
    # =========================================================

    (
        primary_risks,
        primary_retained,
    ) = dr_selective_shift_values_binary_y(

        Z=Z,
        T=T,
        Y=Y,

        policies=
            policies,

        n_splits=
            dr_folds,

        trim=
            trim,

        seed=
            crossfit_seed,
    )

    # =========================================================
    # 4. HISTGRADIENTBOOSTING sensitivity
    #
    # e(Z)   = SAME logistic model
    # mu0(Z) = HistGradientBoosting
    # =========================================================

    (
        histgb_risks,
        histgb_retained,
    ) = dr_selective_shift_values_histgb_outcome(

        Z=Z,
        T=T,
        Y=Y,

        policies=
            policies,

        n_splits=
            dr_folds,

        trim=
            trim,

        seed=
            crossfit_seed,
    )

    # =========================================================
    # 5. Utility-change table
    # =========================================================

    rows = []

    for name in EXPECTED_POLICIES:

        primary_utility = float(
            1.0
            - primary_risks[
                name
            ]
        )

        histgb_utility = float(
            1.0
            - histgb_risks[
                name
            ]
        )

        difference = float(
            histgb_utility
            - primary_utility
        )

        rows.append(
            {
                "Policy":
                    name,

                "Logistic outcome utility":
                    primary_utility,

                "HistGradientBoosting outcome utility":
                    histgb_utility,

                "Difference (HGB - Logistic)":
                    difference,

                "Absolute difference":
                    abs(
                        difference
                    ),
            }
        )

    results = pd.DataFrame(
        rows
    )

    # =========================================================
    # 6. Rank diagnostics
    # =========================================================

    results[
        "Logistic rank"
    ] = (
        results[
            "Logistic outcome utility"
        ]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )

    results[
        "HGB rank"
    ] = (
        results[
            "HistGradientBoosting outcome utility"
        ]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )

    # =========================================================
    # 7. Summary
    # =========================================================

    summary = {

        "Experiment":
            experiment,

        "Primary retained fraction":
            float(
                primary_retained
            ),

        "HistGB retained fraction":
            float(
                histgb_retained
            ),

        "Mean absolute utility difference":
            float(
                results[
                    "Absolute difference"
                ].mean()
            ),

        "Maximum absolute utility difference":
            float(
                results[
                    "Absolute difference"
                ].max()
            ),
    }

    # =========================================================
    # 8. Display
    # =========================================================

    if verbose:

        print(
            "\n"
            + "=" * 65
        )

        print(
            f"{experiment.upper()}: "
            "OUTCOME NUISANCE-MODEL SENSITIVITY"
        )

        print(
            "=" * 65
        )

        print(
            results
            .round(6)
            .to_string(
                index=False
            )
        )

        print(
            "\nPrimary logistic retained fraction:",
            round(
                primary_retained,
                6,
            ),
        )

        print(
            "HistGradientBoosting retained fraction:",
            round(
                histgb_retained,
                6,
            ),
        )

        print(
            "Mean absolute utility difference:",
            round(
                summary[
                    "Mean absolute utility difference"
                ],
                6,
            ),
        )

        print(
            "Maximum absolute utility difference:",
            round(
                summary[
                    "Maximum absolute utility difference"
                ],
                6,
            ),
        )

    return {

        "experiment":
            experiment,

        "base":
            base,

        "policies":
            policies,

        "primary_risks":
            primary_risks,

        "histgb_risks":
            histgb_risks,

        "results":
            results,

        "summary":
            summary,
    }


# =============================================================
# All observational experiments
# =============================================================

def run_all_histgb_sensitivities(
    *,
    verbose=True,
):

    results = {}

    for experiment in (
        "bmi",
        "glucose",
        "smoking",
    ):

        results[
            experiment
        ] = run_histgb_sensitivity(
            experiment,
            verbose=verbose,
        )

    return results


# =============================================================
# Command line
# =============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment",
        choices=[
            "bmi",
            "glucose",
            "smoking",
            "all",
        ],
        default="all",
    )

    args = parser.parse_args()

    if args.experiment == "all":

        run_all_histgb_sensitivities(
            verbose=True
        )

    else:

        run_histgb_sensitivity(
            args.experiment,
            verbose=True,
        )


if __name__ == "__main__":
    main()

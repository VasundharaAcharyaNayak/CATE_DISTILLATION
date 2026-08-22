from __future__ import annotations

import argparse
import pandas as pd

from experiments.run_bmi import run_bmi
from experiments.run_glucose import run_glucose
from experiments.run_smoking import run_smoking

from sensitivity.score_extension import (
    evaluate_score_extension_sensitivity,
)


# =============================================================
# Experiment registry
# =============================================================

EXPERIMENT_RUNNERS = {
    "bmi": run_bmi,
    "glucose": run_glucose,
    "smoking": run_smoking,
}


# =============================================================
# One experiment
# =============================================================

def run_score_extension(
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

 

    base = runner(
        split_seed=split_seed,
        run_inference=False,
        verbose=False,
    )

    config = base[
        "config"
    ]

    # =========================================================
    # 2. Six score-based policies only
    #
    # K-means / Causal Tree are NOT part of this sensitivity.
    # =========================================================

    score_vectors = base[
        "score_vectors"
    ]

    original_policies = base[
        "original_score_policies"
    ]

    expected_names = {
        "CF teacher",
        "CF DT student",
        "CF RF student",
        "BCF teacher",
        "BCF DT student",
        "BCF RF student",
    }

    if set(
        score_vectors
    ) != expected_names:

        raise RuntimeError(
            "Unexpected score-policy set. "
            f"Observed: {sorted(score_vectors)}"
        )

    if set(
        original_policies
    ) != expected_names:

        raise RuntimeError(
            "Unexpected original-policy set. "
            f"Observed: {sorted(original_policies)}"
        )

    # =========================================================
    # 3. Run threshold-extension sensitivity
    #
    # The sensitivity module:
    #
    # - preserves eligible allocation exactly
    # - extends only controls
    # - evaluates ORIGINAL + THRESHOLD jointly
    # - uses paired bootstrap changes
    # =========================================================

    sensitivity = (
        evaluate_score_extension_sensitivity(

            Z=
                base[
                    "Z_evaluation"
                ],

            T=
                base[
                    "T_evaluation"
                ],

            Y=
                base[
                    "Y_evaluation"
                ],

            score_vectors=
                score_vectors,

            original_policies=
                original_policies,

            q=
                float(
                    config[
                        "policy"
                    ][
                        "budget_fraction"
                    ]
                ),

            min_benefit=
                float(
                    config[
                        "policy"
                    ].get(
                        "min_benefit",
                        0.0,
                    )
                ),

            B=
                int(
                    config[
                        "evaluation"
                    ][
                        "bootstrap_replicates"
                    ]
                ),

            bootstrap_seed=
                int(
                    config[
                        "evaluation"
                    ][
                        "bootstrap_seed"
                    ]
                ),

            dr_folds=
                int(
                    config[
                        "evaluation"
                    ][
                        "dr_folds"
                    ]
                ),

            trim=
                float(
                    config[
                        "evaluation"
                    ][
                        "overlap_trim"
                    ]
                ),

            crossfit_seed=
                int(
                    config[
                        "causal_forest"
                    ][
                        "seed"
                    ]
                ),
        )
    )

    # =========================================================
    # 4. Display whatever tables the sensitivity module returns
    #
    # We deliberately do not require particular result-key names
    # here, so the runner stays decoupled from display details.
    # =========================================================

    if verbose:

        print(
            "\n"
            + "=" * 65
        )

        print(
            f"{experiment.upper()} "
            "SCORE-EXTENSION SENSITIVITY"
        )

        print(
            "=" * 65
        )

        if isinstance(
            sensitivity,
            pd.DataFrame,
        ):

            print(
                sensitivity
                .round(6)
                .to_string(
                    index=False
                )
            )

        elif isinstance(
            sensitivity,
            dict,
        ):

            for key, value in sensitivity.items():

                if isinstance(
                    value,
                    pd.DataFrame,
                ):

                    print(
                        f"\n--- {key} ---"
                    )

                    print(
                        value
                        .round(6)
                        .to_string(
                            index=False
                        )
                    )

        print(
            "\nScore-extension sensitivity completed."
        )

    return {
        "experiment":
            experiment,

        "base":
            base,

        "sensitivity":
            sensitivity,
    }


# =============================================================
# Run all observational experiments
# =============================================================

def run_all_score_extensions(
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
        ] = run_score_extension(
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

        run_all_score_extensions(
            verbose=True
        )

    else:

        run_score_extension(
            args.experiment,
            verbose=True,
        )


if __name__ == "__main__":
    main()

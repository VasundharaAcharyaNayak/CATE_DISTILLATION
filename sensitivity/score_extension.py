from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.inference import bootstrap_policy_evaluation


# =============================================================
# Threshold-extension sensitivity policy
# =============================================================

def extend_policy_to_controls_by_threshold(
    score: Sequence[float],
    treatment: Sequence[int],
    original_policy: Sequence[float],
    *,
    q=0.70,
    min_benefit=0.0,
):
    

    score = np.asarray(
        score,
        dtype=float,
    ).reshape(-1)

    T = np.asarray(
        treatment,
        dtype=int,
    ).reshape(-1)

    original = np.asarray(
        original_policy,
        dtype=float,
    ).reshape(-1)

    if not (
        len(score)
        == len(T)
        == len(original)
    ):
        raise ValueError(
            "score, treatment, and original_policy "
            "must have equal length."
        )

    if not np.all(
        np.isin(T, [0, 1])
    ):
        raise ValueError(
            "treatment must contain only 0 and 1."
        )

    if np.any(
        (original < 0.0)
        | (original > 1.0)
    ):
        raise ValueError(
            "original_policy must lie in [0,1]."
        )

    eligible = (
        T == 1
    )

    controls = (
        T == 0
    )

    n_eligible = int(
        eligible.sum()
    )

    target_count = int(
        round(
            float(q)
            * n_eligible
        )
    )

    original_expected = float(
        original[
            eligible
        ].sum()
    )

    # Start with s(X)=0 for controls.
    extended = np.zeros(
        len(score),
        dtype=float,
    )

    selected_eligible = (
        eligible
        & (
            original > 0.0
        )
    )

    # =========================================================
    # Case 1:
    # No eligible individual receives positive allocation
    # =========================================================

    if not np.any(
        selected_eligible
    ):

        cutoff = np.nan

        boundary_probability = (
            0.0
        )

        positivity_binding = (
            False
        )

    # =========================================================
    # Case 2:
    # Positive-benefit restriction binds before q capacity
    # =========================================================

    elif (
        original_expected
        < target_count - 1e-10
    ):

        cutoff = float(
            min_benefit
        )

        boundary_probability = (
            1.0
        )

        positivity_binding = (
            True
        )

        extended[
            controls
            & (
                score
                > float(
                    min_benefit
                )
            )
        ] = 1.0

    # =========================================================
    # Case 3:
    # Budget determines the operational score boundary
    # =========================================================

    else:

        positivity_binding = (
            False
        )

        # Lowest score receiving positive allocation among
        # eligible T=1 individuals.
        cutoff = float(
            np.min(
                score[
                    selected_eligible
                ]
            )
        )

        at_boundary_eligible = (
            eligible
            & np.isclose(
                score,
                cutoff,
                rtol=0.0,
                atol=1e-12,
            )
        )

        boundary_values = (
            original[
                at_boundary_eligible
            ]
        )

        boundary_probability = float(
            np.mean(
                boundary_values
            )
        )

        # Full extension above the boundary.
        extended[
            controls
            & (
                score > cutoff
            )
        ] = 1.0

        # Same fractional probability at the boundary.
        extended[
            controls
            & np.isclose(
                score,
                cutoff,
                rtol=0.0,
                atol=1e-12,
            )
        ] = (
            boundary_probability
        )

    # =========================================================
    # CRITICAL:
    # preserve the eligible allocation EXACTLY
    # =========================================================

    extended[
        eligible
    ] = original[
        eligible
    ]

    if n_eligible > 0:

        max_eligible_difference = float(
            np.max(
                np.abs(
                    original[
                        eligible
                    ]
                    - extended[
                        eligible
                    ]
                )
            )
        )

    else:

        max_eligible_difference = (
            0.0
        )

    metadata = {

        "Eligible N":
            n_eligible,

        "Target count":
            target_count,

        "Original expected shifted eligible":
            original_expected,

        "Sensitivity expected shifted eligible":
            float(
                extended[
                    eligible
                ].sum()
            ),

        "Same eligible allocation":
            bool(
                np.allclose(
                    original[
                        eligible
                    ],
                    extended[
                        eligible
                    ],
                    atol=1e-12,
                    rtol=0.0,
                )
            ),

        "Max abs difference among eligible":
            max_eligible_difference,

        "Original s(X) among T=0":
            float(
                original[
                    controls
                ].sum()
            ),

        "Sensitivity s(X) among T=0":
            float(
                extended[
                    controls
                ].sum()
            ),

        "Sensitivity cutoff":
            cutoff,

        "Boundary probability":
            boundary_probability,

        "Positive-benefit gate binding":
            positivity_binding,
    }

    return (
        extended,
        metadata,
    )


# =============================================================
# Build threshold-extended policies
# =============================================================

def build_score_extension_policies(
    score_vectors: Mapping[
        str,
        Sequence[float],
    ],
    original_policies: Mapping[
        str,
        Sequence[float],
    ],
    treatment,
    *,
    q=0.70,
    min_benefit=0.0,
):
   

    if (
        set(score_vectors)
        != set(original_policies)
    ):
        raise ValueError(
            "score_vectors and original_policies must "
            "contain identical model names."
        )

    extended_policies = {}

    allocation_rows = []

    for name in score_vectors:

        (
            extended,
            metadata,
        ) = (
            extend_policy_to_controls_by_threshold(
                score=
                    score_vectors[
                        name
                    ],

                treatment=
                    treatment,

                original_policy=
                    original_policies[
                        name
                    ],

                q=
                    q,

                min_benefit=
                    min_benefit,
            )
        )

        # This sensitivity analysis is only valid if eligible
        # allocation remains exactly unchanged.
        if not metadata[
            "Same eligible allocation"
        ]:

            raise RuntimeError(
                "Threshold extension changed the eligible "
                f"allocation for {name}."
            )

        extended_policies[
            name
        ] = extended

        allocation_rows.append(
            {
                "Model":
                    name,
                **metadata,
            }
        )

    allocation_check = (
        pd.DataFrame(
            allocation_rows
        )
    )

    return (
        extended_policies,
        allocation_check,
    )


# =============================================================
# Evaluate original versus threshold-extended policies
# =============================================================

def evaluate_score_extension_sensitivity(
    Z,
    T,
    Y,
    score_vectors,
    original_policies,
    *,
    q=0.70,
    min_benefit=0.0,
    B=500,
    bootstrap_seed=0,
    dr_folds=5,
    trim=0.05,
    crossfit_seed=7,
):
  

    # =========================================================
    # 1. Build threshold-extended policies
    # =========================================================

    (
        extended_policies,
        allocation_check,
    ) = (
        build_score_extension_policies(
            score_vectors=
                score_vectors,

            original_policies=
                original_policies,

            treatment=
                T,

            q=
                q,

            min_benefit=
                min_benefit,
        )
    )

    # =========================================================
    # 2. Put ORIGINAL and THRESHOLD versions together
    # =========================================================

    comparison_policies = {}

    for name in score_vectors:

        comparison_policies[
            f"{name} ORIGINAL"
        ] = np.asarray(
            original_policies[
                name
            ],
            dtype=float,
        )

        comparison_policies[
            f"{name} THRESHOLD"
        ] = np.asarray(
            extended_policies[
                name
            ],
            dtype=float,
        )

    # =========================================================
    # 3. Joint DR evaluation + paired bootstrap
    # =========================================================

    (
        policy_results,
        bootstrap_df,
        point_values,
    ) = (
        bootstrap_policy_evaluation(
            Z=Z,
            T=T,
            Y=Y,

            policies=
                comparison_policies,

            B=
                B,

            seed=
                bootstrap_seed,

            n_splits=
                dr_folds,

            trim=
                trim,

            crossfit_seed=
                crossfit_seed,
        )
    )

    # =========================================================
    # 4. Original versus extended utility differences
    # =========================================================

    result_rows = []

    for name in score_vectors:

        original_key = (
            f"{name} ORIGINAL"
        )

        threshold_key = (
            f"{name} THRESHOLD"
        )

        original_utility = float(
            1.0
            - point_values[
                original_key
            ]
        )

        threshold_utility = float(
            1.0
            - point_values[
                threshold_key
            ]
        )

        # Utility(threshold) - Utility(original)
        #
        # U = 1 - risk, therefore:
        #
        # ΔU = risk(original) - risk(threshold)
        change_draws = (
            bootstrap_df[
                original_key
            ]
            - bootstrap_df[
                threshold_key
            ]
        ).dropna().to_numpy(
            dtype=float
        )

        if len(
            change_draws
        ) == 0:

            raise RuntimeError(
                f"No paired bootstrap draws for {name}."
            )

        (
            ci_lower,
            ci_upper,
        ) = np.quantile(
            change_draws,
            [
                0.025,
                0.975,
            ],
        )

        utility_change = (
            threshold_utility
            - original_utility
        )

        result_rows.append(
            {
                "Model":
                    name,

                "Original utility":
                    original_utility,

                "Threshold-extended utility":
                    threshold_utility,

                "Utility change (threshold-original)":
                    utility_change,

                "Change CI lower":
                    float(
                        ci_lower
                    ),

                "Change CI upper":
                    float(
                        ci_upper
                    ),

                "Absolute utility change":
                    float(
                        abs(
                            utility_change
                        )
                    ),

                "Successful paired replicates":
                    int(
                        len(
                            change_draws
                        )
                    ),
            }
        )

    sensitivity_results = (
        pd.DataFrame(
            result_rows
        )
    )

    return {
        "results":
            sensitivity_results,

        "allocation_check":
            allocation_check,

        "extended_policies":
            extended_policies,

        "policy_results":
            policy_results,

        "bootstrap_risk_draws":
            bootstrap_df,

        "point_risks":
            point_values,
    }

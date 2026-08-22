import numpy as np
import pandas as pd

from src.dr_evaluator import (
    dr_selective_shift_values_binary_y,
)


# =============================================================
# Bootstrap policy evaluation
# =============================================================

def bootstrap_policy_evaluation(
    Z,
    T,
    Y,
    policies,
    B=500,
    seed=0,
    n_splits=5,
    trim=0.05,
    crossfit_seed=7,
):
   

    # ---------------------------------------------------------
    # Convert inputs
    # ---------------------------------------------------------

    Z = np.asarray(
        Z,
        dtype=float,
    )

    T = np.asarray(
        T,
        dtype=int,
    ).reshape(-1)

    Y = np.asarray(
        Y,
        dtype=float,
    ).reshape(-1)

    n = len(Y)

    if not (
        len(Z)
        == len(T)
        == n
    ):
        raise ValueError(
            "Z, T, and Y must have equal numbers of rows."
        )

    # ---------------------------------------------------------
    # Validate policies
    # ---------------------------------------------------------

    normalized_policies = {}

    for name, shift_prob in policies.items():

        s = np.asarray(
            shift_prob,
            dtype=float,
        ).reshape(-1)

        if len(s) != n:
            raise ValueError(
                f"{name!r} has policy length {len(s)}, "
                f"but evaluation N={n}."
            )

        if not np.all(
            np.isfinite(s)
        ):
            raise ValueError(
                f"{name!r} contains non-finite policy values."
            )

        if np.any(
            (s < 0.0)
            | (s > 1.0)
        ):
            raise ValueError(
                f"{name!r} contains shift probabilities "
                "outside [0, 1]."
            )

        normalized_policies[
            name
        ] = s

    if not normalized_policies:
        raise ValueError(
            "At least one policy must be supplied."
        )

    # =========================================================
    # Original-sample point estimates
    # =========================================================

    (
        point_values,
        retained_fraction,
    ) = dr_selective_shift_values_binary_y(
        Z=Z,
        T=T,
        Y=Y,
        policies=normalized_policies,
        n_splits=n_splits,
        trim=trim,
        seed=crossfit_seed,
    )

    # =========================================================
    # Joint paired bootstrap
    # =========================================================

    rng = np.random.default_rng(
        seed
    )

    draws = {
        name: []
        for name
        in normalized_policies
    }

    successful_ids = []

    for b in range(
        int(B)
    ):

        # Same evaluation-row resample for every policy.
        idx = rng.integers(
            0,
            n,
            size=n,
        )

        # Frozen policy vectors follow the sampled rows.
        boot_policies = {
            name: shift_prob[
                idx
            ]
            for name, shift_prob
            in normalized_policies.items()
        }

        try:

            (
                values_b,
                _,
            ) = dr_selective_shift_values_binary_y(
                Z=Z[
                    idx
                ],
                T=T[
                    idx
                ],
                Y=Y[
                    idx
                ],
                policies=boot_policies,
                n_splits=n_splits,
                trim=trim,
                seed=
                    crossfit_seed
                    + b
                    + 1,
            )

        except (
            RuntimeError,
            ValueError,
        ):
            # Rare bootstrap samples may not support valid
            # treatment-stratified cross-fitting.
            continue

        successful_ids.append(
            b
        )

        for name, value in values_b.items():

            draws[
                name
            ].append(
                float(value)
            )

    if not successful_ids:
        raise RuntimeError(
            "All bootstrap evaluations failed."
        )

    # Because all policies are evaluated jointly, every policy
    # should have the same number of successful bootstrap draws.
    draw_counts = {
        name: len(values)
        for name, values
        in draws.items()
    }

    if len(
        set(
            draw_counts.values()
        )
    ) != 1:
        raise RuntimeError(
            "Bootstrap draws are not aligned across policies."
        )

    # =========================================================
    # Policy-specific 95% confidence intervals
    # =========================================================

    result_rows = []

    for name in normalized_policies:

        risk_draws = np.asarray(
            draws[
                name
            ],
            dtype=float,
        )

        risk_lo, risk_hi = np.quantile(
            risk_draws,
            [
                0.025,
                0.975,
            ],
        )

        risk_point = float(
            point_values[
                name
            ]
        )

        result_rows.append(
            {
                "Policy":
                    name,

                "Policy risk":
                    risk_point,

                "Policy utility":
                    float(
                        1.0
                        - risk_point
                    ),

                # Utility = 1 - Risk, so CI limits reverse.
                "Utility CI lower":
                    float(
                        1.0
                        - risk_hi
                    ),

                "Utility CI upper":
                    float(
                        1.0
                        - risk_lo
                    ),

                "Retained after overlap trimming":
                    float(
                        retained_fraction
                    ),

                "Successful bootstrap replicates":
                    int(
                        len(
                            risk_draws
                        )
                    ),

                "Eligible evaluation count":
                    int(
                        (
                            T == 1
                        ).sum()
                    ),

                "Expected shifted eligible":
                    float(
                        normalized_policies[
                            name
                        ][
                            T == 1
                        ].sum()
                    ),
            }
        )

    results = pd.DataFrame(
        result_rows
    )

    # ---------------------------------------------------------
    # One column per policy.
    #
    # These are policy-RISK bootstrap draws.
    # They are retained so paired utility-difference CIs can
    # subsequently be calculated.
    # ---------------------------------------------------------

    bootstrap_df = pd.DataFrame(
        {
            name: np.asarray(
                values,
                dtype=float,
            )
            for name, values
            in draws.items()
        }
    )

    return (
        results,
        bootstrap_df,
        point_values,
    )


# =============================================================
# Paired policy-utility confidence intervals
# =============================================================

def paired_utility_comparisons(
    bootstrap_df,
    point_values,
    comparisons,
):
  

    rows = []

    for (
        policy_a,
        policy_b,
    ) in comparisons:

        # -----------------------------------------------------
        # Validate requested policies
        # -----------------------------------------------------

        if policy_a not in bootstrap_df.columns:
            raise ValueError(
                f"{policy_a!r} not found in bootstrap draws."
            )

        if policy_b not in bootstrap_df.columns:
            raise ValueError(
                f"{policy_b!r} not found in bootstrap draws."
            )

        if policy_a not in point_values:
            raise ValueError(
                f"{policy_a!r} not found in point estimates."
            )

        if policy_b not in point_values:
            raise ValueError(
                f"{policy_b!r} not found in point estimates."
            )

        # -----------------------------------------------------
        # Retain only paired successful bootstrap replicates
        # -----------------------------------------------------

        paired = (
            bootstrap_df[
                [
                    policy_a,
                    policy_b,
                ]
            ]
            .dropna()
        )

        if paired.empty:
            raise RuntimeError(
                "No paired bootstrap draws available for "
                f"{policy_a} versus {policy_b}."
            )

        # -----------------------------------------------------
        # Bootstrap utility difference
        #
        # U(A) - U(B)
        # = [1 - R(A)] - [1 - R(B)]
        # = R(B) - R(A)
        # -----------------------------------------------------

        diff = (
            paired[
                policy_b
            ].to_numpy(
                dtype=float
            )
            -
            paired[
                policy_a
            ].to_numpy(
                dtype=float
            )
        )

        # -----------------------------------------------------
        # Original-sample utility difference
        # -----------------------------------------------------

        utility_a = float(
            1.0
            - point_values[
                policy_a
            ]
        )

        utility_b = float(
            1.0
            - point_values[
                policy_b
            ]
        )

        point_diff = float(
            utility_a
            - utility_b
        )

        # -----------------------------------------------------
        # Percentile paired-bootstrap 95% CI
        # -----------------------------------------------------

        ci_lower, ci_upper = np.quantile(
            diff,
            [
                0.025,
                0.975,
            ],
        )

        rows.append(
            {
                "Policy A":
                    policy_a,

                "Policy B":
                    policy_b,

                "Utility A":
                    utility_a,

                "Utility B":
                    utility_b,

                "Utility difference (A-B)":
                    point_diff,

                "95% CI lower":
                    float(
                        ci_lower
                    ),

                "95% CI upper":
                    float(
                        ci_upper
                    ),

                "CI includes zero":
                    bool(
                        ci_lower
                        <= 0.0
                        <= ci_upper
                    ),

                "Successful paired bootstrap replicates":
                    int(
                        len(
                            diff
                        )
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )

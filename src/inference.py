import numpy as np
import pandas as pd

from src.dr_evaluator import (
    dr_selective_shift_values_binary_y,
)


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

    normalized_policies = {
        name: np.asarray(
            shift_prob,
            dtype=float,
        ).reshape(-1)
        for name, shift_prob
        in policies.items()
    }

    # ---------------------------------------------------------
    # Original-sample point estimates
    # ---------------------------------------------------------

    point_values, retained_fraction = (
        dr_selective_shift_values_binary_y(
            Z=Z,
            T=T,
            Y=Y,
            policies=normalized_policies,
            n_splits=n_splits,
            trim=trim,
            seed=crossfit_seed,
        )
    )

    # ---------------------------------------------------------
    # Bootstrap
    # ---------------------------------------------------------

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

        # Same evaluation-row resample for every policy
        idx = rng.integers(
            0,
            len(Y),
            size=len(Y),
        )

        # Frozen policy vectors follow the resampled rows
        boot_policies = {
            name: shift_prob[idx]
            for name, shift_prob
            in normalized_policies.items()
        }

        try:

            values_b, _ = (
                dr_selective_shift_values_binary_y(
                    Z=Z[idx],
                    T=T[idx],
                    Y=Y[idx],
                    policies=boot_policies,
                    n_splits=n_splits,
                    trim=trim,
                    seed=crossfit_seed + b + 1,
                )
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

        for (
            name,
            value,
        ) in values_b.items():

            draws[
                name
            ].append(
                value
            )

    if not successful_ids:
        raise RuntimeError(
            "All bootstrap evaluations failed."
        )

    # ---------------------------------------------------------
    # Policy-specific confidence intervals
    # ---------------------------------------------------------

    result_rows = []

    for name in normalized_policies:

        risk_draws = np.asarray(
            draws[name],
            dtype=float,
        )

        risk_lo, risk_hi = np.quantile(
            risk_draws,
            [0.025, 0.975],
        )

        risk_point = (
            point_values[name]
        )

        result_rows.append(
            {
                "Policy":
                    name,

                "Policy risk":
                    float(
                        risk_point
                    ),

                "Policy utility":
                    float(
                        1.0
                        - risk_point
                    ),

                # Utility = 1 - risk, so the risk interval
                # reverses when converted to utility.
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
                    retained_fraction,

                "Successful bootstrap replicates":
                    len(
                        risk_draws
                    ),

                "Eligible evaluation count":
                    int(
                        (T == 1).sum()
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

    # One column per policy.
    # These are POLICY-RISK draws.
    bootstrap_df = pd.DataFrame(
        {
            name: pd.Series(
                values,
                dtype=float,
            )
            for name, values
            in draws.items()
        }
    )

def paired_utility_comparisons(
    bootstrap_df,
    point_values,
    comparisons,
):
    
    rows = []

    for policy_a, policy_b in comparisons:

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
        # Use only paired successful bootstrap replicates
        # -----------------------------------------------------

        paired = (
            bootstrap_df[
                [policy_a, policy_b]
            ]
            .dropna()
        )

        if paired.empty:
            raise RuntimeError(
                f"No paired bootstrap draws available for "
                f"{policy_a} versus {policy_b}."
            )

        # -----------------------------------------------------
        # Utility(A) - Utility(B)
        #
        # Since U = 1 - risk:
        #
        # U(A) - U(B)
        # = risk(B) - risk(A)
        # -----------------------------------------------------

        diff = (
            paired[policy_b].to_numpy(dtype=float)
            - paired[policy_a].to_numpy(dtype=float)
        )

        point_diff = (
            (1.0 - point_values[policy_a])
            - (1.0 - point_values[policy_b])
        )

        # -----------------------------------------------------
        # Percentile paired-bootstrap CI
        # -----------------------------------------------------

        ci_lower, ci_upper = np.quantile(
            diff,
            [0.025, 0.975],
        )

        # -----------------------------------------------------
        # Two-sided bootstrap p-value
        #
        # Compare the fraction of paired bootstrap differences
        # on either side of zero.
        #
        # The +1 correction prevents an estimated p-value of
        # exactly zero with a finite number of replicates.
        # -----------------------------------------------------

        B_success = len(diff)

        p_lower = (
            np.sum(diff <= 0.0) + 1
        ) / (
            B_success + 1
        )

        p_upper = (
            np.sum(diff >= 0.0) + 1
        ) / (
            B_success + 1
        )

        p_raw = min(
            1.0,
            2.0 * min(
                p_lower,
                p_upper,
            ),
        )

        rows.append(
            {
                "Policy A":
                    policy_a,

                "Policy B":
                    policy_b,

                "Comparison":
                    f"{policy_a} minus {policy_b}",

                "Utility difference":
                    float(point_diff),

                "CI lower":
                    float(ci_lower),

                "CI upper":
                    float(ci_upper),

                "Raw p-value":
                    float(p_raw),

                "Successful paired replicates":
                    int(B_success),
            }
        )

    return pd.DataFrame(
        rows
    )


def holm_adjustment(
    comparisons_df,
    pvalue_column="Raw p-value",
    alpha=0.05,
):
   

    result = comparisons_df.copy()

    if pvalue_column not in result.columns:
        raise ValueError(
            f"{pvalue_column!r} is not present in the table."
        )

    p_values = result[
        pvalue_column
    ].to_numpy(
        dtype=float
    )

    if np.any(
        (p_values < 0.0)
        | (p_values > 1.0)
    ):
        raise ValueError(
            "All p-values must lie in [0, 1]."
        )

    m = len(
        p_values
    )

    if m == 0:
        result[
            "Holm-adjusted p-value"
        ] = []

        result[
            "Reject after Holm"
        ] = []

        return result

    # ---------------------------------------------------------
    # Sort p-values from smallest to largest
    # ---------------------------------------------------------

    order = np.argsort(
        p_values
    )

    ordered_p = p_values[
        order
    ]

    adjusted_ordered = np.empty(
        m,
        dtype=float,
    )

    # ---------------------------------------------------------
    # Holm adjusted p-values:
    #
    # max_{j <= i} [(m-j+1) p_(j)]
    # ---------------------------------------------------------

    running_max = 0.0

    for i, p_value in enumerate(
        ordered_p
    ):

        multiplier = (
            m - i
        )

        candidate = (
            multiplier
            * p_value
        )

        running_max = max(
            running_max,
            candidate,
        )

        adjusted_ordered[i] = min(
            running_max,
            1.0,
        )

    # Restore original comparison ordering
    adjusted = np.empty(
        m,
        dtype=float,
    )

    adjusted[
        order
    ] = adjusted_ordered

    result[
        "Holm-adjusted p-value"
    ] = adjusted

    result[
        "Reject after Holm"
    ] = (
        adjusted
        < alpha
    )

    return result

    return (
        results,
        bootstrap_df,
        point_values,
    )

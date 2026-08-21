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

    return (
        results,
        bootstrap_df,
        point_values,
    )

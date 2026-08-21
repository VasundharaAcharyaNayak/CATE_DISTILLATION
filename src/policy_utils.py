import numpy as np


def get_budget_count(n_eligible, budget_fraction):
   
    if n_eligible < 0:
        raise ValueError(
            "n_eligible must be non-negative."
        )

    if not 0 <= budget_fraction <= 1:
        raise ValueError(
            "budget_fraction must be between 0 and 1."
        )

    return int(
        round(
            budget_fraction * n_eligible
        )
    )


def individual_policy_allocation(
    benefit_scores,
    treatment,
    budget_fraction=0.70,
    positive_benefit_only=True,
    min_benefit=0.0,
):
    

    # ---------------------------------------------------------
    # Convert inputs
    # ---------------------------------------------------------

    score = np.asarray(
        benefit_scores,
        dtype=float,
    ).reshape(-1)

    T = np.asarray(
        treatment,
        dtype=int,
    ).reshape(-1)

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    if len(score) != len(T):
        raise ValueError(
            "benefit_scores and treatment must have equal length."
        )

    if not np.all(
        np.isfinite(score)
    ):
        raise ValueError(
            "benefit_scores contains non-finite values."
        )

    if not np.all(
        np.isin(T, [0, 1])
    ):
        raise ValueError(
            "treatment must contain only 0 and 1."
        )

    if not 0 <= budget_fraction <= 1:
        raise ValueError(
            "budget_fraction must lie in [0, 1]."
        )

    # ---------------------------------------------------------
    # Eligible evaluation population
    # ---------------------------------------------------------

    eligible = (
        T == 1
    )

    n_eligible = int(
        eligible.sum()
    )

    target_count = get_budget_count(
        n_eligible,
        budget_fraction,
    )

    # Start with s(X)=0.
    shift_probability = np.zeros(
        len(score),
        dtype=float,
    )

    # ---------------------------------------------------------
    # Positive-benefit diagnostics
    # ---------------------------------------------------------

    eligible_scores = score[
        eligible
    ]

    n_positive_eligible = int(
        (
            eligible_scores
            > min_benefit
        ).sum()
    )

    positive_fraction_eligible = (
        n_positive_eligible / n_eligible
        if n_eligible > 0
        else 0.0
    )

    # ---------------------------------------------------------
    # Empty policy
    # ---------------------------------------------------------

    if (
        n_eligible == 0
        or target_count == 0
    ):
        return shift_probability, {
            "n_eligible": n_eligible,
            "target_count": target_count,
            "n_positive_eligible":
                n_positive_eligible,
            "positive_fraction_eligible":
                positive_fraction_eligible,
            "expected_shifted": 0.0,
            "realized_fraction_eligible": 0.0,
            "unused_budget_n":
                float(target_count),
            "boundary_score": np.nan,
            "boundary_probability": 0.0,
            "positivity_gate_binding": bool(
                positive_benefit_only
                and n_positive_eligible
                < target_count
            ),
        }

    # ---------------------------------------------------------
    # Rank unique scores observed among eligible T=1 rows
    # ---------------------------------------------------------

    unique_scores = np.sort(
        np.unique(
            eligible_scores
        )
    )[::-1]

    selected_so_far = 0

    boundary_score = np.nan
    boundary_probability = 0.0

    # ---------------------------------------------------------
    # Sequential top-benefit allocation
    # ---------------------------------------------------------

    for value in unique_scores:

        # Positive-benefit requirement.
        if (
            positive_benefit_only
            and value <= min_benefit
        ):
            break

        # Budget is consumed only by factual eligible T=1 rows.
        eligible_at_value = (
            eligible
            & (score == value)
        )

        n_at_value = int(
            eligible_at_value.sum()
        )

        # -----------------------------------------------------
        # Entire score level fits within remaining capacity
        # -----------------------------------------------------

        if (
            selected_so_far
            + n_at_value
            <= target_count
        ):

            # Primary encoding:
            # only observations sharing this EXACT selected score
            # receive the corresponding s(X).
            shift_probability[
                score == value
            ] = 1.0

            selected_so_far += (
                n_at_value
            )

        # -----------------------------------------------------
        # Budget boundary falls inside this score level
        # -----------------------------------------------------

        else:

            remaining = (
                target_count
                - selected_so_far
            )

            if remaining > 0:

                boundary_probability = (
                    remaining
                    / n_at_value
                )

                # Every observation sharing the exact boundary
                # score receives the common fractional value.
                shift_probability[
                    score == value
                ] = (
                    boundary_probability
                )

                selected_so_far += (
                    remaining
                )

                boundary_score = float(
                    value
                )

            break

    # ---------------------------------------------------------
    # Allocation diagnostics are calculated among eligible T=1
    # ---------------------------------------------------------

    expected_shifted = float(
        shift_probability[
            eligible
        ].sum()
    )

    realized_fraction_eligible = (
        expected_shifted
        / n_eligible
        if n_eligible > 0
        else 0.0
    )

    unused_budget = float(
        max(
            target_count
            - expected_shifted,
            0.0,
        )
    )

    details = {
        "n_eligible":
            n_eligible,

        "target_count":
            target_count,

        "n_positive_eligible":
            n_positive_eligible,

        "positive_fraction_eligible":
            positive_fraction_eligible,

        "expected_shifted":
            expected_shifted,

        "realized_fraction_eligible":
            realized_fraction_eligible,

        "unused_budget_n":
            unused_budget,

        "boundary_score":
            boundary_score,

        "boundary_probability":
            boundary_probability,

        "positivity_gate_binding":
            bool(
                positive_benefit_only
                and n_positive_eligible
                < target_count
            ),
    }

    return (
        shift_probability,
        details,
    )

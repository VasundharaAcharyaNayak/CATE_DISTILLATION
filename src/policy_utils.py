import numpy as np


def get_budget_count(n_eligible, budget_fraction):
    """
    Return the maximum number of eligible individuals that can be shifted.

    The repository follows the allocation convention used in the analysis:
        target_count = round(q * n_eligible)
    """
    if n_eligible < 0:
        raise ValueError("n_eligible must be non-negative.")

    if not 0 <= budget_fraction <= 1:
        raise ValueError(
            "budget_fraction must be between 0 and 1."
        )

    return int(round(budget_fraction * n_eligible))


def individual_policy_allocation(
    benefit_scores,
    treatment,
    budget_fraction=0.70,
    positive_benefit_only=True,
    min_benefit=0.0,
):
    """
    Construct a score-based selective T=1 -> T=0 policy.

    The allocation threshold is determined using currently eligible
    individuals (T=1). The resulting shift probability s(X) is then
    defined for the full cohort according to the learned score rule.

    Individuals are ranked by predicted benefit. The budget is a maximum
    rather than a compulsory quota, so individuals with non-positive
    benefit are not included merely to exhaust capacity.

    If multiple eligible individuals share the score at the budget
    boundary, all rows in the cohort sharing that boundary score receive
    the same fractional shift probability.

    Parameters
    ----------
    benefit_scores : array-like
        Predicted intervention benefit for every individual.

    treatment : array-like
        Binary state indicator. T=1 denotes currently eligible
        individuals.

    budget_fraction : float, default=0.70
        Maximum fraction of eligible individuals that can be shifted.

    positive_benefit_only : bool, default=True
        If True, allocation stops once benefit is at or below
        min_benefit.

    min_benefit : float, default=0.0
        Minimum predicted benefit required for allocation.

    Returns
    -------
    shift_probability : np.ndarray
        Full-cohort shift probability s(X).

    details : dict
        Allocation diagnostics.
    """

    score = np.asarray(
        benefit_scores,
        dtype=float,
    ).reshape(-1)

    treatment = np.asarray(
        treatment,
        dtype=int,
    ).reshape(-1)

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    if len(score) != len(treatment):
        raise ValueError(
            "benefit_scores and treatment must have equal length."
        )

    if not np.all(np.isfinite(score)):
        raise ValueError(
            "benefit_scores contains missing or non-finite values."
        )

    if not np.all(np.isin(treatment, [0, 1])):
        raise ValueError(
            "treatment must contain only 0 and 1."
        )

    if not 0 <= budget_fraction <= 1:
        raise ValueError(
            "budget_fraction must lie in [0, 1]."
        )

    # ---------------------------------------------------------
    # Determine eligible population and budget
    # ---------------------------------------------------------

    eligible = treatment == 1

    n_eligible = int(
        eligible.sum()
    )

    target_count = get_budget_count(
        n_eligible,
        budget_fraction,
    )

    # Full-cohort s(X)
    shift_probability = np.zeros(
        len(score),
        dtype=float,
    )

    if n_eligible == 0 or target_count == 0:
        return shift_probability, {
            "n_eligible": n_eligible,
            "target_count": target_count,
            "expected_shifted": 0.0,
            "boundary_score": np.nan,
            "boundary_probability": 0.0,
            "n_positive_eligible": 0,
        }

    # ---------------------------------------------------------
    # Eligible benefit distribution
    # ---------------------------------------------------------

    eligible_scores = score[eligible]

    n_positive_eligible = int(
        (
            eligible_scores > min_benefit
        ).sum()
    )

    positive_fraction_eligible = (
        n_positive_eligible / n_eligible
        if n_eligible > 0
        else 0.0
    )

    # Rank unique eligible scores from highest to lowest.
    unique_scores = np.sort(
        np.unique(eligible_scores)
    )[::-1]

    selected_so_far = 0

    boundary_score = np.nan
    boundary_probability = 0.0

    # ---------------------------------------------------------
    # Sequential allocation
    # ---------------------------------------------------------

    for value in unique_scores:

        if (
            positive_benefit_only
            and value <= min_benefit
        ):
            break

        # Count only factual T=1 individuals when consuming budget.
        eligible_at_value = (
            eligible
            & (score == value)
        )

        n_at_value = int(
            eligible_at_value.sum()
        )

        if (
            selected_so_far + n_at_value
            <= target_count
        ):
            # IMPORTANT:
            # s(X) is defined for EVERY evaluation row sharing
            # this score, including factual T=0 observations.
            shift_probability[
                score == value
            ] = 1.0

            selected_so_far += n_at_value

        else:
            remaining = (
                target_count
                - selected_so_far
            )

            if remaining > 0:

                boundary_probability = (
                    remaining / n_at_value
                )

                # Same fractional probability for every row
                # sharing the boundary score.
                shift_probability[
                    score == value
                ] = boundary_probability

                selected_so_far += remaining

                boundary_score = float(value)

            break

    # Expected number shifted among the eligible T=1 population.
    expected_shifted = float(
        shift_probability[eligible].sum()
    )

    details = {
        "n_eligible": n_eligible,
        "target_count": target_count,
        "expected_shifted": expected_shifted,
        "boundary_score": boundary_score,
        "boundary_probability": boundary_probability,
        "n_positive_eligible": n_positive_eligible,
        "positive_fraction_eligible":
            positive_fraction_eligible,
        "positivity_gate_binding": bool(
            positive_benefit_only
            and n_positive_eligible < target_count
        ),
    }

    return shift_probability, details

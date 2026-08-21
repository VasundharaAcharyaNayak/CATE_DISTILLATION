import numpy as np


def get_budget_count(n_eligible, budget_fraction):
    """
    Maximum number of eligible individuals that can be prioritized.

    Parameters
    ----------
    n_eligible : int
        Number of currently eligible individuals.

    budget_fraction : float
        Maximum fraction of eligible individuals that may be prioritized.

    Returns
    -------
    int
        Maximum intervention count.
    """
    if not 0 <= budget_fraction <= 1:
        raise ValueError("budget_fraction must be between 0 and 1.")

    return int(np.floor(budget_fraction * n_eligible))


def individual_policy_allocation(
    benefit_scores,
    treatment,
    budget_fraction=0.70,
    positive_benefit_only=True,
    min_benefit=0.0,
):
    """
    Construct a budget-constrained individualized policy.

    The learned benefit scores are applied to the evaluation cohort.
    Currently eligible individuals (T=1) are ranked by predicted benefit,
    and up to q * N_eligible are prioritized.

    The budget is a maximum rather than a compulsory quota. Therefore,
    individuals with non-positive predicted benefit are not selected merely
    to exhaust the available capacity.

    Parameters
    ----------
    benefit_scores : array-like
        Predicted intervention benefit for each individual.

    treatment : array-like
        Binary treatment/state indicator. T=1 denotes currently eligible
        individuals.

    budget_fraction : float, default=0.70
        Maximum fraction of eligible individuals that may be prioritized.

    positive_benefit_only : bool, default=True
        If True, only individuals with benefit > min_benefit are considered.

    min_benefit : float, default=0.0
        Minimum predicted benefit required for prioritization.

    Returns
    -------
    allocation : np.ndarray
        Array of length N. Values are 1 for prioritized eligible
        individuals and 0 otherwise.

    details : dict
        Summary of the allocation.
    """
    benefit_scores = np.asarray(benefit_scores, dtype=float)
    treatment = np.asarray(treatment)

    if len(benefit_scores) != len(treatment):
        raise ValueError(
            "benefit_scores and treatment must have the same length."
        )

    # Currently eligible population
    eligible = treatment == 1
    eligible_indices = np.flatnonzero(eligible)

    n_eligible = len(eligible_indices)

    budget_count = get_budget_count(
        n_eligible=n_eligible,
        budget_fraction=budget_fraction,
    )

    # Start with nobody prioritized
    allocation = np.zeros(len(treatment), dtype=float)

    if n_eligible == 0 or budget_count == 0:
        return allocation, {
            "n_eligible": n_eligible,
            "budget_count": budget_count,
            "n_positive_eligible": 0,
            "n_prioritized": 0,
        }

    eligible_scores = benefit_scores[eligible_indices]

    # Positive-benefit restriction
    if positive_benefit_only:
        admissible_mask = eligible_scores > min_benefit
    else:
        admissible_mask = np.ones(
            len(eligible_scores),
            dtype=bool,
        )

    admissible_indices = eligible_indices[admissible_mask]
    admissible_scores = benefit_scores[admissible_indices]

    n_positive_eligible = len(admissible_indices)

    if n_positive_eligible == 0:
        return allocation, {
            "n_eligible": n_eligible,
            "budget_count": budget_count,
            "n_positive_eligible": 0,
            "n_prioritized": 0,
        }

    # Rank eligible individuals from highest to lowest predicted benefit
    order = np.argsort(-admissible_scores, kind="stable")

    ranked_indices = admissible_indices[order]

    n_prioritized = min(
        budget_count,
        n_positive_eligible,
    )

    selected_indices = ranked_indices[:n_prioritized]

    allocation[selected_indices] = 1.0

    details = {
        "n_eligible": n_eligible,
        "budget_count": budget_count,
        "n_positive_eligible": n_positive_eligible,
        "n_prioritized": n_prioritized,
    }

    return allocation, details

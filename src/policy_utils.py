import numpy as np


def get_budget_count(n_eligible, budget_fraction):
   
    if n_eligible < 0:
        raise ValueError("n_eligible must be non-negative.")

    if not 0 <= budget_fraction <= 1:
        raise ValueError("budget_fraction must be between 0 and 1.")

    return float(budget_fraction * n_eligible)


def individual_policy_allocation(
    benefit_scores,
    treatment,
    budget_fraction=0.70,
    positive_benefit_only=True,
    min_benefit=0.0,
):
 

    # ---------------------------------------------------------
    # Basic checks
    # ---------------------------------------------------------
    benefit_scores = np.asarray(benefit_scores, dtype=float)
    treatment = np.asarray(treatment)
    if benefit_scores.ndim != 1 or treatment.ndim != 1:
        raise ValueError(
            "benefit_scores and treatment must be one-dimensional."
        )

    if len(benefit_scores) != len(treatment):
        raise ValueError(
            "benefit_scores and treatment must have the same length."
        )

    if not np.all(np.isin(treatment, [0, 1])):
        raise ValueError(
            "treatment must contain only binary values 0 and 1."
        )

    if not np.all(np.isfinite(benefit_scores)):
        raise ValueError(
            "benefit_scores contains missing or non-finite values."
        )

    # We start with zero allocation for everyone.
    allocation = np.zeros(len(treatment), dtype=float)

    # ---------------------------------------------------------
    # Identify currently eligible individuals
    # ---------------------------------------------------------

    eligible_indices = np.flatnonzero(treatment == 1)

    n_eligible = len(eligible_indices)

    target_count = get_budget_count(
        n_eligible=n_eligible,
        budget_fraction=budget_fraction,
    )

    # No eligible individuals or zero intervention capacity.
    if n_eligible == 0 or target_count <= 0:
        details = {
            "n_eligible": n_eligible,
            "target_count": target_count,
            "n_positive_eligible": 0,
            "expected_allocated": 0.0,
            "boundary_score": None,
            "boundary_probability": 0.0,
        }

        return allocation, details

    # ---------------------------------------------------------
    # Apply positive-benefit restriction
    # ---------------------------------------------------------

    eligible_scores = benefit_scores[eligible_indices]

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

    # No eligible individual has admissible benefit.
    if n_positive_eligible == 0:
        details = {
            "n_eligible": n_eligible,
            "target_count": target_count,
            "n_positive_eligible": 0,
            "expected_allocated": 0.0,
            "boundary_score": None,
            "boundary_probability": 0.0,
        }

        return allocation, details

    # ---------------------------------------------------------
    # If positive-benefit individuals do not exhaust capacity,
    # allocate all of them.
    # ---------------------------------------------------------

    if n_positive_eligible <= target_count:
        allocation[admissible_indices] = 1.0

        details = {
            "n_eligible": n_eligible,
            "target_count": target_count,
            "n_positive_eligible": n_positive_eligible,
            "expected_allocated": float(allocation.sum()),
            "boundary_score": None,
            "boundary_probability": 0.0,
        }

        return allocation, details

    # ---------------------------------------------------------
    # Rank admissible eligible individuals by predicted benefit
    # ---------------------------------------------------------

    order = np.argsort(
        -admissible_scores,
        kind="stable",
    )

    ranked_indices = admissible_indices[order]
    ranked_scores = admissible_scores[order]

    # ---------------------------------------------------------
    # Determine the score at the intervention boundary
    # ---------------------------------------------------------

    boundary_position = int(np.ceil(target_count)) - 1

    boundary_score = ranked_scores[boundary_position]

    # Individuals with scores strictly greater than the boundary
    # score are fully allocated.
    above_boundary_mask = ranked_scores > boundary_score

    above_boundary_indices = ranked_indices[
        above_boundary_mask
    ]

    allocation[above_boundary_indices] = 1.0

    n_above_boundary = len(above_boundary_indices)

    # ---------------------------------------------------------
    # Identify everyone tied at the boundary
    # ---------------------------------------------------------

    boundary_mask = ranked_scores == boundary_score

    boundary_indices = ranked_indices[boundary_mask]

    n_boundary = len(boundary_indices)

    # Remaining expected capacity after fully allocating all
    # individuals above the boundary.
    remaining_capacity = (
        target_count - n_above_boundary
    )

    boundary_probability = (
        remaining_capacity / n_boundary
    )

    # Numerical protection
    boundary_probability = float(
        np.clip(
            boundary_probability,
            0.0,
            1.0,
        )
    )

    # Give every individual sharing the boundary score the
    # same fractional allocation probability.
    allocation[boundary_indices] = (
        boundary_probability
    )

    # ---------------------------------------------------------
    # Return allocation and diagnostics
    # ---------------------------------------------------------

    details = {
        "n_eligible": n_eligible,
        "target_count": target_count,
        "n_positive_eligible": n_positive_eligible,
        "expected_allocated": float(
            allocation[eligible_indices].sum()
        ),
        "boundary_score": float(boundary_score),
        "boundary_probability": boundary_probability,
        "n_above_boundary": n_above_boundary,
        "n_boundary": n_boundary,
    }

    return allocation, details

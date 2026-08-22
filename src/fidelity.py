from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import r2_score


# =============================================================
# Basic validation
# =============================================================

def _as_1d_float(
    values,
    name,
):
    arr = np.asarray(
        values,
        dtype=float,
    ).reshape(-1)

    if not np.isfinite(
        arr
    ).all():
        raise ValueError(
            f"{name} contains non-finite values."
        )

    return arr


# =============================================================
# Hard top-q policy for overlap diagnostics
# =============================================================

def hard_positive_budget_policy(
    score: Sequence[float],
    *,
    q=0.70,
    min_benefit=0.0,
    eligible_mask=None,
    tie_priority=None,
    seed=7,
):
   

    score = _as_1d_float(
        score,
        "score",
    )

    n = len(
        score
    )

    if not 0 <= q <= 1:
        raise ValueError(
            "q must lie in [0,1]."
        )

    # ---------------------------------------------------------
    # Policy-relevant population
    # ---------------------------------------------------------

    if eligible_mask is None:

        eligible = np.ones(
            n,
            dtype=bool,
        )

    else:

        eligible = np.asarray(
            eligible_mask,
            dtype=bool,
        ).reshape(-1)

        if len(
            eligible
        ) != n:
            raise ValueError(
                "eligible_mask and score must have equal length."
            )

    # ---------------------------------------------------------
    # Shared tie priority
    # ---------------------------------------------------------

    if tie_priority is None:

        tie_priority = (
            np.random.default_rng(
                int(seed)
            )
            .random(
                n
            )
        )

    else:

        tie_priority = np.asarray(
            tie_priority,
            dtype=float,
        ).reshape(-1)

        if len(
            tie_priority
        ) != n:
            raise ValueError(
                "tie_priority and score must have equal length."
            )

    n_eligible = int(
        eligible.sum()
    )

    budget_cap = int(
        round(
            q
            * n_eligible
        )
    )

    # Positive-benefit eligible candidates
    admissible = np.flatnonzero(
        eligible
        & (
            score
            > float(
                min_benefit
            )
        )
    )

    n_select = min(
        budget_cap,
        len(
            admissible
        ),
    )

    policy = np.zeros(
        n,
        dtype=int,
    )

    if n_select > 0:

        # Primary ranking:
        #     descending benefit score
        #
        # Secondary ranking:
        #     shared seeded tie priority
        order = np.lexsort(
            (
                tie_priority[
                    admissible
                ],
                -score[
                    admissible
                ],
            )
        )

        selected = admissible[
            order[
                :n_select
            ]
        ]

        policy[
            selected
        ] = 1

    metadata = {
        "eligible_n":
            n_eligible,

        "positive_eligible_n":
            int(
                len(
                    admissible
                )
            ),

        "budget_cap_n":
            budget_cap,

        "selected_n":
            int(
                policy.sum()
            ),

        "unused_budget_n":
            int(
                max(
                    budget_cap
                    - policy.sum(),
                    0,
                )
            ),

        "realized_fraction_eligible":
            (
                float(
                    policy.sum()
                    / n_eligible
                )
                if n_eligible > 0
                else 0.0
            ),
    }

    return (
        policy,
        metadata,
    )


# =============================================================
# Binary policy overlap
# =============================================================

def binary_policy_overlap(
    reference_policy,
    candidate_policy,
    *,
    mask=None,
):
   

    reference = np.asarray(
        reference_policy,
        dtype=bool,
    ).reshape(-1)

    candidate = np.asarray(
        candidate_policy,
        dtype=bool,
    ).reshape(-1)

    if len(
        reference
    ) != len(
        candidate
    ):
        raise ValueError(
            "Policies must have equal length."
        )

    if mask is not None:

        mask = np.asarray(
            mask,
            dtype=bool,
        ).reshape(-1)

        if len(
            mask
        ) != len(
            reference
        ):
            raise ValueError(
                "mask and policies must have equal length."
            )

        reference = reference[
            mask
        ]

        candidate = candidate[
            mask
        ]

    intersection = int(
        np.logical_and(
            reference,
            candidate,
        ).sum()
    )

    union = int(
        np.logical_or(
            reference,
            candidate,
        ).sum()
    )

    reference_n = int(
        reference.sum()
    )

    candidate_n = int(
        candidate.sum()
    )

    return {
        "jaccard":
            (
                float(
                    intersection
                    / union
                )
                if union > 0
                else np.nan
            ),

        "precision":
            (
                float(
                    intersection
                    / candidate_n
                )
                if candidate_n > 0
                else np.nan
            ),

        "recall":
            (
                float(
                    intersection
                    / reference_n
                )
                if reference_n > 0
                else np.nan
            ),

        "shared_selected_n":
            intersection,

        "reference_selected_n":
            reference_n,

        "candidate_selected_n":
            candidate_n,

        "disagreement_rate":
            (
                float(
                    np.logical_xor(
                        reference,
                        candidate,
                    ).mean()
                )
                if len(
                    reference
                ) > 0
                else np.nan
            ),
    }


# =============================================================
# Rank fidelity
# =============================================================

def rank_correlations(
    teacher_score,
    student_score,
    *,
    mask=None,
):
    

    teacher = _as_1d_float(
        teacher_score,
        "teacher_score",
    )

    student = _as_1d_float(
        student_score,
        "student_score",
    )

    if len(
        teacher
    ) != len(
        student
    ):
        raise ValueError(
            "Teacher and student scores must have equal length."
        )

    if mask is not None:

        mask = np.asarray(
            mask,
            dtype=bool,
        ).reshape(-1)

        if len(
            mask
        ) != len(
            teacher
        ):
            raise ValueError(
                "mask and score vectors must have equal length."
            )

        teacher = teacher[
            mask
        ]

        student = student[
            mask
        ]

    if (
        len(
            teacher
        ) < 2
        or np.std(
            teacher
        ) < 1e-12
        or np.std(
            student
        ) < 1e-12
    ):

        return {
            "spearman":
                np.nan,

            "kendall_tau_b":
                np.nan,

            "n_ranked":
                int(
                    len(
                        teacher
                    )
                ),
        }

    spearman_value = (
        spearmanr(
            teacher,
            student,
            nan_policy="raise",
        ).statistic
    )

    kendall_value = (
        kendalltau(
            teacher,
            student,
            variant="b",
            nan_policy="raise",
        ).statistic
    )

    return {
        "spearman":
            float(
                spearman_value
            ),

        "kendall_tau_b":
            float(
                kendall_value
            ),

        "n_ranked":
            int(
                len(
                    teacher
                )
            ),
    }


# =============================================================
# Boundary-local Kendall tau-b
# =============================================================

def allocation_boundary_kendall(
    teacher_score,
    student_score,
    *,
    q=0.70,
    min_benefit=0.0,
    eligible_mask=None,
    total_bandwidth=0.10,
    min_n=10,
):
 

    teacher = _as_1d_float(
        teacher_score,
        "teacher_score",
    )

    student = _as_1d_float(
        student_score,
        "student_score",
    )

    if len(
        teacher
    ) != len(
        student
    ):
        raise ValueError(
            "Teacher and student scores must have equal length."
        )

    if eligible_mask is None:

        eligible = np.ones(
            len(
                teacher
            ),
            dtype=bool,
        )

    else:

        eligible = np.asarray(
            eligible_mask,
            dtype=bool,
        ).reshape(-1)

        if len(
            eligible
        ) != len(
            teacher
        ):
            raise ValueError(
                "eligible_mask and scores must have equal length."
            )

    teacher_e = teacher[
        eligible
    ]

    student_e = student[
        eligible
    ]

    if len(
        teacher_e
    ) == 0:

        return {
            "boundary_kendall_tau_b":
                np.nan,

            "boundary_region_n":
                0,

            "effective_teacher_inclusion_rate":
                0.0,

            "positivity_gate_binding":
                True,

            "teacher_boundary_score":
                np.nan,

            "boundary_lower_score":
                np.nan,

            "boundary_upper_score":
                np.nan,
        }

    positive_fraction = float(
        np.mean(
            teacher_e
            > float(
                min_benefit
            )
        )
    )

    effective_q = min(
        float(q),
        positive_fraction,
    )

    if effective_q <= 0:

        return {
            "boundary_kendall_tau_b":
                np.nan,

            "boundary_region_n":
                0,

            "effective_teacher_inclusion_rate":
                0.0,

            "positivity_gate_binding":
                True,

            "teacher_boundary_score":
                float(
                    min_benefit
                ),

            "boundary_lower_score":
                np.nan,

            "boundary_upper_score":
                np.nan,
        }

    cutoff_quantile = (
        1.0
        - effective_q
    )

    half_bandwidth = (
        float(
            total_bandwidth
        )
        / 2.0
    )

    lower_quantile = np.clip(
        cutoff_quantile
        - half_bandwidth,
        0.0,
        1.0,
    )

    upper_quantile = np.clip(
        cutoff_quantile
        + half_bandwidth,
        0.0,
        1.0,
    )

    (
        lower_score,
        boundary_score,
        upper_score,
    ) = np.quantile(
        teacher_e,
        [
            lower_quantile,
            cutoff_quantile,
            upper_quantile,
        ],
    )

    # Teacher-defined boundary neighborhood
    boundary_mask = (
        (
            teacher_e
            >= lower_score
        )
        &
        (
            teacher_e
            <= upper_score
        )
    )

    n_boundary = int(
        boundary_mask.sum()
    )

    if (
        n_boundary < int(
            min_n
        )
        or np.std(
            teacher_e[
                boundary_mask
            ]
        ) < 1e-12
        or np.std(
            student_e[
                boundary_mask
            ]
        ) < 1e-12
    ):

        tau_value = np.nan

    else:

        tau_value = (
            kendalltau(
                teacher_e[
                    boundary_mask
                ],
                student_e[
                    boundary_mask
                ],
                variant="b",
                nan_policy="raise",
            ).statistic
        )

    return {
        "boundary_kendall_tau_b":
            (
                float(
                    tau_value
                )
                if np.isfinite(
                    tau_value
                )
                else np.nan
            ),

        "boundary_region_n":
            n_boundary,

        "effective_teacher_inclusion_rate":
            float(
                effective_q
            ),

        "positivity_gate_binding":
            bool(
                positive_fraction
                < q
            ),

        "teacher_boundary_score":
            float(
                boundary_score
            ),

        "boundary_lower_score":
            float(
                lower_score
            ),

        "boundary_upper_score":
            float(
                upper_score
            ),

        "boundary_percentile_half_width":
            float(
                half_bandwidth
            ),
    }


# =============================================================
# Complete teacher-student fidelity table
# =============================================================

def evaluate_teacher_student_fidelity(
    teacher_score,
    student_scores: Mapping[str, Sequence[float]],
    *,
    eligible_mask=None,
    q=0.70,
    min_benefit=0.0,
    boundary_bandwidth=0.10,
    boundary_min_n=5,
    seed=7,
):
   

    teacher = _as_1d_float(
        teacher_score,
        "teacher_score",
    )

    n = len(
        teacher
    )

    # ---------------------------------------------------------
    # Policy-relevant population
    # ---------------------------------------------------------

    if eligible_mask is None:

        relevant_mask = np.ones(
            n,
            dtype=bool,
        )

    else:

        relevant_mask = np.asarray(
            eligible_mask,
            dtype=bool,
        ).reshape(-1)

        if len(
            relevant_mask
        ) != n:
            raise ValueError(
                "eligible_mask and teacher_score must have "
                "equal length."
            )

    # ---------------------------------------------------------
    # Shared tie priority
    #
    # Important:
    # Teacher and students use the SAME tie-breaking vector.
    # ---------------------------------------------------------

    tie_priority = (
        np.random.default_rng(
            int(seed)
        )
        .random(
            n
        )
    )

    teacher_policy, teacher_meta = (
        hard_positive_budget_policy(
            teacher,
            q=q,
            min_benefit=
                min_benefit,
            eligible_mask=
                relevant_mask,
            tie_priority=
                tie_priority,
            seed=seed,
        )
    )

    rows = []

    hard_policies = {
        "teacher":
            teacher_policy
    }

    # ---------------------------------------------------------
    # Each student
    # ---------------------------------------------------------

    for (
        student_name,
        student_values,
    ) in student_scores.items():

        student = _as_1d_float(
            student_values,
            student_name,
        )

        if len(
            student
        ) != n:
            raise ValueError(
                f"{student_name} and teacher_score must "
                "have equal length."
            )

        # -----------------------------------------------------
        # Global prediction fidelity
        # -----------------------------------------------------

        r2_value = float(
            r2_score(
                teacher,
                student,
            )
        )

        # -----------------------------------------------------
        # Policy-relevant rank fidelity
        # -----------------------------------------------------

        rank = rank_correlations(
            teacher,
            student,
            mask=
                relevant_mask,
        )

        # -----------------------------------------------------
        # Student hard top-q allocation
        # -----------------------------------------------------

        (
            student_policy,
            student_meta,
        ) = hard_positive_budget_policy(
            student,
            q=q,
            min_benefit=
                min_benefit,
            eligible_mask=
                relevant_mask,
            tie_priority=
                tie_priority,
            seed=seed,
        )

        hard_policies[
            student_name
        ] = student_policy

        overlap = binary_policy_overlap(
            teacher_policy,
            student_policy,
            mask=
                relevant_mask,
        )

        # -----------------------------------------------------
        # Boundary-local rank fidelity
        # -----------------------------------------------------

        boundary = (
            allocation_boundary_kendall(
                teacher,
                student,
                q=q,
                min_benefit=
                    min_benefit,
                eligible_mask=
                    relevant_mask,
                total_bandwidth=
                    boundary_bandwidth,
                min_n=
                    boundary_min_n,
            )
        )

        rows.append(
            {
                "Student":
                    student_name,

                "R2 to teacher":
                    r2_value,

                "Spearman":
                    rank[
                        "spearman"
                    ],

                "Kendall tau-b":
                    rank[
                        "kendall_tau_b"
                    ],

                "Top-q Jaccard":
                    overlap[
                        "jaccard"
                    ],

                "Top-q precision":
                    overlap[
                        "precision"
                    ],

                "Top-q recall":
                    overlap[
                        "recall"
                    ],

                "Teacher selected n":
                    overlap[
                        "reference_selected_n"
                    ],

                "Student selected n":
                    overlap[
                        "candidate_selected_n"
                    ],

                "Shared selected n":
                    overlap[
                        "shared_selected_n"
                    ],

                "Boundary Kendall tau-b":
                    boundary[
                        "boundary_kendall_tau_b"
                    ],

                "Boundary region n":
                    boundary[
                        "boundary_region_n"
                    ],

                "Effective teacher inclusion rate":
                    boundary[
                        "effective_teacher_inclusion_rate"
                    ],

                "Positivity gate binding":
                    boundary[
                        "positivity_gate_binding"
                    ],

                "Teacher boundary score":
                    boundary[
                        "teacher_boundary_score"
                    ],

                "Teacher unused budget n":
                    teacher_meta[
                        "unused_budget_n"
                    ],

                "Student unused budget n":
                    student_meta[
                        "unused_budget_n"
                    ],
            }
        )

    return (
        pd.DataFrame(
            rows
        ),
        hard_policies,
    )

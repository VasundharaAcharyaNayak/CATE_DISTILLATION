from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

from src.dr_evaluator import (
    _fit_logit_or_constant,
    dr_selective_shift_values_binary_y,
)


# =============================================================
# HistGradientBoosting outcome nuisance
# =============================================================

def _fit_histgb_or_constant(
    X,
    y,
    seed=7,
):
  

    X = np.asarray(
        X,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=int,
    ).reshape(-1)

    if len(y) == 0:
        raise ValueError(
            "Cannot fit outcome nuisance model on zero observations."
        )

    # Same protection as the primary logistic evaluator.
    if np.unique(y).size == 1:

        p = float(
            y.mean()
        )

        def predict_constant(X_new):
            return np.full(
                len(X_new),
                p,
                dtype=float,
            )

        return predict_constant

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=7,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=int(seed),
    )

    model.fit(
        X,
        y,
    )

    def predict_probability(X_new):
        return model.predict_proba(
            np.asarray(
                X_new,
                dtype=float,
            )
        )[:, 1]

    return predict_probability


# =============================================================
# Selective-shift DR evaluator with HGB mu0
# =============================================================

def dr_selective_shift_values_histgb_outcome(
    Z,
    T,
    Y,
    policies,
    *,
    n_splits=5,
    trim=0.05,
    seed=7,
    clip_eps=1e-6,
    return_contributions=False,
):
 

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

    n = len(
        Y
    )

    if not (
        len(Z)
        == len(T)
        == n
    ):
        raise ValueError(
            "Z, T, and Y must have equal numbers of rows."
        )

    # ---------------------------------------------------------
    # Frozen policies
    # ---------------------------------------------------------

    normalized_policies = {}

    for name, shift_prob in policies.items():

        s = np.asarray(
            shift_prob,
            dtype=float,
        ).reshape(-1)

        if len(s) != n:
            raise ValueError(
                f"{name} has incorrect policy length."
            )

        if not np.isfinite(
            s
        ).all():
            raise ValueError(
                f"{name} contains non-finite values."
            )

        if np.any(
            (s < 0.0)
            | (s > 1.0)
        ):
            raise ValueError(
                f"{name} contains shift probabilities "
                "outside [0,1]."
            )

        normalized_policies[
            name
        ] = s

    # ---------------------------------------------------------
    # Same treatment-stratified folds as primary evaluator
    # ---------------------------------------------------------

    class_counts = np.bincount(
        T,
        minlength=2,
    )

    effective_folds = min(
        int(n_splits),
        int(
            class_counts.min()
        ),
    )

    if effective_folds < 2:
        raise RuntimeError(
            "Insufficient treatment overlap for cross-fitting."
        )

    splitter = StratifiedKFold(
        n_splits=effective_folds,
        shuffle=True,
        random_state=int(seed),
    )

    contributions = {
        name: np.full(
            n,
            np.nan,
            dtype=float,
        )
        for name in normalized_policies
    }

    # =========================================================
    # Cross-fitting
    # =========================================================

    for fold_id, (
        fit_idx,
        score_idx,
    ) in enumerate(
        splitter.split(
            Z,
            T,
        ),
        start=1,
    ):

        # =====================================================
        # 1. SAME logistic propensity e(Z)
        # =====================================================

        e_predict = _fit_logit_or_constant(
            Z[
                fit_idx
            ],
            T[
                fit_idx
            ],
        )

        e_score = np.clip(
            e_predict(
                Z[
                    score_idx
                ]
            ),
            clip_eps,
            1.0 - clip_eps,
        )

        # =====================================================
        # 2. SAME overlap trimming
        # =====================================================

        keep_local = (
            (e_score >= trim)
            &
            (
                e_score
                <= 1.0 - trim
            )
        )

        if not np.any(
            keep_local
        ):
            continue

        kept_idx = score_idx[
            keep_local
        ]

        # =====================================================
        # 3. ONLY CHANGE:
        #    HGB outcome model under T=0
        # =====================================================

        control_fit_idx = fit_idx[
            T[
                fit_idx
            ] == 0
        ]

        if len(
            control_fit_idx
        ) == 0:
            raise RuntimeError(
                "No controls in outcome-model training fold."
            )

        mu0_predict = (
            _fit_histgb_or_constant(
                Z[
                    control_fit_idx
                ],
                Y[
                    control_fit_idx
                ].astype(int),
                seed=
                    int(seed)
                    + fold_id,
            )
        )

        mu0 = np.clip(
            mu0_predict(
                Z[
                    kept_idx
                ]
            ),
            clip_eps,
            1.0 - clip_eps,
        )

        ti = T[
            kept_idx
        ].astype(
            float
        )

        yi = Y[
            kept_idx
        ]

        ei = e_score[
            keep_local
        ]

        # =====================================================
        # 4. SAME DR score for every frozen policy
        # =====================================================

        for (
            name,
            shift_probability,
        ) in normalized_policies.items():

            si = shift_probability[
                kept_idx
            ]

            contributions[
                name
            ][
                kept_idx
            ] = (

                (1.0 - ti * si)
                * yi

                + ti
                * si
                * mu0

                + (1.0 - ti)
                * si
                * (
                    ei
                    / (1.0 - ei)
                )
                * (
                    yi
                    - mu0
                )
            )

    # =========================================================
    # Common retained overlap population
    # =========================================================

    first_name = next(
        iter(
            contributions
        )
    )

    used = ~np.isnan(
        contributions[
            first_name
        ]
    )

    if not np.any(
        used
    ):
        raise RuntimeError(
            "No observations passed overlap trimming."
        )

    for name, score in contributions.items():

        if np.isnan(
            score[
                used
            ]
        ).any():
            raise RuntimeError(
                f"{name} contains missing DR contributions "
                "inside the retained population."
            )

    values = {
        name: float(
            np.mean(
                score[
                    used
                ]
            )
        )
        for name, score
        in contributions.items()
    }

    retained_fraction = float(
        used.mean()
    )

    if return_contributions:

        return (
            values,
            retained_fraction,
            contributions,
            used,
        )

    return (
        values,
        retained_fraction,
    )


# =============================================================
# Primary-vs-HGB sensitivity comparison
# =============================================================

def compare_outcome_nuisance_models(
    Z,
    T,
    Y,
    policies: Mapping[
        str,
        Sequence[float],
    ],
    *,
    n_splits=5,
    trim=0.05,
    seed=7,
):
    

    # =========================================================
    # Primary logistic outcome nuisance
    # =========================================================

    (
        logistic_risks,
        logistic_retained,
        _,
        logistic_mask,
    ) = dr_selective_shift_values_binary_y(
        Z=Z,
        T=T,
        Y=Y,
        policies=policies,
        n_splits=n_splits,
        trim=trim,
        seed=seed,
        return_contributions=True,
    )

    # =========================================================
    # HGB outcome nuisance
    # =========================================================

    (
        histgb_risks,
        histgb_retained,
        _,
        histgb_mask,
    ) = (
        dr_selective_shift_values_histgb_outcome(
            Z=Z,
            T=T,
            Y=Y,
            policies=policies,
            n_splits=n_splits,
            trim=trim,
            seed=seed,
            return_contributions=True,
        )
    )

    # ---------------------------------------------------------
    # Since propensity and trimming are unchanged,
    # the retained evaluation population must be identical.
    # ---------------------------------------------------------

    same_overlap_mask = bool(
        np.array_equal(
            logistic_mask,
            histgb_mask,
        )
    )

    if not same_overlap_mask:
        raise RuntimeError(
            "Outcome-nuisance ablation unexpectedly changed "
            "the overlap-trimmed evaluation population."
        )

    # =========================================================
    # Comparison table
    # =========================================================

    rows = []

    for name in policies:

        utility_logistic = float(
            1.0
            - logistic_risks[
                name
            ]
        )

        utility_histgb = float(
            1.0
            - histgb_risks[
                name
            ]
        )

        difference = (
            utility_histgb
            - utility_logistic
        )

        rows.append(
            {
                "Policy":
                    name,

                "Logistic outcome utility":
                    utility_logistic,

                "HistGradientBoosting outcome utility":
                    utility_histgb,

                "Difference (HGB - Logistic)":
                    float(
                        difference
                    ),

                "Absolute difference":
                    float(
                        abs(
                            difference
                        )
                    ),
            }
        )

    results = pd.DataFrame(
        rows
    )

    # =========================================================
    # Ranking stability
    # =========================================================

    results[
        "Logistic rank"
    ] = (
        results[
            "Logistic outcome utility"
        ]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )

    results[
        "HGB rank"
    ] = (
        results[
            "HistGradientBoosting outcome utility"
        ]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )

    metadata = {
        "Logistic retained fraction":
            float(
                logistic_retained
            ),

        "HGB retained fraction":
            float(
                histgb_retained
            ),

        "Same overlap mask":
            same_overlap_mask,

        "Maximum absolute utility difference":
            float(
                results[
                    "Absolute difference"
                ].max()
            ),

        "Mean absolute utility difference":
            float(
                results[
                    "Absolute difference"
                ].mean()
            ),
    }

    return (
        results,
        metadata,
    )

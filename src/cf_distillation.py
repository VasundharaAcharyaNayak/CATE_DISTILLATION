from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from econml.dml import CausalForestDML

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


# =============================================================
# Nuisance-model helpers
# =============================================================

def _binary_logit():
    """
    Logistic nuisance model used in the observational
    PIMA/NHANES causal-forest analyses.
    """

    return LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=10_000,
    )


def _continuous_outcome_model(seed):
    """
    Outcome nuisance model used for the continuous-outcome
    IHDP causal-forest analysis.
    """

    return RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=5,
        max_features=1.0,
        random_state=int(seed),
        n_jobs=-1,
    )


# =============================================================
# Causal Forest teacher
# =============================================================

def make_causal_forest_teacher(
    teacher_params: Mapping[str, Any],
    *,
    outcome_type="binary",
    seed=7,
    internal_cv=3,
):
   

    params = dict(
        teacher_params
    )

    n_estimators = int(
        params["n_estimators"]
    )

    subforest_size = int(
        params.get(
            "subforest_size",
            4,
        )
    )

    if (
        n_estimators
        % subforest_size
        != 0
    ):
        raise ValueError(
            "n_estimators must be divisible by "
            "subforest_size."
        )

    # ---------------------------------------------------------
    # Observational binary-outcome experiments
    # ---------------------------------------------------------

    if outcome_type == "binary":

        model_y = _binary_logit()

        model_t = _binary_logit()

        discrete_outcome = True

    # ---------------------------------------------------------
    # IHDP continuous outcome
    # ---------------------------------------------------------

    elif outcome_type == "continuous":

        model_y = (
            _continuous_outcome_model(
                seed
            )
        )

        model_t = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=5000,
                random_state=int(seed),
            ),
        )

        discrete_outcome = False

    else:

        raise ValueError(
            "outcome_type must be either "
            "'binary' or 'continuous'."
        )

    # ---------------------------------------------------------
    # Teacher
    # ---------------------------------------------------------

    teacher = CausalForestDML(
        model_y=model_y,
        model_t=model_t,

        discrete_outcome=
            discrete_outcome,

        discrete_treatment=True,

        cv=int(
            internal_cv
        ),

        n_estimators=
            n_estimators,

        max_depth=
            params.get(
                "max_depth"
            ),

        min_samples_split=
            int(
                params.get(
                    "min_samples_split",
                    10,
                )
            ),

        min_samples_leaf=
            int(
                params.get(
                    "min_samples_leaf",
                    5,
                )
            ),

        max_samples=
            float(
                params.get(
                    "max_samples",
                    0.45,
                )
            ),

        max_features=
            params.get(
                "max_features"
            ),

        honest=
            bool(
                params.get(
                    "honest",
                    True,
                )
            ),

        inference=False,

        subforest_size=
            subforest_size,

        n_jobs=
            int(
                params.get(
                    "n_jobs",
                    -1,
                )
            ),

        random_state=
            int(seed),
    )

    return teacher


# =============================================================
# OOF teacher CATE targets
# =============================================================

def generate_oof_cf_targets(
    X,
    T,
    Y,
    teacher_params,
    *,
    W=None,
    outcome_type="binary",
    n_splits=5,
    seed=7,
    internal_cv=3,
    fold_seed_offset=0,
):
   

    X = np.asarray(
        X,
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

    if W is not None:

        W = np.asarray(
            W,
            dtype=float,
        )

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    if not (
        len(X)
        == len(T)
        == len(Y)
    ):
        raise ValueError(
            "X, T, and Y must have equal numbers of rows."
        )

    if (
        W is not None
        and len(W) != len(Y)
    ):
        raise ValueError(
            "W must have the same number of rows as X."
        )

    if not np.all(
        np.isin(
            T,
            [0, 1],
        )
    ):
        raise ValueError(
            "T must contain only 0 and 1."
        )

    class_counts = np.bincount(
        T,
        minlength=2,
    )

    if (
        class_counts.min()
        < int(n_splits)
    ):
        raise ValueError(
            f"Treatment class counts "
            f"{class_counts.tolist()} "
            f"are too small for "
            f"{n_splits} OOF folds."
        )

    # ---------------------------------------------------------
    # Treatment-stratified OOF folds
    # ---------------------------------------------------------

    splitter = StratifiedKFold(
        n_splits=int(n_splits),
        shuffle=True,
        random_state=int(seed),
    )

    tau_oof = np.full(
        len(Y),
        np.nan,
        dtype=float,
    )

    fold_rows = []

    # ---------------------------------------------------------
    # OOF teacher fitting
    # ---------------------------------------------------------

    for (
        fold_id,
        (fit_idx, hold_idx),
    ) in enumerate(
        splitter.split(
            X,
            T,
        ),
        start=1,
    ):

        fold_seed = (
            int(seed)
            + int(fold_seed_offset)
            + fold_id
        )

        teacher_fold = (
            make_causal_forest_teacher(
                teacher_params,
                outcome_type=
                    outcome_type,
                seed=
                    fold_seed,
                internal_cv=
                    internal_cv,
            )
        )

        fit_kwargs = {
            "Y":
                Y[fit_idx],

            "T":
                T[fit_idx],

            "X":
                X[fit_idx],
        }

        if W is not None:

            fit_kwargs[
                "W"
            ] = W[
                fit_idx
            ]

        teacher_fold.fit(
            **fit_kwargs
        )

        tau_oof[
            hold_idx
        ] = np.asarray(
            teacher_fold.effect(
                X[
                    hold_idx
                ],
                T0=0,
                T1=1,
            ),
            dtype=float,
        ).reshape(-1)

        fold_rows.append(
            {
                "fold":
                    fold_id,

                "n_fit":
                    len(
                        fit_idx
                    ),

                "n_holdout":
                    len(
                        hold_idx
                    ),

                "treated_fit":
                    int(
                        T[
                            fit_idx
                        ].sum()
                    ),

                "treated_holdout":
                    int(
                        T[
                            hold_idx
                        ].sum()
                    ),
            }
        )

    if np.isnan(
        tau_oof
    ).any():

        raise RuntimeError(
            "Some discovery rows did not receive "
            "an OOF CF target."
        )

    return (
        tau_oof,
        pd.DataFrame(
            fold_rows
        ),
    )


# =============================================================
# DT / RF student approximators
# =============================================================

def fit_student_models(
    X_discovery,
    teacher_oof_cate,
    *,
    dt_params,
    rf_params,
    seed=7,
):
   

    X_discovery = np.asarray(
        X_discovery,
        dtype=float,
    )

    teacher_oof_cate = np.asarray(
        teacher_oof_cate,
        dtype=float,
    ).reshape(-1)

    if (
        len(X_discovery)
        != len(
            teacher_oof_cate
        )
    ):
        raise ValueError(
            "X_discovery and teacher_oof_cate "
            "must have equal numbers of rows."
        )

    dt_params = dict(
        dt_params
    )

    rf_params = dict(
        rf_params
    )

    # ---------------------------------------------------------
    # Decision-tree student
    # ---------------------------------------------------------

    dt_student = DecisionTreeRegressor(
        criterion="squared_error",

        max_depth=
            dt_params.get(
                "max_depth"
            ),

        min_samples_leaf=
            int(
                dt_params.get(
                    "min_samples_leaf",
                    1,
                )
            ),

        max_leaf_nodes=
            dt_params.get(
                "max_leaf_nodes"
            ),

        random_state=
            int(seed),
    )

    # ---------------------------------------------------------
    # Random-forest student
    # ---------------------------------------------------------

    rf_student = RandomForestRegressor(
        n_estimators=
            int(
                rf_params[
                    "n_estimators"
                ]
            ),

        criterion="squared_error",

        max_depth=
            rf_params.get(
                "max_depth"
            ),

        min_samples_leaf=
            int(
                rf_params.get(
                    "min_samples_leaf",
                    1,
                )
            ),

        max_leaf_nodes=
            rf_params.get(
                "max_leaf_nodes"
            ),

        bootstrap=True,

        n_jobs=-1,

        random_state=
            int(seed),
    )

    # ---------------------------------------------------------
    # Distillation
    # ---------------------------------------------------------

    dt_student.fit(
        X_discovery,
        teacher_oof_cate,
    )

    rf_student.fit(
        X_discovery,
        teacher_oof_cate,
    )

    return (
        dt_student,
        rf_student,
    )


# =============================================================
# Complete CF distillation
# =============================================================

def fit_cf_distillation(
    X_discovery,
    T_discovery,
    Y_discovery,
    X_evaluation,
    *,
    teacher_params,
    dt_params,
    rf_params,
    W_discovery=None,
    outcome_type="binary",
    teacher_oof_folds=5,
    internal_cv=3,
    seed=7,
    fold_seed_offset=0,
):
    

    # ---------------------------------------------------------
    # Inputs
    # ---------------------------------------------------------

    X_discovery = np.asarray(
        X_discovery,
        dtype=float,
    )

    T_discovery = np.asarray(
        T_discovery,
        dtype=int,
    ).reshape(-1)

    Y_discovery = np.asarray(
        Y_discovery,
        dtype=float,
    ).reshape(-1)

    X_evaluation = np.asarray(
        X_evaluation,
        dtype=float,
    )

    if W_discovery is not None:

        W_discovery = np.asarray(
            W_discovery,
            dtype=float,
        )

    # =========================================================
    # 1. OOF teacher targets on discovery
    # =========================================================

    (
        teacher_oof_cate,
        oof_fold_table,
    ) = generate_oof_cf_targets(
        X=X_discovery,
        T=T_discovery,
        Y=Y_discovery,
        W=W_discovery,
        teacher_params=
            teacher_params,
        outcome_type=
            outcome_type,
        n_splits=
            teacher_oof_folds,
        seed=
            seed,
        internal_cv=
            internal_cv,
        fold_seed_offset=
            fold_seed_offset,
    )

    # =========================================================
    # 2. Final teacher on full discovery cohort
    # =========================================================

    teacher = (
        make_causal_forest_teacher(
            teacher_params,
            outcome_type=
                outcome_type,
            seed=
                seed,
            internal_cv=
                internal_cv,
        )
    )

    fit_kwargs = {
        "Y":
            Y_discovery,

        "T":
            T_discovery,

        "X":
            X_discovery,
    }

    if W_discovery is not None:

        fit_kwargs[
            "W"
        ] = W_discovery

    teacher.fit(
        **fit_kwargs
    )

    teacher_cate_evaluation = np.asarray(
        teacher.effect(
            X_evaluation,
            T0=0,
            T1=1,
        ),
        dtype=float,
    ).reshape(-1)

    # =========================================================
    # 3. Fit DT/RF students to OOF teacher targets
    # =========================================================

    (
        dt_student,
        rf_student,
    ) = fit_student_models(
        X_discovery,
        teacher_oof_cate,
        dt_params=
            dt_params,
        rf_params=
            rf_params,
        seed=
            seed,
    )

    # =========================================================
    # 4. Frozen student predictions on evaluation
    # =========================================================

    dt_cate_evaluation = np.asarray(
        dt_student.predict(
            X_evaluation
        ),
        dtype=float,
    ).reshape(-1)

    rf_cate_evaluation = np.asarray(
        rf_student.predict(
            X_evaluation
        ),
        dtype=float,
    ).reshape(-1)

    # =========================================================
    # Return models + CATE surfaces
    # =========================================================

    return {
        "teacher":
            teacher,

        "dt_student":
            dt_student,

        "rf_student":
            rf_student,

        "teacher_oof_cate":
            teacher_oof_cate,

        "oof_fold_table":
            oof_fold_table,

        "teacher_cate_evaluation":
            teacher_cate_evaluation,

        "dt_cate_evaluation":
            dt_cate_evaluation,

        "rf_cate_evaluation":
            rf_cate_evaluation,
    }

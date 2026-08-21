from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

try:
    from stochtree import BCFModel, OutcomeModel
except ImportError as exc:
    raise ImportError(
        "bcf_distillation requires stochtree. "
        "Install it with `pip install stochtree`."
    ) from exc


# =============================================================
# Default BCF configuration
# =============================================================

def default_bcf_general_params(outcome_type="binary"):
  

    if outcome_type == "binary":
        outcome_model = OutcomeModel(
            outcome="binary",
            link="probit",
        )

    elif outcome_type == "continuous":
        outcome_model = OutcomeModel(
            outcome="continuous"
        )

    else:
        raise ValueError(
            "outcome_type must be 'binary' or 'continuous'."
        )

    return {
        "outcome_model": outcome_model,
        "propensity_covariate": "prognostic",
        "adaptive_coding": True,
        "keep_every": 2,
        "standardize": True,
        "num_threads": 1,
    }


def default_bcf_prognostic_forest_params(
    n_features,
):
    return {
        "num_trees": 200,
        "alpha": 0.95,
        "beta": 2.0,
        "min_samples_leaf": 5,
        "max_depth": 8,
        "keep_vars": list(
            range(
                int(n_features)
            )
        ),
    }


def default_bcf_treatment_forest_params(
    n_features,
    outcome_type="binary",
):
    params = {
        "num_trees": 50,
        "alpha": 0.25,
        "beta": 3.0,
        "min_samples_leaf": 5,
        "max_depth": 5,
        "keep_vars": list(
            range(
                int(n_features)
            )
        ),
    }

    # Used in the binary-probit observational analyses.
    if outcome_type == "binary":
        params[
            "delta_max"
        ] = 0.90

    elif outcome_type != "continuous":
        raise ValueError(
            "outcome_type must be 'binary' or 'continuous'."
        )

    return params


# =============================================================
# External propensity model
# =============================================================

def _fit_external_propensity(
    X_train,
    T_train,
    seed,
    outcome_type="binary",
):
    """
    Fit the external propensity model used by BCF.

        e(X) = P(T=1 | X)

    The model is fitted only on the corresponding training fold.
    """

    # Matches the observational notebooks versus IHDP.
    max_iter = (
        10_000
        if outcome_type == "binary"
        else 5_000
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=max_iter,
            random_state=int(seed),
        ),
    )

    model.fit(
        np.asarray(
            X_train,
            dtype=float,
        ),
        np.asarray(
            T_train,
            dtype=int,
        ).reshape(-1),
    )

    return model


# =============================================================
# Normalize StochTree posterior draws
# =============================================================

def _normalize_draw_matrix(
    draws,
    n_observations,
):
    """
    Return posterior draws in:

        n_observations x n_posterior_draws

    orientation.
    """

    arr = np.asarray(
        draws,
        dtype=float,
    )

    arr = np.squeeze(
        arr
    )

    if arr.ndim == 1:
        arr = arr[:, None]

    if arr.ndim != 2:
        raise ValueError(
            "Unexpected BCF posterior contrast shape: "
            f"{arr.shape}"
        )

    if (
        arr.shape[0]
        != n_observations
        and arr.shape[1]
        == n_observations
    ):
        arr = arr.T

    if (
        arr.shape[0]
        != n_observations
    ):
        raise ValueError(
            "Could not align BCF posterior contrasts "
            "with observations: "
            f"shape={arr.shape}, "
            f"n={n_observations}."
        )

    if not np.isfinite(
        arr
    ).all():
        raise ValueError(
            "BCF posterior contrasts contain "
            "non-finite values."
        )

    return arr


# =============================================================
# Fit one BCF model and obtain posterior CATEs
# =============================================================

def fit_bcf_and_predict(
    X_train,
    T_train,
    Y_train,
    X_pred,
    T_pred,
    *,
    outcome_type="binary",
    seed=7,
    num_gfr=10,
    num_mcmc=1000,
    general_params=None,
    prognostic_forest_params=None,
    treatment_effect_forest_params=None,
    propensity_clip_eps=1e-6,
):
   

    X_train = np.asarray(
        X_train,
        dtype=float,
    )

    T_train = np.asarray(
        T_train,
        dtype=int,
    ).reshape(-1)

    Y_train = np.asarray(
        Y_train,
        dtype=float,
    ).reshape(-1)

    X_pred = np.asarray(
        X_pred,
        dtype=float,
    )

    T_pred = np.asarray(
        T_pred,
        dtype=int,
    ).reshape(-1)

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    if not (
        len(X_train)
        == len(T_train)
        == len(Y_train)
    ):
        raise ValueError(
            "BCF training arrays must have matching lengths."
        )

    if (
        len(X_pred)
        != len(T_pred)
    ):
        raise ValueError(
            "BCF prediction X and T must have matching lengths."
        )

    if (
        np.unique(
            T_train
        ).size
        < 2
    ):
        raise ValueError(
            "BCF training data must contain both treatment states."
        )

    if outcome_type == "binary":

        if not set(
            np.unique(
                Y_train
            ).tolist()
        ).issubset(
            {0.0, 1.0}
        ):
            raise ValueError(
                "Binary-probit BCF requires Y in {0,1}."
            )

    elif outcome_type != "continuous":

        raise ValueError(
            "outcome_type must be 'binary' or 'continuous'."
        )

    # =========================================================
    # External propensity
    # =========================================================

    propensity_model = (
        _fit_external_propensity(
            X_train,
            T_train,
            seed=seed,
            outcome_type=outcome_type,
        )
    )

    propensity_train = np.clip(
        propensity_model.predict_proba(
            X_train
        )[:, 1],
        propensity_clip_eps,
        1.0 - propensity_clip_eps,
    )

    propensity_pred = np.clip(
        propensity_model.predict_proba(
            X_pred
        )[:, 1],
        propensity_clip_eps,
        1.0 - propensity_clip_eps,
    )

    # =========================================================
    # BCF parameters
    # =========================================================

    resolved_general = (
        default_bcf_general_params(
            outcome_type=
                outcome_type
        )
    )

    if general_params is not None:
        resolved_general.update(
            dict(
                general_params
            )
        )

    resolved_general[
        "random_seed"
    ] = int(
        seed
    )

    # Prevent an accidental mismatch in outcome model.
    if outcome_type == "binary":

        resolved_general[
            "outcome_model"
        ] = OutcomeModel(
            outcome="binary",
            link="probit",
        )

    else:

        resolved_general[
            "outcome_model"
        ] = OutcomeModel(
            outcome="continuous"
        )

    n_features = (
        X_train.shape[1]
    )

    resolved_prog = (
        default_bcf_prognostic_forest_params(
            n_features
        )
    )

    if (
        prognostic_forest_params
        is not None
    ):
        resolved_prog.update(
            dict(
                prognostic_forest_params
            )
        )

    resolved_prog.setdefault(
        "keep_vars",
        list(
            range(
                n_features
            )
        ),
    )

    resolved_tau = (
        default_bcf_treatment_forest_params(
            n_features,
            outcome_type=
                outcome_type,
        )
    )

    if (
        treatment_effect_forest_params
        is not None
    ):
        resolved_tau.update(
            dict(
                treatment_effect_forest_params
            )
        )

    resolved_tau.setdefault(
        "keep_vars",
        list(
            range(
                n_features
            )
        ),
    )

    # =========================================================
    # Fit BCF
    # =========================================================

    model = BCFModel()

    model.sample(
        X_train=X_train,
        Z_train=T_train.astype(
            float
        ),
        y_train=Y_train,
        propensity_train=
            propensity_train,

        X_test=X_pred,
        Z_test=T_pred.astype(
            float
        ),
        propensity_test=
            propensity_pred,

        num_gfr=int(
            num_gfr
        ),

        num_mcmc=int(
            num_mcmc
        ),

        general_params=
            resolved_general,

        prognostic_forest_params=
            resolved_prog,

        treatment_effect_forest_params=
            resolved_tau,
    )

    # =========================================================
    # Posterior treatment contrast
    # =========================================================

    n_pred = len(
        X_pred
    )

    z0 = np.zeros(
        n_pred,
        dtype=float,
    )

    z1 = np.ones(
        n_pred,
        dtype=float,
    )

    contrast_kwargs = {
        "X_0":
            X_pred,

        "X_1":
            X_pred,

        "Z_0":
            z0,

        "Z_1":
            z1,

        "propensity_0":
            propensity_pred,

        "propensity_1":
            propensity_pred,

        "type":
            "posterior",
    }

    # ---------------------------------------------------------
    # Binary adverse outcome:
    # probability-scale contrast
    # ---------------------------------------------------------

    if outcome_type == "binary":

        contrast_raw = (
            model.compute_contrast(
                **contrast_kwargs,
                scale="probability",
            )
        )

    # ---------------------------------------------------------
    # Continuous IHDP outcome:
    # linear-scale contrast
    # ---------------------------------------------------------

    else:

        try:

            contrast_raw = (
                model.compute_contrast(
                    **contrast_kwargs,
                    scale="linear",
                )
            )

        except TypeError:

            # Supports older StochTree APIs.
            contrast_raw = (
                model.compute_contrast(
                    **contrast_kwargs
                )
            )

    draws = _normalize_draw_matrix(
        contrast_raw,
        n_pred,
    )

    # =========================================================
    # Posterior summaries
    # =========================================================

    return {
        "model":
            model,

        "propensity_model":
            propensity_model,

        "propensity_train":
            propensity_train,

        "propensity_pred":
            propensity_pred,

        "cate_draws":
            draws,

        "cate_mean":
            np.mean(
                draws,
                axis=1,
            ),

        "cate_lower":
            np.quantile(
                draws,
                0.025,
                axis=1,
            ),

        "cate_upper":
            np.quantile(
                draws,
                0.975,
                axis=1,
            ),

        "prob_cate_positive":
            np.mean(
                draws > 0.0,
                axis=1,
            ),

        "general_params":
            resolved_general,

        "prognostic_forest_params":
            resolved_prog,

        "treatment_effect_forest_params":
            resolved_tau,
    }


# =============================================================
# OOF BCF teacher targets
# =============================================================

def generate_oof_bcf_targets(
    X,
    T,
    Y,
    *,
    outcome_type="binary",
    n_splits=5,
    seed=7,
    num_gfr=10,
    num_mcmc=1000,
    general_params=None,
    prognostic_forest_params=None,
    treatment_effect_forest_params=None,
    oof_seed_offset=None,
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

    if not (
        len(X)
        == len(T)
        == len(Y)
    ):
        raise ValueError(
            "X, T, and Y must have equal numbers of rows."
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
            f"Treatment counts "
            f"{class_counts.tolist()} "
            f"are too small for "
            f"{n_splits} OOF folds."
        )

    # ---------------------------------------------------------
    # Preserve seeds used in the notebooks.
    #
    # Binary notebooks:
    #     seed+1, ..., seed+5
    #
    # IHDP notebook:
    #     seed+0, ..., seed+4
    # ---------------------------------------------------------

    if oof_seed_offset is None:

        oof_seed_offset = (
            1
            if outcome_type == "binary"
            else 0
        )

    targets = np.full(
        len(Y),
        np.nan,
        dtype=float,
    )

    fold_rows = []

    splitter = StratifiedKFold(
        n_splits=int(
            n_splits
        ),
        shuffle=True,
        random_state=int(
            seed
        ),
    )

    for fold_index, (
        fit_idx,
        hold_idx,
    ) in enumerate(
        splitter.split(
            X,
            T,
        )
    ):

        fold_seed = (
            int(seed)
            + int(
                oof_seed_offset
            )
            + fold_index
        )

        fit = fit_bcf_and_predict(
            X_train=
                X[fit_idx],

            T_train=
                T[fit_idx],

            Y_train=
                Y[fit_idx],

            X_pred=
                X[hold_idx],

            T_pred=
                T[hold_idx],

            outcome_type=
                outcome_type,

            seed=
                fold_seed,

            num_gfr=
                num_gfr,

            num_mcmc=
                num_mcmc,

            general_params=
                general_params,

            prognostic_forest_params=
                prognostic_forest_params,

            treatment_effect_forest_params=
                treatment_effect_forest_params,
        )

        targets[
            hold_idx
        ] = fit[
            "cate_mean"
        ]

        fold_rows.append(
            {
                "fold":
                    fold_index + 1,

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

                "bcf_seed":
                    fold_seed,
            }
        )

    if np.isnan(
        targets
    ).any():
        raise RuntimeError(
            "Some discovery observations did not receive "
            "an OOF BCF target."
        )

    return (
        targets,
        pd.DataFrame(
            fold_rows
        ),
    )


# =============================================================
# DT / RF student approximation
# =============================================================

def fit_bcf_students(
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
    # Decision Tree student
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
    # Random Forest student
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
# Complete BCF distillation pipeline
# =============================================================

def fit_bcf_distillation(
    X_discovery,
    T_discovery,
    Y_discovery,
    X_evaluation,
    T_evaluation,
    *,
    dt_params,
    rf_params,
    outcome_type="binary",
    seed=7,
    oof_folds=5,
    num_gfr=10,
    num_mcmc=1000,
    general_params=None,
    prognostic_forest_params=None,
    treatment_effect_forest_params=None,
    oof_seed_offset=None,
):
  

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

    T_evaluation = np.asarray(
        T_evaluation,
        dtype=int,
    ).reshape(-1)

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    if not (
        len(X_discovery)
        == len(T_discovery)
        == len(Y_discovery)
    ):
        raise ValueError(
            "Discovery X, T, and Y must be aligned."
        )

    if (
        len(X_evaluation)
        != len(T_evaluation)
    ):
        raise ValueError(
            "Evaluation X and T must be aligned."
        )

    n_features = (
        X_discovery.shape[1]
    )

    # =========================================================
    # Resolve fixed BCF parameters
    # =========================================================

    resolved_general = (
        default_bcf_general_params(
            outcome_type=
                outcome_type
        )
    )

    if general_params is not None:
        resolved_general.update(
            dict(
                general_params
            )
        )

    resolved_prog = (
        default_bcf_prognostic_forest_params(
            n_features
        )
    )

    if (
        prognostic_forest_params
        is not None
    ):
        resolved_prog.update(
            dict(
                prognostic_forest_params
            )
        )

    resolved_tau = (
        default_bcf_treatment_forest_params(
            n_features,
            outcome_type=
                outcome_type,
        )
    )

    if (
        treatment_effect_forest_params
        is not None
    ):
        resolved_tau.update(
            dict(
                treatment_effect_forest_params
            )
        )

    # =========================================================
    # 1. OOF teacher CATE targets
    # =========================================================

    (
        teacher_oof_cate,
        oof_fold_table,
    ) = generate_oof_bcf_targets(
        X=X_discovery,
        T=T_discovery,
        Y=Y_discovery,

        outcome_type=
            outcome_type,

        n_splits=
            oof_folds,

        seed=
            seed,

        num_gfr=
            num_gfr,

        num_mcmc=
            num_mcmc,

        general_params=
            resolved_general,

        prognostic_forest_params=
            resolved_prog,

        treatment_effect_forest_params=
            resolved_tau,

        oof_seed_offset=
            oof_seed_offset,
    )

    # =========================================================
    # 2. Students trained on OOF BCF targets
    # =========================================================

    (
        dt_student,
        rf_student,
    ) = fit_bcf_students(
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
    # 3. Final BCF teacher on full discovery cohort
    # =========================================================

    final_teacher = fit_bcf_and_predict(
        X_train=
            X_discovery,

        T_train=
            T_discovery,

        Y_train=
            Y_discovery,

        X_pred=
            X_evaluation,

        T_pred=
            T_evaluation,

        outcome_type=
            outcome_type,

        seed=
            seed,

        num_gfr=
            num_gfr,

        num_mcmc=
            num_mcmc,

        general_params=
            resolved_general,

        prognostic_forest_params=
            resolved_prog,

        treatment_effect_forest_params=
            resolved_tau,
    )

    # =========================================================
    # 4. Frozen evaluation predictions
    # =========================================================

    teacher_cate_evaluation = np.asarray(
        final_teacher[
            "cate_mean"
        ],
        dtype=float,
    ).reshape(-1)

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
    # Outputs
    # =========================================================

    return {
        "teacher":
            final_teacher[
                "model"
            ],

        # Keeps posterior draws and other BCF information
        # available without mixing them into policy evaluation.
        "teacher_fit":
            final_teacher,

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

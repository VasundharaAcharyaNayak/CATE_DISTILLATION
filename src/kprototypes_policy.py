

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Iterable, Literal

import numpy as np
import pandas as pd

from kmodes.kprototypes import KPrototypes

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =============================================================
# IHDP feature structure
# =============================================================

IHDP_FEATURE_NAMES = [
    "birth_weight",                   # x1
    "head_circumference",             # x2
    "weeks_preterm",                  # x3
    "birth_order",                    # x4
    "neonatal_health_index",          # x5
    "mother_age",                     # x6
    "female",                         # x7
    "twin",                           # x8
    "mother_married_at_birth",        # x9
    "mother_left_high_school",        # x10
    "mother_completed_high_school",   # x11
    "mother_some_college",            # x12
    "first_born",                     # x13
    "mother_smoked_pregnancy",        # x14
    "mother_alcohol_pregnancy",       # x15
    "mother_drugs_pregnancy",         # x16
    "mother_worked_pregnancy",        # x17
    "mother_received_prenatal_care",  # x18
    "site_1",                         # x19
    "site_2",                         # x20
    "site_3",                         # x21
    "site_4",                         # x22
    "site_5",                         # x23
    "site_6",                         # x24
    "site_7",                         # x25
]


# Six continuous covariates
CONTINUOUS_IDX = list(range(0, 6))

# Nine standalone two-level categorical covariates
STANDALONE_BINARY_IDX = [
    6, 7, 8,
    12, 13, 14, 15, 16, 17,
]

# Collapse these dummy blocks before K-Prototypes
EDUCATION_DUMMY_IDX = [9, 10, 11]
SITE_DUMMY_IDX = list(range(18, 25))


# =============================================================
# Containers
# =============================================================

@dataclass(frozen=True)
class MixedFeatureState:
    scaler: StandardScaler
    numeric_feature_names: list[str]
    categorical_feature_names: list[str]
    train_category_levels: dict[str, list[int]]


@dataclass(frozen=True)
class DRNuisance:
    e_hat: np.ndarray
    mu0_hat: np.ndarray
    mu1_hat: np.ndarray


# =============================================================
# Mixed-data preprocessing
# =============================================================

def _validate_binary_block(
    X,
    indices: Iterable[int],
    block_name: str,
):
    """
    Validate true one-hot indicator blocks.
    """

    X = np.asarray(
        X,
        dtype=float,
    )

    for j in indices:

        values = np.unique(
            X[:, j]
        )

        valid = (
            np.isclose(values, 0.0)
            | np.isclose(values, 1.0)
        )

        if not np.all(valid):

            raise ValueError(
                f"{block_name}: column {j} "
                f"({IHDP_FEATURE_NAMES[j]}) "
                "must use 0/1 indicator coding."
            )


def _validate_two_level_categorical_block(
    X,
    indices: Iterable[int],
    block_name: str,
):
    

    X = np.asarray(
        X,
        dtype=float,
    )

    for j in indices:

        column = X[:, j]

        if not np.isfinite(
            column
        ).all():

            raise ValueError(
                f"{block_name}: column {j} "
                "contains non-finite values."
            )

        rounded = np.rint(
            column
        )

        if not np.allclose(
            column,
            rounded,
        ):

            raise ValueError(
                f"{block_name}: column {j} "
                "must use integer-valued category labels."
            )

        levels = np.unique(
            rounded.astype(int)
        )

        if len(levels) > 2:

            raise ValueError(
                f"{block_name}: column {j} "
                "has more than two levels."
            )


def _collapse_one_hot_block(
    X,
    indices,
    block_name,
):
  

    X = np.asarray(
        X,
        dtype=float,
    )

    _validate_binary_block(
        X,
        indices,
        block_name,
    )

    block = X[
        :,
        indices
    ]

    active = np.isclose(
        block,
        1.0,
    )

    counts = active.sum(
        axis=1
    )

    if np.any(
        counts > 1
    ):

        bad_rows = np.where(
            counts > 1
        )[0][:10]

        raise ValueError(
            f"{block_name} is not mutually exclusive. "
            f"Example rows: {bad_rows.tolist()}."
        )

    category = np.zeros(
        len(X),
        dtype=int,
    )

    has_active = (
        counts == 1
    )

    category[
        has_active
    ] = (
        np.argmax(
            active[
                has_active
            ],
            axis=1,
        )
        + 1
    )

    return category


def fit_mixed_feature_preprocessor(
    X_discovery,
):
    

    X_discovery = np.asarray(
        X_discovery,
        dtype=float,
    )

    if X_discovery.shape[1] != 25:
        raise ValueError(
            "Expected exactly 25 IHDP covariates."
        )

    _validate_two_level_categorical_block(
        X_discovery,
        STANDALONE_BINARY_IDX,
        "standalone categorical block",
    )

    education = _collapse_one_hot_block(
        X_discovery,
        EDUCATION_DUMMY_IDX,
        "maternal education block",
    )

    site = _collapse_one_hot_block(
        X_discovery,
        SITE_DUMMY_IDX,
        "site block",
    )

    # Discovery-only scaling for the six continuous variables.
    scaler = StandardScaler()

    scaler.fit(
        X_discovery[
            :,
            CONTINUOUS_IDX
        ]
    )

    categorical_names = [
        IHDP_FEATURE_NAMES[j]
        for j
        in STANDALONE_BINARY_IDX
    ]

    categorical_names += [
        "mother_education_category",
        "site_category",
    ]

    categorical_discovery = np.column_stack(
        [
            np.rint(
                X_discovery[
                    :,
                    STANDALONE_BINARY_IDX
                ]
            ).astype(int),

            education,
            site,
        ]
    )

    levels = {
        name:
            sorted(
                np.unique(
                    categorical_discovery[
                        :,
                        j
                    ]
                )
                .astype(int)
                .tolist()
            )

        for j, name
        in enumerate(
            categorical_names
        )
    }

    return MixedFeatureState(
        scaler=scaler,

        numeric_feature_names=[
            IHDP_FEATURE_NAMES[j]
            for j
            in CONTINUOUS_IDX
        ],

        categorical_feature_names=
            categorical_names,

        train_category_levels=
            levels,
    )


def transform_mixed_features(
    X,
    state,
    *,
    warn_on_unseen_categories=True,
):
    

    X = np.asarray(
        X,
        dtype=float,
    )

    if X.shape[1] != 25:
        raise ValueError(
            "Expected exactly 25 IHDP covariates."
        )

    _validate_two_level_categorical_block(
        X,
        STANDALONE_BINARY_IDX,
        "standalone categorical block",
    )

    education = _collapse_one_hot_block(
        X,
        EDUCATION_DUMMY_IDX,
        "maternal education block",
    )

    site = _collapse_one_hot_block(
        X,
        SITE_DUMMY_IDX,
        "site block",
    )

    # Continuous portion
    X_numeric = (
        state.scaler
        .transform(
            X[
                :,
                CONTINUOUS_IDX
            ]
        )
        .astype(float)
    )

    # Categorical portion
    X_categorical = np.column_stack(
        [
            np.rint(
                X[
                    :,
                    STANDALONE_BINARY_IDX
                ]
            ).astype(int),

            education,
            site,
        ]
    ).astype(int)

    # Check whether evaluation introduces categories that were
    # absent from discovery.
    if warn_on_unseen_categories:

        for j, name in enumerate(
            state.categorical_feature_names
        ):

            observed = set(
                np.unique(
                    X_categorical[
                        :,
                        j
                    ]
                )
                .astype(int)
                .tolist()
            )

            known = set(
                state.train_category_levels[
                    name
                ]
            )

            unseen = sorted(
                observed.difference(
                    known
                )
            )

            if unseen:

                warnings.warn(
                    f"Evaluation contains unseen categories "
                    f"{unseen} for {name}.",
                    RuntimeWarning,
                )

    n_numeric = (
        X_numeric.shape[1]
    )

    n_categorical = (
        X_categorical.shape[1]
    )

    X_kproto = np.empty(
        (
            len(X),
            n_numeric
            + n_categorical,
        ),
        dtype=object,
    )

    X_kproto[
        :,
        :n_numeric
    ] = X_numeric

    X_kproto[
        :,
        n_numeric:
    ] = X_categorical

    categorical_indices = list(
        range(
            n_numeric,
            n_numeric
            + n_categorical,
        )
    )

    return (
        X_numeric,
        X_categorical,
        X_kproto,
        categorical_indices,
    )


# =============================================================
# Fixed K-Prototypes gamma
# =============================================================

def default_fixed_gamma(
    X_numeric_scaled,
):
    

    X_numeric_scaled = np.asarray(
        X_numeric_scaled,
        dtype=float,
    )

    mean_sd = float(
        np.mean(
            np.std(
                X_numeric_scaled,
                axis=0,
                ddof=0,
            )
        )
    )

    gamma = (
        0.5
        * mean_sd
    )

    if (
        not np.isfinite(
            gamma
        )
        or gamma <= 0
    ):
        gamma = 0.5

    return float(
        gamma
    )


# =============================================================
# Fixed-K K-Prototypes fitting
# =============================================================

def fit_kprototypes_groups(
    X_discovery,
    X_evaluation,
    *,
    n_clusters=3,
    init="Huang",
    n_init=20,
    seed=7,
    gamma=None,
):
    

    # ---------------------------------------------------------
    # Discovery-fitted preprocessing
    # ---------------------------------------------------------

    state = (
        fit_mixed_feature_preprocessor(
            X_discovery
        )
    )

    (
        X_numeric_discovery,
        _,
        X_kproto_discovery,
        categorical_indices,
    ) = transform_mixed_features(
        X_discovery,
        state,
        warn_on_unseen_categories=False,
    )

    (
        _,
        _,
        X_kproto_evaluation,
        categorical_indices_eval,
    ) = transform_mixed_features(
        X_evaluation,
        state,
        warn_on_unseen_categories=True,
    )

    if (
        categorical_indices
        != categorical_indices_eval
    ):

        raise RuntimeError(
            "Discovery and evaluation categorical "
            "indices do not match."
        )

    # ---------------------------------------------------------
    # Fixed discovery-only gamma
    # ---------------------------------------------------------

    resolved_gamma = (
        default_fixed_gamma(
            X_numeric_discovery
        )
        if gamma is None
        else float(gamma)
    )

    if (
        not np.isfinite(
            resolved_gamma
        )
        or resolved_gamma <= 0
    ):

        raise ValueError(
            "gamma must be positive and finite."
        )

    # ---------------------------------------------------------
    # K-Prototypes
    # ---------------------------------------------------------

    model = KPrototypes(
        n_clusters=int(
            n_clusters
        ),

        init=init,

        n_init=int(
            n_init
        ),

        gamma=
            resolved_gamma,

        random_state=int(
            seed
        ),

        verbose=0,
    )

    cluster_discovery = np.asarray(
        model.fit_predict(
            X_kproto_discovery,
            categorical=
                categorical_indices,
        ),
        dtype=int,
    )

    cluster_evaluation = np.asarray(
        model.predict(
            X_kproto_evaluation,
            categorical=
                categorical_indices,
        ),
        dtype=int,
    )

    return {
        "model":
            model,

        "preprocessor":
            state,

        "gamma":
            resolved_gamma,

        "categorical_indices":
            categorical_indices,

        "cluster_discovery":
            cluster_discovery,

        "cluster_evaluation":
            cluster_evaluation,
    }


# =============================================================
# Discovery DR pseudo-outcome
# =============================================================

def _fit_regressor_or_constant(
    X,
    y,
    *,
    seed,
    n_estimators=300,
    min_samples_leaf=10,
) -> Callable:
   

    X = np.asarray(
        X,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    ).reshape(-1)

    if len(y) == 0:
        raise ValueError(
            "Cannot fit an outcome model on zero rows."
        )

    if np.std(
        y
    ) < 1e-12:

        constant = float(
            np.mean(y)
        )

        return lambda X_new: np.full(
            len(X_new),
            constant,
            dtype=float,
        )

    model = RandomForestRegressor(
        n_estimators=int(
            n_estimators
        ),

        min_samples_leaf=int(
            min_samples_leaf
        ),

        max_features="sqrt",

        random_state=int(
            seed
        ),

        n_jobs=-1,
    )

    model.fit(
        X,
        y,
    )

    return lambda X_new: np.asarray(
        model.predict(
            np.asarray(
                X_new,
                dtype=float,
            )
        ),
        dtype=float,
    )


def crossfit_dr_nuisance_continuous_y(
    X,
    T,
    Y,
    *,
    n_splits=5,
    seed=7,
    propensity_clip=1e-6,
    outcome_n_estimators=300,
    outcome_min_samples_leaf=10,
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
            "X, T, and Y must have equal lengths."
        )

    if (
        min(
            np.bincount(
                T,
                minlength=2,
            )
        )
        < int(n_splits)
    ):

        raise ValueError(
            "Each treatment arm must contain at least "
            "n_splits observations."
        )

    n = len(Y)

    e_hat = np.full(
        n,
        np.nan,
    )

    mu0_hat = np.full(
        n,
        np.nan,
    )

    mu1_hat = np.full(
        n,
        np.nan,
    )

    folds = StratifiedKFold(
        n_splits=int(
            n_splits
        ),
        shuffle=True,
        random_state=int(
            seed
        ),
    )

    for fold_id, (
        train_idx,
        score_idx,
    ) in enumerate(
        folds.split(
            X,
            T,
        )
    ):

        X_train = X[
            train_idx
        ]

        T_train = T[
            train_idx
        ]

        Y_train = Y[
            train_idx
        ]

        # -----------------------------------------------------
        # Propensity
        # -----------------------------------------------------

        propensity = Pipeline(
            steps=[
                (
                    "scale",
                    StandardScaler(),
                ),
                (
                    "logistic",
                    LogisticRegression(
                        max_iter=5000,
                        solver="lbfgs",
                        random_state=int(
                            seed
                        ),
                    ),
                ),
            ]
        )

        propensity.fit(
            X_train,
            T_train,
        )

        e_hat[
            score_idx
        ] = np.clip(
            propensity.predict_proba(
                X[
                    score_idx
                ]
            )[:, 1],

            propensity_clip,
            1.0 - propensity_clip,
        )

        # -----------------------------------------------------
        # Treatment-specific outcome models
        # -----------------------------------------------------

        mu1_predict = (
            _fit_regressor_or_constant(
                X_train[
                    T_train == 1
                ],

                Y_train[
                    T_train == 1
                ],

                seed=
                    int(seed)
                    + 1000
                    + fold_id,

                n_estimators=
                    outcome_n_estimators,

                min_samples_leaf=
                    outcome_min_samples_leaf,
            )
        )

        mu0_predict = (
            _fit_regressor_or_constant(
                X_train[
                    T_train == 0
                ],

                Y_train[
                    T_train == 0
                ],

                seed=
                    int(seed)
                    + 2000
                    + fold_id,

                n_estimators=
                    outcome_n_estimators,

                min_samples_leaf=
                    outcome_min_samples_leaf,
            )
        )

        mu1_hat[
            score_idx
        ] = mu1_predict(
            X[
                score_idx
            ]
        )

        mu0_hat[
            score_idx
        ] = mu0_predict(
            X[
                score_idx
            ]
        )

    if (
        np.isnan(e_hat).any()
        or np.isnan(mu0_hat).any()
        or np.isnan(mu1_hat).any()
    ):

        raise RuntimeError(
            "Some discovery rows did not receive "
            "cross-fitted nuisance estimates."
        )

    return DRNuisance(
        e_hat=e_hat,
        mu0_hat=mu0_hat,
        mu1_hat=mu1_hat,
    )


def dr_treatment_effect_score(
    T,
    Y,
    nuisance,
    *,
    clip=0.05,
):
    

    T = np.asarray(
        T,
        dtype=int,
    )

    Y = np.asarray(
        Y,
        dtype=float,
    )

    e = np.clip(
        nuisance.e_hat,
        float(clip),
        1.0 - float(clip),
    )

    mu0 = nuisance.mu0_hat
    mu1 = nuisance.mu1_hat

    return (
        mu1
        - mu0

        + T
        / e
        * (
            Y - mu1
        )

        - (1 - T)
        / (1 - e)
        * (
            Y - mu0
        )
    )


# =============================================================
# Discovery cluster effect/gain ranking
# =============================================================

def build_frozen_cluster_ranking(
    cluster_discovery,
    dr_score_discovery,
    *,
    rank_by: Literal[
        "gain",
        "tau",
    ] = "gain",
    seed=7,
):
   

    frame = pd.DataFrame(
        {
            "cluster":
                np.asarray(
                    cluster_discovery,
                    dtype=int,
                ),

            "dr_effect_score":
                np.asarray(
                    dr_score_discovery,
                    dtype=float,
                ),
        }
    )

    table = (
        frame
        .groupby(
            "cluster"
        )
        .agg(
            n_discovery=(
                "dr_effect_score",
                "size",
            ),

            tau_cluster_discovery=(
                "dr_effect_score",
                "mean",
            ),

            sd_cluster_discovery=(
                "dr_effect_score",
                "std",
            ),
        )
        .reset_index()
    )

    table[
        "sd_cluster_discovery"
    ] = table[
        "sd_cluster_discovery"
    ].fillna(
        0.0
    )

    table[
        "se_cluster_discovery"
    ] = (
        table[
            "sd_cluster_discovery"
        ]
        / np.sqrt(
            table[
                "n_discovery"
            ]
        )
    )

    # Aggregate gain
    table[
        "gain_discovery"
    ] = (
        table[
            "tau_cluster_discovery"
        ]
        * table[
            "n_discovery"
        ]
    )

    if rank_by == "gain":

        primary = (
            "gain_discovery"
        )

    elif rank_by == "tau":

        primary = (
            "tau_cluster_discovery"
        )

    else:

        raise ValueError(
            "rank_by must be 'gain' or 'tau'."
        )

    # Seeded tie resolution
    rng = np.random.default_rng(
        int(seed)
    )

    table[
        "_tie"
    ] = rng.random(
        len(table)
    )

    table = (
        table
        .sort_values(
            [
                primary,
                "tau_cluster_discovery",
                "_tie",
            ],

            ascending=[
                False,
                False,
                False,
            ],
        )
        .drop(
            columns="_tie"
        )
        .reset_index(
            drop=True
        )
    )

    table[
        "frozen_rank_discovery"
    ] = np.arange(
        1,
        len(table) + 1,
    )

    table[
        "rank_by"
    ] = rank_by

    return table


# =============================================================
# Evaluation cluster effects
# =============================================================

def assign_evaluation_cluster_effects(
    cluster_evaluation,
    frozen_cluster_table,
):
    """
    Assign each evaluation observation the discovery-estimated
    effect of its frozen K-Prototypes cluster.
    """

    effect_map = (
        frozen_cluster_table
        .set_index(
            "cluster"
        )[
            "tau_cluster_discovery"
        ]
        .to_dict()
    )

    values = np.array(
        [
            effect_map.get(
                int(cluster),
                np.nan,
            )

            for cluster in np.asarray(
                cluster_evaluation
            )
        ],

        dtype=float,
    )

    if np.isnan(
        values
    ).any():

        raise RuntimeError(
            "An evaluation cluster is missing from "
            "the discovery cluster table."
        )

    return values


# =============================================================
# Frozen K-Prototypes treatment policy
# =============================================================

def deploy_frozen_cluster_policy(
    cluster_evaluation,
    frozen_cluster_table,
    *,
    q=0.70,
    seed=7,
    min_gain=0.0,
):
    """
    Deploy the frozen K-Prototypes subgroup policy.

    We specifically note that unlike the observational state-shift policies, IHDP is a
    standard binary treatment-assignment policy.

    Therefore the budget is

        round(q * N_evaluation),

    not round(q * N_T=1).

    Only discovery-positive-gain clusters are admitted.
    """

    cluster_evaluation = np.asarray(
        cluster_evaluation,
        dtype=int,
    )

    n_evaluation = len(
        cluster_evaluation
    )

    budget_cap_n = int(
        round(
            float(q)
            * n_evaluation
        )
    )

    policy = np.zeros(
        n_evaluation,
        dtype=int,
    )

    counts = (
        pd.Series(
            cluster_evaluation,
            name="cluster",
        )
        .value_counts()
        .rename(
            "n_evaluation"
        )
        .rename_axis(
            "cluster"
        )
        .reset_index()
    )

    deployment = (
        frozen_cluster_table
        .merge(
            counts,
            on="cluster",
            how="left",
        )
    )

    deployment[
        "n_evaluation"
    ] = (
        deployment[
            "n_evaluation"
        ]
        .fillna(0)
        .astype(int)
    )

    deployment[
        "positive_gain_admitted"
    ] = (
        deployment[
            "gain_discovery"
        ]
        > float(
            min_gain
        )
    )

    deployment[
        "selected_n_evaluation"
    ] = 0

    deployment[
        "selected_cluster"
    ] = False

    deployment[
        "boundary_cluster"
    ] = False

    rng = np.random.default_rng(
        int(seed)
    )

    selected_n = 0
    selected_clusters = []

    admissible = (
        deployment[
            deployment[
                "positive_gain_admitted"
            ]
        ]
        .sort_values(
            "frozen_rank_discovery"
        )
    )

    for row_index, row in admissible.iterrows():

        if selected_n >= budget_cap_n:
            break

        cluster = int(
            row[
                "cluster"
            ]
        )

        members = np.flatnonzero(
            cluster_evaluation
            == cluster
        )

        if len(members) == 0:
            continue

        remaining = (
            budget_cap_n
            - selected_n
        )

        # Entire cluster fits
        if len(members) <= remaining:

            chosen = members
            boundary = False

        # Boundary cluster
        else:

            chosen = rng.choice(
                members,
                size=remaining,
                replace=False,
            )

            boundary = True

        policy[
            chosen
        ] = 1

        chosen_n = int(
            len(chosen)
        )

        selected_n += (
            chosen_n
        )

        deployment.loc[
            row_index,
            "selected_n_evaluation",
        ] = chosen_n

        deployment.loc[
            row_index,
            "selected_cluster",
        ] = (
            chosen_n > 0
        )

        deployment.loc[
            row_index,
            "boundary_cluster",
        ] = boundary

        selected_clusters.append(
            cluster
        )

        if boundary:
            break

    deployment[
        "selected_fraction_within_cluster"
    ] = np.where(
        deployment[
            "n_evaluation"
        ] > 0,

        deployment[
            "selected_n_evaluation"
        ]
        / deployment[
            "n_evaluation"
        ],

        np.nan,
    )

    deployment[
        "budget_cap_n"
    ] = budget_cap_n

    deployment[
        "actual_selected_n"
    ] = selected_n

    deployment[
        "unused_budget_n"
    ] = (
        budget_cap_n
        - selected_n
    )

    deployment[
        "budget_utilization"
    ] = (
        selected_n
        / budget_cap_n
        if budget_cap_n > 0
        else np.nan
    )

    deployment = (
        deployment
        .sort_values(
            "frozen_rank_discovery"
        )
        .reset_index(
            drop=True
        )
    )

    return (
        policy,
        deployment,
        selected_clusters,
    )


# =============================================================
# Complete K-Prototypes subgroup comparator
# =============================================================

def fit_kprototypes_subgroup_policy(
    X_discovery,
    T_discovery,
    Y_discovery,
    X_evaluation,
    *,
    n_clusters=3,
    q=0.70,
    init="Huang",
    n_init=20,
    seed=7,
    gamma=None,
    n_splits=5,
    dr_score_clip=0.05,
    rank_by="gain",
    min_gain=0.0,
):
   

    # =========================================================
    # 1. Fit frozen K-Prototypes partition
    # =========================================================

    clustering = (
        fit_kprototypes_groups(
            X_discovery,
            X_evaluation,

            n_clusters=
                n_clusters,

            init=
                init,

            n_init=
                n_init,

            seed=
                seed,

            gamma=
                gamma,
        )
    )

    # =========================================================
    # 2. Discovery OOF DR treatment-effect scores
    # =========================================================

    nuisance = (
        crossfit_dr_nuisance_continuous_y(
            X_discovery,
            T_discovery,
            Y_discovery,

            n_splits=
                n_splits,

            seed=
                seed,
        )
    )

    dr_score_discovery = (
        dr_treatment_effect_score(
            T_discovery,
            Y_discovery,
            nuisance,

            clip=
                dr_score_clip,
        )
    )

    # =========================================================
    # 3. Freeze discovery cluster ranking
    # =========================================================

    frozen_cluster_table = (
        build_frozen_cluster_ranking(
            clustering[
                "cluster_discovery"
            ],

            dr_score_discovery,

            rank_by=
                rank_by,

            seed=
                seed,
        )
    )

    # =========================================================
    # 4. Assign evaluation cluster effects
    # =========================================================

    tau_cluster_evaluation = (
        assign_evaluation_cluster_effects(
            clustering[
                "cluster_evaluation"
            ],

            frozen_cluster_table,
        )
    )

    # =========================================================
    # 5. Build frozen evaluation treatment policy
    # =========================================================

    (
        policy_evaluation,
        deployment_table,
        selected_clusters,
    ) = deploy_frozen_cluster_policy(
        clustering[
            "cluster_evaluation"
        ],

        frozen_cluster_table,

        q=
            q,

        seed=
            seed,

        min_gain=
            min_gain,
    )

    return {
        "model":
            clustering[
                "model"
            ],

        "preprocessor":
            clustering[
                "preprocessor"
            ],

        "gamma":
            clustering[
                "gamma"
            ],

        "categorical_indices":
            clustering[
                "categorical_indices"
            ],

        "cluster_discovery":
            clustering[
                "cluster_discovery"
            ],

        "cluster_evaluation":
            clustering[
                "cluster_evaluation"
            ],

        "discovery_nuisance":
            nuisance,

        "dr_score_discovery":
            dr_score_discovery,

        "frozen_cluster_table":
            frozen_cluster_table,

        "tau_cluster_evaluation":
            tau_cluster_evaluation,

        "policy_evaluation":
            policy_evaluation,

        "deployment_table":
            deployment_table,

        "selected_clusters":
            selected_clusters,
    }

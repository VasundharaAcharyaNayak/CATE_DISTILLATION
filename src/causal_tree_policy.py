import numpy as np
import pandas as pd

from causalml.inference.tree import CausalTreeRegressor

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.policy_utils import get_budget_count


# =============================================================
# Discovery nuisance models for AIPW leaf-effect estimation
# =============================================================

def _fit_binary_or_constant(X, y, seed):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int).reshape(-1)

    if len(y) == 0:
        raise ValueError(
            "Cannot fit nuisance model on zero observations."
        )

    if np.unique(y).size == 1:
        p = float(y.mean())

        return lambda X_new: np.full(
            len(X_new),
            p,
            dtype=float,
        )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=10_000,
            random_state=int(seed),
        ),
    )

    model.fit(X, y)

    return lambda X_new: model.predict_proba(
        np.asarray(X_new, dtype=float)
    )[:, 1]


def crossfit_nuisance(
    Z,
    T,
    Y,
    n_splits=5,
    seed=7,
):
   

    Z = np.asarray(Z, dtype=float)
    T = np.asarray(T, dtype=int).reshape(-1)
    Y = np.asarray(Y, dtype=int).reshape(-1)

    if not (
        len(Z)
        == len(T)
        == len(Y)
    ):
        raise ValueError(
            "Z, T, and Y must have equal numbers of rows."
        )

    counts = np.bincount(
        T,
        minlength=2,
    )

    folds = min(
        int(n_splits),
        int(counts.min()),
    )

    if folds < 2:
        raise ValueError(
            "Insufficient treatment overlap for "
            f"cross-fitting: {counts.tolist()}."
        )

    e = np.full(
        len(Y),
        np.nan,
        dtype=float,
    )

    mu0 = np.full(
        len(Y),
        np.nan,
        dtype=float,
    )

    mu1 = np.full(
        len(Y),
        np.nan,
        dtype=float,
    )

    cv = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=int(seed),
    )

    for fold_id, (
        fit_idx,
        score_idx,
    ) in enumerate(
        cv.split(Z, T),
        start=1,
    ):

        # -----------------------------------------------------
        # Propensity model
        # -----------------------------------------------------

        propensity = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=10_000,
                random_state=int(seed) + fold_id,
            ),
        )

        propensity.fit(
            Z[fit_idx],
            T[fit_idx],
        )

        e[score_idx] = np.clip(
            propensity.predict_proba(
                Z[score_idx]
            )[:, 1],
            1e-6,
            1.0 - 1e-6,
        )

        # -----------------------------------------------------
        # Outcome regressions
        # -----------------------------------------------------

        fit0 = fit_idx[
            T[fit_idx] == 0
        ]

        fit1 = fit_idx[
            T[fit_idx] == 1
        ]

        mu0_predict = _fit_binary_or_constant(
            Z[fit0],
            Y[fit0],
            int(seed) + 1000 + fold_id,
        )

        mu1_predict = _fit_binary_or_constant(
            Z[fit1],
            Y[fit1],
            int(seed) + 2000 + fold_id,
        )

        mu0[score_idx] = np.clip(
            mu0_predict(
                Z[score_idx]
            ),
            1e-6,
            1.0 - 1e-6,
        )

        mu1[score_idx] = np.clip(
            mu1_predict(
                Z[score_idx]
            ),
            1e-6,
            1.0 - 1e-6,
        )

    return {
        "e": e,
        "mu0": mu0,
        "mu1": mu1,
    }


# =============================================================
# AIPW treatment-effect pseudo-outcome
# =============================================================

def aipw_effect_score(
    T,
    Y,
    e,
    mu0,
    mu1,
    trim=0.05,
):
   

    T = np.asarray(
        T,
        dtype=int,
    ).reshape(-1)

    Y = np.asarray(
        Y,
        dtype=float,
    ).reshape(-1)

    e = np.clip(
        np.asarray(
            e,
            dtype=float,
        ).reshape(-1),
        float(trim),
        1.0 - float(trim),
    )

    mu0 = np.asarray(
        mu0,
        dtype=float,
    ).reshape(-1)

    mu1 = np.asarray(
        mu1,
        dtype=float,
    ).reshape(-1)

    return (
        mu1
        - mu0
        + T / e
        * (Y - mu1)
        - (1 - T)
        / (1 - e)
        * (Y - mu0)
    )


# =============================================================
# Causal Tree
# =============================================================

def make_causal_tree(
    k,
    seed,
    min_group_samples,
    min_samples_leaf,
):
    return CausalTreeRegressor(
        criterion="causal_mse",
        splitter="best",
        control_name=0,
        max_leaf_nodes=int(k),
        min_group_samples=int(
            min_group_samples
        ),
        min_samples_leaf=int(
            min_samples_leaf
        ),
        min_samples_split=max(
            2 * int(min_samples_leaf),
            2,
        ),
        min_impurity_decrease=float("-inf"),
        ccp_alpha=0.0,
        random_state=int(seed),
    )


def leaf_ids(
    model,
    X,
):
    return np.asarray(
        model.apply(
            np.asarray(
                X,
                dtype=float,
            )
        ),
        dtype=int,
    ).reshape(-1)


# =============================================================
# Exact-K fitting
# =============================================================

def fit_exact_k_tree(
    X,
    T,
    Y,
    k,
    seed=7,
    preferred_min_group_samples=5,
    preferred_min_samples_leaf=10,
):
   

    group_candidates = []

    for value in [
        preferred_min_group_samples,
        5,
        4,
        3,
        2,
        1,
    ]:
        value = int(value)

        if value not in group_candidates:
            group_candidates.append(
                value
            )

    leaf_candidates = []

    for value in [
        preferred_min_samples_leaf,
        10,
        8,
        6,
        5,
        4,
        3,
        2,
    ]:
        value = int(value)

        if value not in leaf_candidates:
            leaf_candidates.append(
                value
            )

    attempts = []

    for min_group in group_candidates:

        for min_leaf in leaf_candidates:

            model = make_causal_tree(
                k=k,
                seed=seed,
                min_group_samples=min_group,
                min_samples_leaf=min_leaf,
            )

            try:

                model.fit(
                    X=np.asarray(
                        X,
                        dtype=float,
                    ),
                    treatment=np.asarray(
                        T,
                        dtype=int,
                    ),
                    y=np.asarray(
                        Y,
                        dtype=float,
                    ),
                )

                leaves = leaf_ids(
                    model,
                    X,
                )

                realized = int(
                    np.unique(
                        leaves
                    ).size
                )

                attempts.append(
                    {
                        "min_group_samples":
                            min_group,

                        "min_samples_leaf":
                            min_leaf,

                        "realized_leaves":
                            realized,
                    }
                )

                if realized == int(k):

                    return (
                        model,
                        {
                            "K":
                                int(k),

                            "realized_leaves":
                                realized,

                            "min_group_samples":
                                min_group,

                            "min_samples_leaf":
                                min_leaf,

                            "min_samples_split":
                                max(
                                    2 * min_leaf,
                                    2,
                                ),

                            "seed":
                                int(seed),

                            "attempts":
                                attempts,
                        },
                    )

            except Exception as exc:

                attempts.append(
                    {
                        "min_group_samples":
                            min_group,

                        "min_samples_leaf":
                            min_leaf,

                        "realized_leaves":
                            np.nan,

                        "error":
                            (
                                f"{type(exc).__name__}: "
                                f"{exc}"
                            ),
                    }
                )

    raise RuntimeError(
        f"Could not realize exactly K={k} leaves.\n"
        + pd.DataFrame(
            attempts
        ).to_string(
            index=False
        )
    )


# =============================================================
# Honest leaf-effect estimation and gain ranking
# =============================================================

def build_leaf_table(
    leaf_est,
    T_est,
    gamma_est,
    leaf_eval,
    T_eval,
    min_gain=0.0,
    seed=7,
):
   

    leaf_est = np.asarray(
        leaf_est,
        dtype=int,
    ).reshape(-1)

    T_est = np.asarray(
        T_est,
        dtype=int,
    ).reshape(-1)

    gamma_est = np.asarray(
        gamma_est,
        dtype=float,
    ).reshape(-1)

    leaf_eval = np.asarray(
        leaf_eval,
        dtype=int,
    ).reshape(-1)

    T_eval = np.asarray(
        T_eval,
        dtype=int,
    ).reshape(-1)

    discovery = pd.DataFrame(
        {
            "leaf":
                leaf_est,

            "T":
                T_est,

            "gamma":
                gamma_est,
        }
    )

    eligible_discovery = discovery[
        discovery["T"] == 1
    ].copy()

    table = (
        eligible_discovery
        .groupby("leaf")
        .agg(
            n_eligible_discovery=(
                "gamma",
                "size",
            ),

            tau_leaf_discovery=(
                "gamma",
                "mean",
            ),

            sd_leaf_discovery=(
                "gamma",
                "std",
            ),
        )
        .reset_index()
    )

    table[
        "sd_leaf_discovery"
    ] = table[
        "sd_leaf_discovery"
    ].fillna(
        0.0
    )

    # Evaluation counts are used only for deployment capacity,
    # not for ranking.
    evaluation_counts = (
        pd.DataFrame(
            {
                "leaf":
                    leaf_eval,

                "T":
                    T_eval,
            }
        )
        .query(
            "T == 1"
        )
        .groupby(
            "leaf"
        )
        .size()
        .rename(
            "n_eligible_evaluation"
        )
        .reset_index()
    )

    table = table.merge(
        evaluation_counts,
        on="leaf",
        how="outer",
    )

    for column in [
        "n_eligible_discovery",
        "n_eligible_evaluation",
    ]:

        table[
            column
        ] = (
            table[column]
            .fillna(0)
            .astype(int)
        )

    if table[
        "tau_leaf_discovery"
    ].isna().any():

        bad = (
            table.loc[
                table[
                    "tau_leaf_discovery"
                ].isna(),
                "leaf",
            ]
            .astype(int)
            .tolist()
        )

        raise RuntimeError(
            "No eligible discovery observations were available "
            f"for leaf-effect estimation in leaves {bad}."
        )

    # ---------------------------------------------------------
    # Discovery aggregate gain
    # ---------------------------------------------------------

    table[
        "gain_discovery"
    ] = (
        table[
            "tau_leaf_discovery"
        ]
        * table[
            "n_eligible_discovery"
        ]
    )

    table[
        "positive_gain"
    ] = (
        table[
            "gain_discovery"
        ]
        > float(
            min_gain
        )
    )

    # Seeded tie resolution used in the analysis.
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
                "gain_discovery",
                "tau_leaf_discovery",
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
        "rank"
    ] = (
        np.arange(
            len(table)
        )
        + 1
    )

    return table


# =============================================================
# Evaluation allocation
# =============================================================

def build_leaf_policy(
    leaf_eval,
    T_eval,
    leaf_table,
    q=0.70,
    min_gain=0.0,
    seed=7,
):
   

    leaf_eval = np.asarray(
        leaf_eval,
        dtype=int,
    ).reshape(-1)

    T_eval = np.asarray(
        T_eval,
        dtype=int,
    ).reshape(-1)

    if len(
        leaf_eval
    ) != len(
        T_eval
    ):
        raise ValueError(
            "leaf_eval and T_eval must have equal length."
        )

    eligible = (
        T_eval == 1
    )

    n_eligible = int(
        eligible.sum()
    )

    cap = get_budget_count(
        n_eligible,
        q,
    )

    targeted = np.zeros(
        len(T_eval),
        dtype=int,
    )

    deployment = (
        leaf_table.copy()
    )

    deployment[
        "positive_gain"
    ] = (
        deployment[
            "gain_discovery"
        ]
        > float(
            min_gain
        )
    )

    deployment[
        "selected_eval_n"
    ] = 0

    deployment[
        "boundary_leaf"
    ] = False

    rng = np.random.default_rng(
        int(seed)
    )

    selected_n = 0
    selected_leaves = []
    boundary_leaf = None

    admissible = (
        deployment[
            deployment[
                "positive_gain"
            ]
        ]
        .sort_values(
            "rank"
        )
    )

    for idx, row in admissible.iterrows():

        if selected_n >= cap:
            break

        leaf = int(
            row["leaf"]
        )

        members = np.flatnonzero(
            (leaf_eval == leaf)
            & eligible
        )

        remaining = (
            cap
            - selected_n
        )

        if len(
            members
        ) <= remaining:

            chosen = members
            is_boundary = False

        else:

            chosen = rng.choice(
                members,
                size=remaining,
                replace=False,
            )

            is_boundary = True

        targeted[
            chosen
        ] = 1

        selected_n += len(
            chosen
        )

        deployment.loc[
            idx,
            "selected_eval_n",
        ] = int(
            len(chosen)
        )

        deployment.loc[
            idx,
            "boundary_leaf",
        ] = bool(
            is_boundary
        )

        selected_leaves.append(
            leaf
        )

        if is_boundary:
            boundary_leaf = leaf
            break

    deployment[
        "budget_cap_n"
    ] = cap

    deployment[
        "selected_total_n"
    ] = int(
        selected_n
    )

    deployment[
        "unused_budget_n"
    ] = int(
        cap
        - selected_n
    )

    deployment[
        "selected_fraction_in_eligible_leaf"
    ] = np.where(
        deployment[
            "n_eligible_evaluation"
        ] > 0,

        deployment[
            "selected_eval_n"
        ]
        / deployment[
            "n_eligible_evaluation"
        ],

        np.nan,
    )

    return (
        targeted,
        deployment
        .sort_values(
            "rank"
        )
        .reset_index(
            drop=True
        ),
        selected_leaves,
        boundary_leaf,
    )


# =============================================================
# Convert targeting to full leaf-level s(X)
# =============================================================

def targeting_to_leaf_shift_probability(
    leaf_eval,
    T_eval,
    targeted,
):
   

    leaf_eval = np.asarray(
        leaf_eval,
        dtype=int,
    ).reshape(-1)

    T_eval = np.asarray(
        T_eval,
        dtype=int,
    ).reshape(-1)

    targeted = np.asarray(
        targeted,
        dtype=int,
    ).reshape(-1)

    if not (
        len(leaf_eval)
        == len(T_eval)
        == len(targeted)
    ):
        raise ValueError(
            "leaf_eval, T_eval, and targeted must have equal length."
        )

    shift_probability = np.zeros(
        len(T_eval),
        dtype=float,
    )

    for leaf in np.unique(
        leaf_eval
    ):

        eligible_in_leaf = (
            (leaf_eval == leaf)
            & (T_eval == 1)
        )

        n_eligible_leaf = int(
            eligible_in_leaf.sum()
        )

        if n_eligible_leaf == 0:
            continue

        selected_fraction = float(
            targeted[
                eligible_in_leaf
            ].mean()
        )

        # Common leaf-level policy probability for all
        # evaluation observations in the leaf.
        shift_probability[
            leaf_eval == leaf
        ] = selected_fraction

    return shift_probability


# =============================================================
# Complete observational causal-tree policy constructor
# =============================================================

def fit_causal_tree_subgroup_policy(
    X_discovery,
    Z_discovery,
    T_discovery,
    Y_discovery,
    X_evaluation,
    T_evaluation,
    *,
    k_groups,
    q=0.70,
    honesty_fraction=0.50,
    seed=7,
    n_splits=5,
    min_gain=0.0,
    preferred_min_group_samples=5,
    preferred_min_samples_leaf=10,
    trim=0.05,
):
    

    X_discovery = np.asarray(
        X_discovery,
        dtype=float,
    )

    Z_discovery = np.asarray(
        Z_discovery,
        dtype=float,
    )

    T_discovery = np.asarray(
        T_discovery,
        dtype=int,
    ).reshape(-1)

    Y_discovery = np.asarray(
        Y_discovery,
        dtype=int,
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
    # Honest split within discovery
    # ---------------------------------------------------------

    all_idx = np.arange(
        len(T_discovery)
    )

    (
        idx_structure,
        idx_estimation,
    ) = train_test_split(
        all_idx,
        test_size=float(
            honesty_fraction
        ),
        random_state=int(
            seed
        ),
        stratify=T_discovery,
    )

    # ---------------------------------------------------------
    # Discovery AIPW pseudo-outcomes
    # ---------------------------------------------------------

    nuisance = crossfit_nuisance(
        Z_discovery,
        T_discovery,
        Y_discovery,
        n_splits=n_splits,
        seed=seed,
    )

    gamma = aipw_effect_score(
        T_discovery,
        Y_discovery,
        nuisance["e"],
        nuisance["mu0"],
        nuisance["mu1"],
        trim=trim,
    )

    # ---------------------------------------------------------
    # Fit tree structure on honest structure half
    # ---------------------------------------------------------

    tree, tree_params = fit_exact_k_tree(
        X_discovery[
            idx_structure
        ],
        T_discovery[
            idx_structure
        ],
        Y_discovery[
            idx_structure
        ],
        k=k_groups,
        seed=seed,
        preferred_min_group_samples=
            preferred_min_group_samples,
        preferred_min_samples_leaf=
            preferred_min_samples_leaf,
    )

    leaf_structure = leaf_ids(
        tree,
        X_discovery[
            idx_structure
        ],
    )

    leaf_estimation = leaf_ids(
        tree,
        X_discovery[
            idx_estimation
        ],
    )

    leaf_evaluation = leaf_ids(
        tree,
        X_evaluation,
    )

    realized_leaves = int(
        np.unique(
            leaf_structure
        ).size
    )

    if realized_leaves != int(
        k_groups
    ):
        raise RuntimeError(
            f"Expected exactly {k_groups} leaves; "
            f"got {realized_leaves}."
        )

    # ---------------------------------------------------------
    # Estimate/rank leaf benefit on honest estimation half
    # ---------------------------------------------------------

    leaf_table = build_leaf_table(
        leaf_est=leaf_estimation,

        T_est=T_discovery[
            idx_estimation
        ],

        gamma_est=gamma[
            idx_estimation
        ],

        leaf_eval=leaf_evaluation,

        T_eval=T_evaluation,

        min_gain=min_gain,
        seed=seed,
    )

    # ---------------------------------------------------------
    # Apply frozen discovery ranking to evaluation
    # ---------------------------------------------------------

    (
        targeted_evaluation,
        deployment,
        selected_leaves,
        boundary_leaf,
    ) = build_leaf_policy(
        leaf_eval=leaf_evaluation,
        T_eval=T_evaluation,
        leaf_table=leaf_table,
        q=q,
        min_gain=min_gain,
        seed=seed,
    )

    shift_probability = (
        targeting_to_leaf_shift_probability(
            leaf_eval=leaf_evaluation,
            T_eval=T_evaluation,
            targeted=targeted_evaluation,
        )
    )

    n_eval_eligible = int(
        (
            T_evaluation == 1
        ).sum()
    )

    metadata = {
        "requested_leaves":
            int(k_groups),

        "realized_leaves":
            realized_leaves,

        "honesty_fraction":
            float(
                honesty_fraction
            ),

        "structure_n":
            int(
                len(
                    idx_structure
                )
            ),

        "leaf_estimation_n":
            int(
                len(
                    idx_estimation
                )
            ),

        "eligible_evaluation_n":
            n_eval_eligible,

        "budget_cap_n":
            get_budget_count(
                n_eval_eligible,
                q,
            ),

        "selected_eligible_n":
            int(
                targeted_evaluation.sum()
            ),

        "expected_shifted_eligible_n":
            float(
                shift_probability[
                    T_evaluation == 1
                ].sum()
            ),

        "selected_leaves":
            selected_leaves,

        "boundary_leaf":
            boundary_leaf,
    }

    return {
        "model":
            tree,

        "tree_params":
            tree_params,

        "leaf_table":
            deployment,

        "leaf_evaluation":
            leaf_evaluation,

        "targeted_evaluation":
            targeted_evaluation,

        "shift_probability_evaluation":
            shift_probability,

        "metadata":
            metadata,
    }

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.dr_evaluator import _fit_logit_or_constant
from src.policy_utils import get_budget_count


# =============================================================
# Fit fixed-K K-means
# =============================================================

def fit_kmeans_policy_groups(
    X_discovery,
    X_evaluation,
    n_clusters,
    standardize=True,
    n_init=50,
    seed=0,
):
   

    X_discovery = np.asarray(
        X_discovery,
        dtype=float,
    )

    X_evaluation = np.asarray(
        X_evaluation,
        dtype=float,
    )

    if X_discovery.ndim != 2:
        raise ValueError(
            "X_discovery must be a two-dimensional matrix."
        )

    if X_evaluation.ndim != 2:
        raise ValueError(
            "X_evaluation must be a two-dimensional matrix."
        )

    if X_discovery.shape[1] != X_evaluation.shape[1]:
        raise ValueError(
            "Discovery and evaluation clustering matrices "
            "must have the same number of columns."
        )

    if n_clusters < 2:
        raise ValueError(
            "n_clusters must be at least 2."
        )

    # ---------------------------------------------------------
    # Discovery-only standardization
    # ---------------------------------------------------------

    if standardize:

        scaler = StandardScaler()

        X_discovery_fit = scaler.fit_transform(
            X_discovery
        )

        X_evaluation_fit = scaler.transform(
            X_evaluation
        )

    else:

        scaler = None

        X_discovery_fit = X_discovery
        X_evaluation_fit = X_evaluation

    # ---------------------------------------------------------
    # Fixed-K clustering
    # ---------------------------------------------------------

    kmeans = KMeans(
        n_clusters=int(n_clusters),
        init="k-means++",
        n_init=int(n_init),
        random_state=int(seed),
    )

    kmeans.fit(
        X_discovery_fit
    )

    discovery_groups = kmeans.predict(
        X_discovery_fit
    )

    evaluation_groups = kmeans.predict(
        X_evaluation_fit
    )

    return (
        discovery_groups,
        evaluation_groups,
        kmeans,
        scaler,
    )


# =============================================================
# Discovery OOF benefit estimation
# =============================================================

def estimate_oof_cluster_benefits(
    Z,
    T,
    Y,
    groups,
    n_splits=5,
    seed=7,
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

    groups = np.asarray(
        groups,
        dtype=int,
    ).reshape(-1)

    n = len(Y)

    if not (
        len(Z)
        == len(T)
        == len(groups)
        == n
    ):
        raise ValueError(
            "Z, T, Y, and groups must be aligned."
        )

    if not np.all(
        np.isin(T, [0, 1])
    ):
        raise ValueError(
            "T must contain only 0 and 1."
        )

    # ---------------------------------------------------------
    # Treatment-stratified cross-fitting
    # ---------------------------------------------------------

    treatment_counts = np.bincount(
        T,
        minlength=2,
    )

    effective_folds = min(
        int(n_splits),
        int(
            treatment_counts.min()
        ),
    )

    if effective_folds < 2:
        raise RuntimeError(
            "Insufficient treatment overlap for "
            "discovery OOF benefit estimation."
        )

    splitter = StratifiedKFold(
        n_splits=effective_folds,
        shuffle=True,
        random_state=seed,
    )

    benefit_oof = np.full(
        n,
        np.nan,
        dtype=float,
    )

    for fit_idx, score_idx in splitter.split(
        Z,
        T,
    ):

        treated_fit_idx = fit_idx[
            T[fit_idx] == 1
        ]

        control_fit_idx = fit_idx[
            T[fit_idx] == 0
        ]

        # Outcome model under T=1
        mu1_predict = _fit_logit_or_constant(
            Z[treated_fit_idx],
            Y[treated_fit_idx].astype(int),
        )

        # Outcome model under T=0
        mu0_predict = _fit_logit_or_constant(
            Z[control_fit_idx],
            Y[control_fit_idx].astype(int),
        )

        mu1 = mu1_predict(
            Z[score_idx]
        )

        mu0 = mu0_predict(
            Z[score_idx]
        )

        # Positive = estimated benefit of T=1 -> T=0
        benefit_oof[
            score_idx
        ] = (
            mu1 - mu0
        )

    if np.isnan(
        benefit_oof
    ).any():
        raise RuntimeError(
            "Some discovery observations did not receive "
            "an OOF benefit estimate."
        )

    # ---------------------------------------------------------
    # Cluster summaries among eligible discovery T=1 only
    # ---------------------------------------------------------

    eligible = (
        T == 1
    )

    cluster_data = pd.DataFrame(
        {
            "cluster":
                groups[eligible],

            "benefit":
                benefit_oof[eligible],
        }
    )

    cluster_summary = (
        cluster_data
        .groupby(
            "cluster"
        )["benefit"]
        .agg(
            tau_hat="mean",

            variance=lambda x: (
                x.var(ddof=1)
                if len(x) > 1
                else np.nan
            ),

            n_train_eligible="size",

            se=lambda x: (
                x.std(ddof=1)
                / np.sqrt(len(x))
                if len(x) > 1
                else np.nan
            ),
        )
        .reset_index()
    )

    return (
        cluster_summary,
        benefit_oof,
    )


# =============================================================
# Freeze discovery cluster ranking
# =============================================================

def rank_clusters_by_gain(
    cluster_summary,
    positive_gain_only=True,
):
    """
    Rank discovery clusters by aggregate estimated gain.

        gain_c = tau_hat_c * n_eligible,c

    The ranking is learned entirely from the discovery cohort.
    """

    ranked = cluster_summary.copy()

    ranked[
        "train_gain"
    ] = (
        ranked["tau_hat"]
        * ranked["n_train_eligible"]
    )

    ranked[
        "positive_gain"
    ] = (
        ranked["train_gain"] > 0
    )

    if positive_gain_only:

        ranked = ranked[
            ranked["positive_gain"]
        ].copy()

    ranked = (
        ranked
        .sort_values(
            [
                "train_gain",
                "tau_hat",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    ranked[
        "rank"
    ] = (
        np.arange(
            len(ranked)
        )
        + 1
    )

    cluster_order = (
        ranked["cluster"]
        .astype(int)
        .tolist()
    )

    return (
        ranked,
        cluster_order,
    )


# =============================================================
# Apply frozen discovery ranking to evaluation cohort
# =============================================================

def build_cluster_shift_probability(
    evaluation_groups,
    treatment,
    cluster_order,
    budget_fraction=0.70,
):
   

    groups = np.asarray(
        evaluation_groups,
        dtype=int,
    ).reshape(-1)

    T = np.asarray(
        treatment,
        dtype=int,
    ).reshape(-1)

    if len(groups) != len(T):
        raise ValueError(
            "evaluation_groups and treatment must have "
            "equal length."
        )

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

    shift_probability = np.zeros(
        len(T),
        dtype=float,
    )

    selected_so_far = 0

    rank_map = {
        int(cluster_id): rank
        for rank, cluster_id
        in enumerate(
            cluster_order,
            start=1,
        )
    }

    # ---------------------------------------------------------
    # Sequential allocation using frozen discovery ranking
    # ---------------------------------------------------------

    for cluster_id in cluster_order:

        cluster_id = int(
            cluster_id
        )

        eligible_cluster = (
            (groups == cluster_id)
            & eligible
        )

        n_cluster_eligible = int(
            eligible_cluster.sum()
        )

        if n_cluster_eligible == 0:
            continue

        remaining_capacity = (
            target_count
            - selected_so_far
        )

        if remaining_capacity <= 0:
            break

        # -----------------------------------------------------
        # Entire cluster fits
        # -----------------------------------------------------

        if (
            n_cluster_eligible
            <= remaining_capacity
        ):

            shift_probability[
                groups == cluster_id
            ] = 1.0

            selected_so_far += (
                n_cluster_eligible
            )

        # -----------------------------------------------------
        # Boundary cluster
        # -----------------------------------------------------

        else:

            boundary_probability = (
                remaining_capacity
                / n_cluster_eligible
            )

            shift_probability[
                groups == cluster_id
            ] = (
                boundary_probability
            )

            selected_so_far += (
                remaining_capacity
            )

            break

    # ---------------------------------------------------------
    # Evaluation allocation diagnostics
    # ---------------------------------------------------------

    expected_shifted = float(
        shift_probability[
            eligible
        ].sum()
    )

    allocation_rows = []

    for cluster_id in np.unique(
        groups
    ):

        cluster_id = int(
            cluster_id
        )

        cluster_mask = (
            groups == cluster_id
        )

        eligible_cluster = (
            cluster_mask
            & eligible
        )

        cluster_probability = float(
            shift_probability[
                cluster_mask
            ][0]
        )

        allocation_rows.append(
            {
                "cluster":
                    cluster_id,

                "discovery_rank":
                    rank_map.get(
                        cluster_id,
                        np.nan,
                    ),

                "n_eval_total":
                    int(
                        cluster_mask.sum()
                    ),

                "n_eval_eligible":
                    int(
                        eligible_cluster.sum()
                    ),

                "shift_probability":
                    cluster_probability,

                "expected_shifted_eval":
                    float(
                        cluster_probability
                        * eligible_cluster.sum()
                    ),
            }
        )

    allocation_df = (
        pd.DataFrame(
            allocation_rows
        )
        .sort_values(
            "discovery_rank",
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )

    metadata = {
        "n_eligible":
            n_eligible,

        "target_count":
            target_count,

        "expected_shifted":
            expected_shifted,

        "realized_fraction_eligible":
            (
                expected_shifted
                / n_eligible
                if n_eligible > 0
                else 0.0
            ),

        "unused_budget":
            float(
                target_count
                - expected_shifted
            ),
    }

    return (
        shift_probability,
        allocation_df,
        metadata,
    )

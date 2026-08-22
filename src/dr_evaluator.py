import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


# =============================================================
# Logistic-regression nuisance helper
# =============================================================

def _fit_logit_or_constant(X, y):
    """
    Fit a binary logistic-regression nuisance model.

    If a training fold contains only one outcome class, return a
    constant-probability predictor instead. This protects bootstrap
    and cross-fitting folds from single-class failures.
    """

    X = np.asarray(
        X,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=int,
    ).reshape(-1)

    unique = np.unique(y)

    # ---------------------------------------------------------
    # Safety for folds containing only one outcome class
    # ---------------------------------------------------------

    if len(unique) == 1:

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

    # ---------------------------------------------------------
    # Logistic regression
    # ---------------------------------------------------------

    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=10_000,
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
# Cross-fitted selective-shift DR evaluator
# =============================================================

def dr_selective_shift_values_binary_y(
    Z,
    T,
    Y,
    policies,
    n_splits=5,
    trim=0.05,
    seed=7,
    clip_eps=1e-6,
    return_contributions=False,
):
   

    # ---------------------------------------------------------
    # Convert inputs
    # ---------------------------------------------------------

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

    n = len(Y)

    # ---------------------------------------------------------
    # Basic validation
    # ---------------------------------------------------------

    if not (
        len(Z)
        == len(T)
        == n
    ):
        raise ValueError(
            "Z, T, and Y must have equal numbers of rows."
        )

    # ---------------------------------------------------------
    # Validate policy shift-probability vectors
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
    # Determine feasible treatment-stratified fold count
    # ---------------------------------------------------------

    class_counts = np.bincount(
        T,
        minlength=2,
    )

    effective_folds = min(
        n_splits,
        int(
            class_counts.min()
        ),
    )

    if effective_folds < 2:
        raise RuntimeError(
            "Insufficient treatment overlap "
            "for DR cross-fitting."
        )

    splitter = StratifiedKFold(
        n_splits=effective_folds,
        shuffle=True,
        random_state=seed,
    )

    # ---------------------------------------------------------
    # One contribution vector for each frozen policy
    # ---------------------------------------------------------

    contributions = {
        name: np.full(
            n,
            np.nan,
            dtype=float,
        )
        for name
        in normalized_policies
    }

    # =========================================================
    # Cross-fitting
    # =========================================================

    for fit_idx, score_idx in splitter.split(
        Z,
        T,
    ):

        # -----------------------------------------------------
        # Propensity nuisance:
        #
        # e(Z) = P(T=1 | Z)
        # -----------------------------------------------------

        e_predict = _fit_logit_or_constant(
            Z[fit_idx],
            T[fit_idx],
        )

        e_score = np.clip(
            e_predict(
                Z[score_idx]
            ),
            clip_eps,
            1.0 - clip_eps,
        )

        # -----------------------------------------------------
        # Common overlap trimming
        # -----------------------------------------------------

        keep_local = (
            (e_score >= trim)
            & (
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

        # -----------------------------------------------------
        # Target-state outcome regression:
        #
        # mu0(Z) = E[Y | T=0, Z]
        # -----------------------------------------------------

        control_fit_idx = fit_idx[
            T[fit_idx] == 0
        ]

        mu0_predict = (
            _fit_logit_or_constant(
                Z[
                    control_fit_idx
                ],
                Y[
                    control_fit_idx
                ].astype(int),
            )
        )

        mu0 = np.clip(
            mu0_predict(
                Z[kept_idx]
            ),
            clip_eps,
            1.0 - clip_eps,
        )

        # -----------------------------------------------------
        # Evaluation-fold quantities
        # -----------------------------------------------------

        ti = T[
            kept_idx
        ].astype(float)

        yi = Y[
            kept_idx
        ]

        ei = e_score[
            keep_local
        ]

        # -----------------------------------------------------
        # Evaluate every policy using the SAME nuisance fits
        # and the SAME overlap-restricted observations.
        # -----------------------------------------------------

        for (
            name,
            shift_prob,
        ) in normalized_policies.items():

            si = shift_prob[
                kept_idx
            ]

            # -------------------------------------------------
            # Selective T=1 -> T=0 doubly robust score
            #
            # (1 - T*s)Y
            # + T*s*mu0
            # + (1-T)*s*e/(1-e)*(Y-mu0)
            # -------------------------------------------------

            contributions[
                name
            ][kept_idx] = (

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
    # Common overlap-restricted evaluation sample
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

    # ---------------------------------------------------------
    # Policy risk
    # ---------------------------------------------------------

    values = {
        name: float(
            np.mean(
                score[
                    used
                ]
            )
        )
        for (
            name,
            score,
        )
        in contributions.items()
    }

    retained_fraction = float(
        used.mean()
    )

    # ---------------------------------------------------------
    # Optional observation-level output
    # ---------------------------------------------------------

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

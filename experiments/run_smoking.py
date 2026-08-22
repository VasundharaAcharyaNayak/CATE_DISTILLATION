from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from src.config import load_config
from src.data import (
    load_observational_data,
    prepare_smoking_cohort,
)
from src.smoking_preprocessing import (
    prepare_smoking_representations,
)
from src.policy_utils import (
    individual_policy_allocation,
)
from src.kmeans_policy import (
    fit_kmeans_policy_groups,
    estimate_oof_cluster_benefits,
    rank_clusters_by_gain,
    build_cluster_shift_probability,
)
from src.causal_tree_policy import (
    fit_causal_tree_subgroup_policy,
)
from src.cf_distillation import (
    fit_cf_distillation,
)
from src.bcf_distillation import (
    fit_bcf_distillation,
)
from src.inference import (
    bootstrap_policy_evaluation,
    paired_utility_comparisons,
)
from src.fidelity import (
    evaluate_teacher_student_fidelity,
)


# =============================================================
# Paths
# =============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "config"
    / "smoking.yaml"
)


# =============================================================
# Eight prespecified comparisons
#
# Delta U = U(A) - U(B)
#
# The runner reports paired bootstrap CIs only.
# P-values / Holm are handled separately.
# =============================================================

PRESPECIFIED_COMPARISONS = [

    ("K-means", "Causal Tree"),

    ("K-means", "CF teacher"),

    ("K-means", "BCF teacher"),

    ("CF teacher", "CF DT student"),

    ("CF teacher", "CF RF student"),

    ("BCF teacher", "BCF DT student"),

    ("BCF teacher", "BCF RF student"),

    ("CF teacher", "BCF teacher"),
]


# =============================================================
# Helpers
# =============================================================

def _resolve_data_path(config):

    config = dict(
        config
    )

    config[
        "experiment"
    ] = dict(
        config[
            "experiment"
        ]
    )

    data_path = Path(
        config[
            "experiment"
        ][
            "data_path"
        ]
    )

    if not data_path.is_absolute():

        data_path = (
            PROJECT_ROOT
            / data_path
        )

    config[
        "experiment"
    ][
        "data_path"
    ] = str(
        data_path
    )

    return config


def _extract_evaluation_score(
    result,
    *candidate_keys,
):

    for key in candidate_keys:

        if key in result:

            score = np.asarray(
                result[
                    key
                ],
                dtype=float,
            ).reshape(-1)

            if not np.all(
                np.isfinite(
                    score
                )
            ):

                raise ValueError(
                    f"{key!r} contains "
                    "non-finite evaluation scores."
                )

            return score

    raise KeyError(
        "None of the expected prediction keys "
        "were found: "
        + ", ".join(
            repr(
                key
            )
            for key
            in candidate_keys
        )
    )


def _build_individual_policy(
    score,
    treatment,
    policy_config,
):
   

    return individual_policy_allocation(

        benefit_scores=
            score,

        treatment=
            treatment,

        budget_fraction=
            float(
                policy_config[
                    "budget_fraction"
                ]
            ),

        positive_benefit_only=
            bool(
                policy_config[
                    "positive_benefit_only"
                ]
            ),

        min_benefit=
            float(
                policy_config.get(
                    "min_benefit",
                    0.0,
                )
            ),
    )


def _check_default_cohort_counts(
    analysis_df,
    discovery_df,
    evaluation_df,
    treatment_evaluation,
    config,
    split_seed,
):

    expected = (
        config
        .get(
            "split",
            {},
        )
        .get(
            "expected_sizes",
            {},
        )
    )

    default_seed = int(
        config[
            "split"
        ][
            "seed"
        ]
    )

    if (
        not expected
        or int(
            split_seed
        ) != default_seed
    ):
        return

    observed = {

        "analytic_n":
            len(
                analysis_df
            ),

        "discovery_n":
            len(
                discovery_df
            ),

        "evaluation_n":
            len(
                evaluation_df
            ),

        "evaluation_eligible_n":
            int(
                (
                    treatment_evaluation
                    == 1
                ).sum()
            ),
    }

    mismatches = {

        key:
            (
                observed[
                    key
                ],
                int(
                    expected[
                        key
                    ]
                ),
            )

        for key
        in observed

        if (
            key in expected
            and observed[
                key
            ]
            != int(
                expected[
                    key
                ]
            )
        )
    }

    if mismatches:

        details = ", ".join(

            f"{key}: "
            f"observed={observed_value}, "
            f"expected={expected_value}"

            for (
                key,
                (
                    observed_value,
                    expected_value,
                ),
            )
            in mismatches.items()
        )

        raise RuntimeError(
            "Smoking cohort does not match the "
            "configured representative split: "
            + details
        )


# =============================================================
# Main smoking experiment
# =============================================================

def run_smoking(
    config_path=DEFAULT_CONFIG,
    *,
    split_seed=None,
    run_inference=True,
    verbose=True,
):
    

    # =========================================================
    # 1. Configuration and analytic cohort
    # =========================================================

    config = _resolve_data_path(
        load_config(
            config_path
        )
    )

    raw_df = (
        load_observational_data(
            config
        )
    )

    analysis_df = (
        prepare_smoking_cohort(
            raw_df,
            config,
        )
    )

    # =========================================================
    # 2. Discovery / evaluation split
    #
    # IMPORTANT:
    # The final smoking analysis uses:
    #
    #     test_size = 0.40
    #     random_state = 70
    #     stratify = None
    #
    # unlike the glucose experiment.
    # =========================================================

    if split_seed is None:

        split_seed = int(
            config[
                "split"
            ][
                "seed"
            ]
        )

    (
        discovery_df,
        evaluation_df,
    ) = train_test_split(

        analysis_df,

        test_size=
            float(
                config[
                    "split"
                ][
                    "evaluation_fraction"
                ]
            ),

        random_state=
            int(
                split_seed
            ),

        stratify=None,
    )

    discovery_df = (
        discovery_df
        .reset_index(
            drop=True
        )
        .copy()
    )

    evaluation_df = (
        evaluation_df
        .reset_index(
            drop=True
        )
        .copy()
    )

    # =========================================================
    # 3. Variable definitions
    # =========================================================

    confounder_cols = list(
        config[
            "covariates"
        ][
            "adjustment"
        ]
    )

    effect_modifier_cols = list(
        config[
            "covariates"
        ][
            "effect_modifiers"
        ]
    )

    treatment_col = (
        config[
            "experiment"
        ][
            "treatment"
        ][
            "column"
        ]
    )

    outcome_col = (
        config[
            "experiment"
        ][
            "outcome"
        ][
            "column"
        ]
    )

    # =========================================================
    # 4. Frozen smoking representations
    #
    # Fit preprocessing on DISCOVERY only.
    #
    # One-hot effect modifiers:
    #     CF / BCF / students
    #
    # MCA:
    #     K-means / Causal Tree
    #
    # One-hot confounders:
    #     selective-shift DR evaluator
    # =========================================================

    representations = (
        prepare_smoking_representations(

            discovery_df=
                discovery_df,

            evaluation_df=
                evaluation_df,

            effect_modifier_cols=effect_modifier_columns,

            confounder_columns=
                confounder_cols,

            n_mca_components=
                int(
                    config[
                        "preprocessing"
                    ][
                        "mca"
                    ][
                        "n_components"
                    ]
                ),

            mca_seed=
                int(
                    config[
                        "preprocessing"
                    ][
                        "mca"
                    ][
                        "seed"
                    ]
                ),
        )
    )

    # ---------------------------------------------------------
    # One-hot original effect modifiers
    #
    # These are the final CF / BCF / student inputs.
    # ---------------------------------------------------------

    X_discovery = np.asarray(
        representations[
            "effect_onehot_discovery"
        ],
        dtype=float,
    )

    X_evaluation = np.asarray(
        representations[
            "effect_onehot_evaluation"
        ],
        dtype=float,
    )

    # ---------------------------------------------------------
    # 7-component MCA coordinates
    #
    # Used ONLY for subgroup models.
    # ---------------------------------------------------------

    X_mca_discovery = np.asarray(
        representations[
            "mca_discovery"
        ],
        dtype=float,
    )

    X_mca_evaluation = np.asarray(
        representations[
            "mca_evaluation"
        ],
        dtype=float,
    )

    # ---------------------------------------------------------
    # One-hot confounder representation
    #
    # Used for discovery subgroup-benefit estimation and
    # held-out selective-shift DR evaluation.
    # ---------------------------------------------------------

    Z_discovery = np.asarray(
        representations[
            "confounder_discovery"
        ],
        dtype=float,
    )

    Z_evaluation = np.asarray(
        representations[
            "confounder_evaluation"
        ],
        dtype=float,
    )

    # ---------------------------------------------------------
    # Treatment and outcome
    # ---------------------------------------------------------

    T_discovery = (
        discovery_df[
            treatment_col
        ]
        .to_numpy(
            dtype=int
        )
    )

    T_evaluation = (
        evaluation_df[
            treatment_col
        ]
        .to_numpy(
            dtype=int
        )
    )

    Y_discovery = (
        discovery_df[
            outcome_col
        ]
        .to_numpy(
            dtype=float
        )
    )

    Y_evaluation = (
        evaluation_df[
            outcome_col
        ]
        .to_numpy(
            dtype=float
        )
    )

    # =========================================================
    # 5. Alignment checks
    # =========================================================

    if not (
        len(
            X_discovery
        )
        == len(
            X_mca_discovery
        )
        == len(
            Z_discovery
        )
        == len(
            T_discovery
        )
        == len(
            Y_discovery
        )
    ):

        raise RuntimeError(
            "Smoking discovery representations "
            "are not row-aligned."
        )

    if not (
        len(
            X_evaluation
        )
        == len(
            X_mca_evaluation
        )
        == len(
            Z_evaluation
        )
        == len(
            T_evaluation
        )
        == len(
            Y_evaluation
        )
    ):

        raise RuntimeError(
            "Smoking evaluation representations "
            "are not row-aligned."
        )

    if X_mca_discovery.shape[1] != int(
        config[
            "preprocessing"
        ][
            "mca"
        ][
            "n_components"
        ]
    ):

        raise RuntimeError(
            "Unexpected number of smoking MCA dimensions."
        )

    for name, matrix in {

        "X_discovery":
            X_discovery,

        "X_evaluation":
            X_evaluation,

        "X_mca_discovery":
            X_mca_discovery,

        "X_mca_evaluation":
            X_mca_evaluation,

        "Z_discovery":
            Z_discovery,

        "Z_evaluation":
            Z_evaluation,

    }.items():

        if not np.isfinite(
            matrix
        ).all():

            raise ValueError(
                f"{name} contains non-finite values."
            )

    # =========================================================
    # 6. Representative-cohort check
    # =========================================================

    _check_default_cohort_counts(

        analysis_df=
            analysis_df,

        discovery_df=
            discovery_df,

        evaluation_df=
            evaluation_df,

        treatment_evaluation=
            T_evaluation,

        config=
            config,

        split_seed=
            split_seed,
    )

    q = float(
        config[
            "policy"
        ][
            "budget_fraction"
        ]
    )

    min_benefit = float(
        config[
            "policy"
        ].get(
            "min_benefit",
            0.0,
        )
    )

    # =========================================================
    # 7. K-means subgroup policy
    #
    # IMPORTANT:
    # K-means is fit directly in the MCA space.
    #
    # No additional StandardScaler is applied here.
    # =========================================================

    (
        kmeans_group_discovery,
        kmeans_group_evaluation,
        kmeans_model,
        kmeans_scaler,
    ) = fit_kmeans_policy_groups(

        X_discovery=
            X_mca_discovery,

        X_evaluation=
            X_mca_evaluation,

        n_clusters=
            int(
                config[
                    "kmeans"
                ][
                    "selected_k"
                ]
            ),

        standardize=False,

        n_init=
            int(
                config[
                    "kmeans"
                ][
                    "n_init"
                ]
            ),

        seed=
            int(
                config[
                    "kmeans"
                ][
                    "seed"
                ]
            ),
    )

    (
        kmeans_cluster_summary,
        kmeans_oof_benefit,
    ) = estimate_oof_cluster_benefits(

        Z=
            Z_discovery,

        T=
            T_discovery,

        Y=
            Y_discovery,

        groups=
            kmeans_group_discovery,

        n_splits=
            int(
                config[
                    "evaluation"
                ][
                    "dr_folds"
                ]
            ),

        seed=
            int(
                config[
                    "causal_forest"
                ][
                    "seed"
                ]
            ),
    )

    (
        kmeans_ranked_clusters,
        kmeans_cluster_order,
    ) = rank_clusters_by_gain(

        kmeans_cluster_summary,

        positive_gain_only=
            bool(
                config[
                    "kmeans"
                ][
                    "positive_gain_only"
                ]
            ),
    )

    (
        kmeans_policy,
        kmeans_allocation,
        kmeans_metadata,
    ) = build_cluster_shift_probability(

        evaluation_groups=
            kmeans_group_evaluation,

        treatment=
            T_evaluation,

        cluster_order=
            kmeans_cluster_order,

        budget_fraction=
            q,
    )

    # =========================================================
    # 8. Matched Causal Tree
    #
    # Also uses MCA representation.
    #
    # Maximum leaves = K-means K = 6.
    # =========================================================

    causal_tree = (
        fit_causal_tree_subgroup_policy(

            X_discovery=
                X_mca_discovery,

            Z_discovery=
                Z_discovery,

            T_discovery=
                T_discovery,

            Y_discovery=
                Y_discovery,

            X_evaluation=
                X_mca_evaluation,

            T_evaluation=
                T_evaluation,

            k_groups=
                int(
                    config[
                        "causal_tree"
                    ][
                        "max_leaf_nodes"
                    ]
                ),

            q=
                q,

            honesty_fraction=
                float(
                    config[
                        "causal_tree"
                    ][
                        "honest_split_fraction"
                    ]
                ),

            seed=
                int(
                    config[
                        "causal_tree"
                    ][
                        "seed"
                    ]
                ),

            n_splits=
                int(
                    config[
                        "causal_tree"
                    ][
                        "aipw_folds"
                    ]
                ),

            min_gain=
                min_benefit,

            preferred_min_group_samples=
                int(
                    config[
                        "causal_tree"
                    ][
                        "preferred_min_group_samples"
                    ]
                ),

            preferred_min_samples_leaf=
                int(
                    config[
                        "causal_tree"
                    ][
                        "preferred_min_samples_leaf"
                    ]
                ),

            trim=
                float(
                    config[
                        "evaluation"
                    ][
                        "overlap_trim"
                    ]
                ),
        )
    )

    causal_tree_policy = np.asarray(
        causal_tree[
            "shift_probability_evaluation"
        ],
        dtype=float,
    ).reshape(-1)

    # =========================================================
    # 9. Causal Forest teacher + students
    #
    # IMPORTANT:
    #
    # Final smoking CF:
    #     X = discovery-fitted one-hot effect modifiers
    #     W = None
    #
    # Adjustment variables are contained within X.
    # =========================================================

    cf = fit_cf_distillation(

        X_discovery=
            X_discovery,

        T_discovery=
            T_discovery,

        Y_discovery=
            Y_discovery,

        X_evaluation=
            X_evaluation,

        teacher_params=
            config[
                "causal_forest"
            ][
                "selected_params"
            ],

        dt_params=
            config[
                "students"
            ][
                "decision_tree"
            ],

        rf_params=
            config[
                "students"
            ][
                "random_forest"
            ],

        W_discovery=None,

        outcome_type=
            "binary",

        teacher_oof_folds=
            int(
                config[
                    "causal_forest"
                ][
                    "oof_teacher_folds"
                ]
            ),

        internal_cv=3,

        seed=
            int(
                config[
                    "causal_forest"
                ][
                    "seed"
                ]
            ),
    )

    # =========================================================
    # 10. BCF teacher + students
    #
    # Same one-hot effect-modifier representation as CF.
    # BCF internally fits the external propensity model.
    # =========================================================

    bcf = fit_bcf_distillation(

        X_discovery=
            X_discovery,

        T_discovery=
            T_discovery,

        Y_discovery=
            Y_discovery,

        X_evaluation=
            X_evaluation,

        T_evaluation=
            T_evaluation,

        dt_params=
            config[
                "students"
            ][
                "decision_tree"
            ],

        rf_params=
            config[
                "students"
            ][
                "random_forest"
            ],

        outcome_type=
            "binary",

        seed=
            int(
                config[
                    "students"
                ][
                    "seed"
                ]
            ),

        oof_folds=
            int(
                config[
                    "bcf"
                ][
                    "oof_folds"
                ]
            ),

        num_gfr=
            int(
                config[
                    "bcf"
                ][
                    "num_gfr"
                ]
            ),

        num_mcmc=
            int(
                config[
                    "bcf"
                ][
                    "num_mcmc"
                ]
            ),

        general_params=
            config[
                "bcf"
            ][
                "general"
            ],

        prognostic_forest_params=
            config[
                "bcf"
            ][
                "prognostic_forest"
            ],

        treatment_effect_forest_params=
            config[
                "bcf"
            ][
                "treatment_effect_forest"
            ],
    )

    # =========================================================
    # 11. Evaluation benefit scores
    # =========================================================

    score_vectors = {

        "CF teacher":
            _extract_evaluation_score(
                cf,
                "teacher_cate_evaluation",
                "teacher_benefit_evaluation",
            ),

        "CF DT student":
            _extract_evaluation_score(
                cf,
                "dt_cate_evaluation",
                "dt_benefit_evaluation",
            ),

        "CF RF student":
            _extract_evaluation_score(
                cf,
                "rf_cate_evaluation",
                "rf_benefit_evaluation",
            ),

        "BCF teacher":
            _extract_evaluation_score(
                bcf,
                "teacher_cate_evaluation",
                "teacher_benefit_evaluation",
            ),

        "BCF DT student":
            _extract_evaluation_score(
                bcf,
                "dt_cate_evaluation",
                "dt_benefit_evaluation",
            ),

        "BCF RF student":
            _extract_evaluation_score(
                bcf,
                "rf_cate_evaluation",
                "rf_benefit_evaluation",
            ),
    }

    for name, score in score_vectors.items():

        if len(
            score
        ) != len(
            evaluation_df
        ):

            raise RuntimeError(
                f"{name} produced {len(score)} scores "
                f"for evaluation N={len(evaluation_df)}."
            )

    # =========================================================
    # 12. Primary individualized policies
    #
    # Exact-score implementation.
    #
    # Threshold extension belongs only to the sensitivity
    # analysis and is NOT used here.
    # =========================================================

    original_score_policies = {}

    score_policy_metadata = {}

    for (
        name,
        score,
    ) in score_vectors.items():

        (
            policy,
            metadata,
        ) = _build_individual_policy(

            score=
                score,

            treatment=
                T_evaluation,

            policy_config=
                config[
                    "policy"
                ],
        )

        original_score_policies[
            name
        ] = policy

        score_policy_metadata[
            name
        ] = metadata

    # =========================================================
    # 13. Learned policies
    # =========================================================

    learned_policies = {

        "K-means":
            np.asarray(
                kmeans_policy,
                dtype=float,
            ).reshape(-1),

        "Causal Tree":
            causal_tree_policy,

        **original_score_policies,
    }

    # =========================================================
    # 14. Reference + learned policies
    # =========================================================

    all_policies = {

        "No shift":
            np.zeros(
                len(
                    evaluation_df
                ),
                dtype=float,
            ),

        "Shift all eligible":
            np.ones(
                len(
                    evaluation_df
                ),
                dtype=float,
            ),

        **learned_policies,
    }

    # =========================================================
    # 15. Teacher-student fidelity
    # =========================================================

    half_width = float(
        config[
            "evaluation"
        ][
            "boundary_kendall"
        ][
            "teacher_quantile_half_width"
        ]
    )

    (
        cf_fidelity,
        cf_hard_policies,
    ) = evaluate_teacher_student_fidelity(

        teacher_score=
            score_vectors[
                "CF teacher"
            ],

        student_scores={

            "CF DT student":
                score_vectors[
                    "CF DT student"
                ],

            "CF RF student":
                score_vectors[
                    "CF RF student"
                ],
        },

        eligible_mask=
            (
                T_evaluation
                == 1
            ),

        q=
            q,

        min_benefit=
            min_benefit,

        boundary_bandwidth=
            2.0
            * half_width,

        seed=
            int(
                config[
                    "students"
                ][
                    "seed"
                ]
            ),
    )

    (
        bcf_fidelity,
        bcf_hard_policies,
    ) = evaluate_teacher_student_fidelity(

        teacher_score=
            score_vectors[
                "BCF teacher"
            ],

        student_scores={

            "BCF DT student":
                score_vectors[
                    "BCF DT student"
                ],

            "BCF RF student":
                score_vectors[
                    "BCF RF student"
                ],
        },

        eligible_mask=
            (
                T_evaluation
                == 1
            ),

        q=
            q,

        min_benefit=
            min_benefit,

        boundary_bandwidth=
            2.0
            * half_width,

        seed=
            int(
                config[
                    "students"
                ][
                    "seed"
                ]
            ),
    )

    # =========================================================
    # 16. Selective-shift DR evaluation + bootstrap CIs
    # =========================================================

    policy_results = None
    bootstrap_draws = None
    point_risks = None
    paired_comparisons = None

    if run_inference:

        (
            policy_results,
            bootstrap_draws,
            point_risks,
        ) = bootstrap_policy_evaluation(

            Z=
                Z_evaluation,

            T=
                T_evaluation,

            Y=
                Y_evaluation,

            policies=
                all_policies,

            B=
                int(
                    config[
                        "evaluation"
                    ][
                        "bootstrap_replicates"
                    ]
                ),

            seed=
                int(
                    config[
                        "evaluation"
                    ][
                        "bootstrap_seed"
                    ]
                ),

            n_splits=
                int(
                    config[
                        "evaluation"
                    ][
                        "dr_folds"
                    ]
                ),

            trim=
                float(
                    config[
                        "evaluation"
                    ][
                        "overlap_trim"
                    ]
                ),

            crossfit_seed=
                int(
                    config[
                        "causal_forest"
                    ][
                        "seed"
                    ]
                ),
        )

        # -----------------------------------------------------
        # Paired bootstrap 95% CIs only.
        #
        # Sign-tail p-values and Holm correction remain in the
        # separate inferential analysis.
        # -----------------------------------------------------

        paired_comparisons = (
            paired_utility_comparisons(

                bootstrap_df=
                    bootstrap_draws,

                point_values=
                    point_risks,

                comparisons=
                    PRESPECIFIED_COMPARISONS,
            )
        )

    # =========================================================
    # 17. Console output
    # =========================================================

    if verbose:

        n_eligible = int(
            (
                T_evaluation
                == 1
            ).sum()
        )

        target_count = int(
            round(
                q
                * n_eligible
            )
        )

        print(
            "\n=== NHANES SMOKING-HISTORY COHORT ==="
        )

        print(
            "Analytic N:",
            len(
                analysis_df
            ),
        )

        print(
            "Discovery N:",
            len(
                discovery_df
            ),
        )

        print(
            "Evaluation N:",
            len(
                evaluation_df
            ),
        )

        print(
            "Evaluation eligible T=1:",
            n_eligible,
        )

        print(
            "Budget target:",
            target_count,
        )

        print(
            "One-hot effect-modifier dimension:",
            X_discovery.shape[1],
        )

        print(
            "MCA dimension:",
            X_mca_discovery.shape[1],
        )

        print(
            "DR adjustment dimension:",
            Z_discovery.shape[1],
        )

        if policy_results is not None:

            print(
                "\n=== HELD-OUT POLICY RESULTS ==="
            )

            print(
                policy_results
                .round(6)
                .to_string(
                    index=False
                )
            )

        if paired_comparisons is not None:

            print(
                "\n=== EIGHT PRESPECIFIED "
                "PAIRED UTILITY-DIFFERENCE CIs ==="
            )

            print(
                paired_comparisons
                .round(6)
                .to_string(
                    index=False
                )
            )

        print(
            "\n=== CF TEACHER-STUDENT FIDELITY ==="
        )

        print(
            cf_fidelity
            .round(6)
            .to_string(
                index=False
            )
        )

        print(
            "\n=== BCF TEACHER-STUDENT FIDELITY ==="
        )

        print(
            bcf_fidelity
            .round(6)
            .to_string(
                index=False
            )
        )

    # =========================================================
    # 18. Return objects
    # =========================================================

    return {

        "config":
            config,

        "split_seed":
            int(
                split_seed
            ),

        "analysis_df":
            analysis_df,

        "discovery_df":
            discovery_df,

        "evaluation_df":
            evaluation_df,

        # Teacher/student representation
        "X_discovery":
            X_discovery,

        "X_evaluation":
            X_evaluation,

        # MCA subgroup representation
        "X_mca_discovery":
            X_mca_discovery,

        "X_mca_evaluation":
            X_mca_evaluation,

        # DR adjustment representation
        "Z_discovery":
            Z_discovery,

        "Z_evaluation":
            Z_evaluation,

        "T_discovery":
            T_discovery,

        "T_evaluation":
            T_evaluation,

        "Y_discovery":
            Y_discovery,

        "Y_evaluation":
            Y_evaluation,

        "representations":
            representations,

        "cf":
            cf,

        "bcf":
            bcf,

        # Needed by score-extension sensitivity
        "score_vectors":
            score_vectors,

        "original_score_policies":
            original_score_policies,

        "score_policy_metadata":
            score_policy_metadata,

        "kmeans": {

            "model":
                kmeans_model,

            "scaler":
                kmeans_scaler,

            "group_discovery":
                kmeans_group_discovery,

            "group_evaluation":
                kmeans_group_evaluation,

            "oof_benefit":
                kmeans_oof_benefit,

            "cluster_summary":
                kmeans_cluster_summary,

            "ranked_clusters":
                kmeans_ranked_clusters,

            "cluster_order":
                kmeans_cluster_order,

            "allocation":
                kmeans_allocation,

            "metadata":
                kmeans_metadata,
        },

        "causal_tree":
            causal_tree,

        # Needed by nuisance-model sensitivity
        "learned_policies":
            learned_policies,

        "all_policies":
            all_policies,

        "policy_results":
            policy_results,

        "bootstrap_draws":
            bootstrap_draws,

        "point_risks":
            point_risks,

        "paired_comparisons":
            paired_comparisons,

        "cf_fidelity":
            cf_fidelity,

        "bcf_fidelity":
            bcf_fidelity,

        "cf_hard_policies":
            cf_hard_policies,

        "bcf_hard_policies":
            bcf_hard_policies,
    }


# =============================================================
# Standalone execution
# =============================================================

def main():

    run_smoking(

        config_path=
            DEFAULT_CONFIG,

        run_inference=True,

        verbose=True,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

from typing import Sequence

import pandas as pd
import prince


def _discovery_frozen_dummies(
    discovery_df: pd.DataFrame,
    evaluation_df: pd.DataFrame,
    columns: Sequence[str],
):
   

    discovery_cat = discovery_df[
        list(columns)
    ].copy()

    evaluation_cat = evaluation_df[
        list(columns)
    ].copy()

    for col in columns:

        discovery_cat[col] = (
            discovery_cat[col]
            .astype(str)
        )

        evaluation_cat[col] = (
            evaluation_cat[col]
            .astype(str)
        )

    discovery_oh = pd.get_dummies(
        discovery_cat,
        drop_first=False,
        dtype=float,
    )

    evaluation_oh = pd.get_dummies(
        evaluation_cat,
        drop_first=False,
        dtype=float,
    )

    # Freeze discovery representation
    evaluation_oh = evaluation_oh.reindex(
        columns=discovery_oh.columns,
        fill_value=0.0,
    )

    return (
        discovery_oh,
        evaluation_oh,
    )


def prepare_smoking_representations(
    discovery_df: pd.DataFrame,
    evaluation_df: pd.DataFrame,
    *,
    effect_modifier_cols: Sequence[str],
    confounder_cols: Sequence[str],
    n_mca_components=7,
    mca_seed=7,
):
    

    # =========================================================
    # 1. Effect modifiers: discovery-frozen one-hot encoding
    # =========================================================

    (
        effect_discovery_oh,
        effect_evaluation_oh,
    ) = _discovery_frozen_dummies(
        discovery_df,
        evaluation_df,
        effect_modifier_cols,
    )

    # =========================================================
    # 2. Fit MCA on discovery only
    # =========================================================

    mca = prince.MCA(
        n_components=int(
            n_mca_components
        ),
        n_iter=10,
        random_state=int(
            mca_seed
        ),

        # We already manually one-hot encoded.
        one_hot=False,
    )

    mca.fit(
        effect_discovery_oh
    )

    # =========================================================
    # 3. Frozen MCA projection
    # =========================================================

    mca_discovery = (
        mca.row_coordinates(
            effect_discovery_oh
        )
    )

    mca_evaluation = (
        mca.row_coordinates(
            effect_evaluation_oh
        )
    )

    mca_columns = [
        f"MCA_{j + 1}"
        for j in range(
            mca_discovery.shape[1]
        )
    ]

    mca_discovery.columns = (
        mca_columns
    )

    mca_evaluation.columns = (
        mca_columns
    )

    # =========================================================
    # 4. Prespecified confounder representation
    # =========================================================

    (
        confounder_discovery_oh,
        confounder_evaluation_oh,
    ) = _discovery_frozen_dummies(
        discovery_df,
        evaluation_df,
        confounder_cols,
    )

    # =========================================================
    # 5. Return frozen representations
    # =========================================================

    return {

        # -----------------------------------------------------
        # Original effect-modifier one-hot representation
        #
        # Current CF / BCF / DT / RF smoking branch
        # -----------------------------------------------------

        "effect_onehot_discovery":
            effect_discovery_oh
            .to_numpy(
                dtype=float
            ),

        "effect_onehot_evaluation":
            effect_evaluation_oh
            .to_numpy(
                dtype=float
            ),

        "effect_onehot_columns":
            list(
                effect_discovery_oh.columns
            ),

        # -----------------------------------------------------
        # MCA representation
        #
        # K-means / matched causal-tree subgroup branch
        # -----------------------------------------------------

        "mca_discovery":
            mca_discovery
            .to_numpy(
                dtype=float
            ),

        "mca_evaluation":
            mca_evaluation
            .to_numpy(
                dtype=float
            ),

        "mca_columns":
            mca_columns,

        "mca_model":
            mca,

        # -----------------------------------------------------
        # Causal adjustment representation
        # -----------------------------------------------------

        "confounder_discovery":
            confounder_discovery_oh
            .to_numpy(
                dtype=float
            ),

        "confounder_evaluation":
            confounder_evaluation_oh
            .to_numpy(
                dtype=float
            ),

        "confounder_columns":
            list(
                confounder_discovery_oh.columns
            ),
    }

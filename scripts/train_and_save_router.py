import json
from pathlib import Path

import joblib
import pandas as pd

from app.classifier.success_router import (
    build_feature_transformer,
    build_success_model,
    build_binary_target,
    TIER_SCORE_COLUMNS,
    QUALITY_THRESHOLD,
)


DATA_PATH = Path(
    "data/classifier/sprout/routing_train.csv"
)


ARTIFACT_DIR = Path(
    "artifacts/router"
)


def load_training_data():

    df = pd.read_csv(
        DATA_PATH
    )

    df = df.dropna(
        subset=[
            "prompt",
            "tier_1_score",
            "tier_2_score",
            "tier_3_score",
        ]
    ).copy()

    return df


def main():

    print(
        "Loading training data..."
    )

    train_df = (
        load_training_data()
    )

    print(
        f"Training rows: "
        f"{len(train_df)}"
    )


    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # =========================================
    # Train shared feature transformer
    # =========================================

    print(
        "\nTraining feature transformer..."
    )


    feature_transformer = (
        build_feature_transformer()
    )


    X_train = (
        feature_transformer.fit_transform(
            train_df["prompt"]
        )
    )


    print(
        f"Feature matrix: "
        f"{X_train.shape}"
    )


    # =========================================
    # Save transformer
    # =========================================

    transformer_path = (
        ARTIFACT_DIR
        / "feature_transformer.joblib"
    )


    joblib.dump(
        feature_transformer,
        transformer_path,
    )


    print(
        f"Saved: {transformer_path}"
    )


    # =========================================
    # Train + save each predictor
    # =========================================

    for (
        tier,
        score_column
    ) in TIER_SCORE_COLUMNS.items():

        print(
            f"\nTraining {tier} "
            f"success predictor..."
        )


        y_train = (
            build_binary_target(
                train_df,
                score_column,
            )
        )


        model = (
            build_success_model()
        )


        model.fit(
            X_train,
            y_train,
        )


        model_path = (
            ARTIFACT_DIR
            / f"{tier}_predictor.joblib"
        )


        joblib.dump(
            model,
            model_path,
        )


        print(
            f"Saved: {model_path}"
        )


    # =========================================
    # Router policy configuration
    # =========================================

    config = {

        "quality_threshold":
            QUALITY_THRESHOLD,

        "tier_1_threshold":
            0.80,

        "tier_2_threshold":
            0.70,

        "fallback_tier":
            "tier_3",

        "routing_strategy":
            "minimum_predicted_capability",

        "feature_type":
            "tfidf_plus_handcrafted",
    }


    config_path = (
        ARTIFACT_DIR
        / "router_config.json"
    )


    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            config,
            file,
            indent=4,
        )


    print(
        f"\nSaved: {config_path}"
    )


    print(
        "\nRouter training complete."
    )


if __name__ == "__main__":
    main()
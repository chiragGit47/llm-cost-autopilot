from pathlib import Path

import pandas as pd

from sklearn.pipeline import (
    Pipeline,
    FeatureUnion,
)

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)

from sklearn.preprocessing import (
    StandardScaler,
)

from sklearn.linear_model import (
    LogisticRegression,
)

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)

from app.classifier.transformers import (
    HandcraftedFeatureTransformer,
)


DATA_DIR = Path(
    "data/classifier/sprout"
)

TRAIN_PATH = (
    DATA_DIR / "routing_train.csv"
)

VALIDATION_PATH = (
    DATA_DIR / "routing_validation.csv"
)


QUALITY_THRESHOLD = 0.8


TIER_COLUMNS = {
    "tier_1": "tier_1_score",
    "tier_2": "tier_2_score",
    "tier_3": "tier_3_score",
}


def load_data(path):

    df = pd.read_csv(path)

    df = df.dropna(
        subset=[
            "prompt",
            "tier_1_score",
            "tier_2_score",
            "tier_3_score",
        ]
    ).copy()

    return df


def build_feature_transformer():

    handcrafted_pipeline = Pipeline([
        (
            "extract",
            HandcraftedFeatureTransformer(),
        ),
        (
            "scale",
            StandardScaler(),
        ),
    ])


    return FeatureUnion([

        (
            "tfidf",

            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                max_features=40000,
                sublinear_tf=True,
            ),
        ),

        (
            "handcrafted",
            handcrafted_pipeline,
        ),

    ])


def create_binary_target(
    df,
    score_column,
):

    return (
        df[score_column]
        >= QUALITY_THRESHOLD
    ).astype(int)


def evaluate_predictor(
    tier_name,
    model,
    X_validation,
    y_validation,
):

    predictions = model.predict(
        X_validation
    )

    probabilities = (
        model.predict_proba(
            X_validation
        )[:, 1]
    )


    accuracy = accuracy_score(
        y_validation,
        predictions,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_validation,
            predictions,
        )
    )

    f1 = f1_score(
        y_validation,
        predictions,
    )

    roc_auc = roc_auc_score(
        y_validation,
        probabilities,
    )


    print("\n")
    print("=" * 70)
    print(
        f"{tier_name.upper()} SUCCESS PREDICTOR"
    )
    print("=" * 70)


    print(
        f"Accuracy: "
        f"{accuracy:.4f}"
    )

    print(
        f"Balanced Accuracy: "
        f"{balanced_accuracy:.4f}"
    )

    print(
        f"F1: "
        f"{f1:.4f}"
    )

    print(
        f"ROC AUC: "
        f"{roc_auc:.4f}"
    )


    print("\nCLASSIFICATION REPORT")

    print(
        classification_report(
            y_validation,
            predictions,
            target_names=[
                "fail",
                "pass",
            ],
            zero_division=0,
        )
    )


    return probabilities


def main():

    train_df = load_data(
        TRAIN_PATH
    )

    validation_df = load_data(
        VALIDATION_PATH
    )


    print(
        f"Training rows: "
        f"{len(train_df)}"
    )

    print(
        f"Validation rows: "
        f"{len(validation_df)}"
    )


    # ==================================================
    # Fit feature extraction ON TRAINING DATA ONLY
    # ==================================================

    feature_transformer = (
        build_feature_transformer()
    )


    print(
        "\nBuilding TF-IDF features..."
    )


    X_train = (
        feature_transformer.fit_transform(
            train_df["prompt"]
        )
    )


    X_validation = (
        feature_transformer.transform(
            validation_df["prompt"]
        )
    )


    print(
        "Feature extraction complete."
    )


    print(
        f"\nFeature matrix:"
        f" {X_train.shape}"
    )


    models = {}

    validation_probabilities = {}


    # ==================================================
    # Train one binary model per tier
    # ==================================================

    for (
        tier_name,
        score_column
    ) in TIER_COLUMNS.items():


        print("\n")
        print("#" * 70)

        print(
            f"Training {tier_name}"
        )

        print("#" * 70)


        y_train = create_binary_target(
            train_df,
            score_column,
        )


        y_validation = create_binary_target(
            validation_df,
            score_column,
        )


        print(
            "\nTraining target:"
        )

        print(
            y_train.value_counts(
                normalize=True
            )
        )


        model = LogisticRegression(
            max_iter=2000,
            solver="saga",
            class_weight="balanced",
            random_state=42,
        )


        model.fit(
            X_train,
            y_train,
        )


        probabilities = (
            evaluate_predictor(
                tier_name,
                model,
                X_validation,
                y_validation,
            )
        )


        models[tier_name] = model

        validation_probabilities[
            tier_name
        ] = probabilities


if __name__ == "__main__":
    main()
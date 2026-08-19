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

from app.classifier.transformers import (
    HandcraftedFeatureTransformer,
)

from sklearn.calibration import (
    CalibratedClassifierCV,
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


TIER_SCORE_COLUMNS = {
    "tier_1": "tier_1_score",
    "tier_2": "tier_2_score",
    "tier_3": "tier_3_score",
}


TIER_ORDER = {
    "tier_1": 1,
    "tier_2": 2,
    "tier_3": 3,
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


def build_binary_target(
    df,
    score_column,
):

    return (
        df[score_column]
        >= QUALITY_THRESHOLD
    ).astype(int)


def get_oracle_tier(row):

    """
    Smallest ACTUAL model that passes
    the quality threshold.
    """

    if (
        row["tier_1_score"]
        >= QUALITY_THRESHOLD
    ):
        return "tier_1"

    if (
        row["tier_2_score"]
        >= QUALITY_THRESHOLD
    ):
        return "tier_2"

    if (
        row["tier_3_score"]
        >= QUALITY_THRESHOLD
    ):
        return "tier_3"

    return "unresolved"


def route_prompt(
    row,
    threshold,
):

    """
    Choose the cheapest model whose
    predicted pass score reaches
    the routing threshold.
    """

    if (
        row["tier_1_probability"]
        >= threshold
    ):
        return "tier_1"

    if (
        row["tier_2_probability"]
        >= threshold
    ):
        return "tier_2"

    if (
        row["tier_3_probability"]
        >= threshold
    ):
        return "tier_3"

    # Safety fallback:
    # if nothing is trusted,
    # use strongest model.
    return "tier_3"


def selected_model_succeeded(row):

    selected_tier = row["selected_tier"]

    score_column = (
        TIER_SCORE_COLUMNS[
            selected_tier
        ]
    )

    return (
        row[score_column]
        >= QUALITY_THRESHOLD
    )


def evaluate_threshold(
    df,
    threshold,
):

    experiment = df.copy()


    experiment[
        "selected_tier"
    ] = experiment.apply(

        lambda row: route_prompt(
            row,
            threshold,
        ),

        axis=1,
    )


    experiment[
        "success"
    ] = experiment.apply(
        selected_model_succeeded,
        axis=1,
    )


    experiment[
        "oracle_tier"
    ] = experiment.apply(
        get_oracle_tier,
        axis=1,
    )


    solvable = experiment[
        experiment["oracle_tier"]
        != "unresolved"
    ].copy()


    # ----------------------------------
    # Success metrics
    # ----------------------------------

    overall_success = (
        experiment["success"].mean()
    )

    solvable_success = (
        solvable["success"].mean()
    )


    # ----------------------------------
    # Routing metrics
    # ----------------------------------

    exact_match = (
        solvable["selected_tier"]
        ==
        solvable["oracle_tier"]
    ).mean()


    selected_number = (
        solvable["selected_tier"]
        .map(TIER_ORDER)
    )

    oracle_number = (
        solvable["oracle_tier"]
        .map(TIER_ORDER)
    )


    under_routing = (
        selected_number
        <
        oracle_number
    ).mean()


    over_routing = (
        selected_number
        >
        oracle_number
    ).mean()


    average_selected_tier = (
        experiment["selected_tier"]
        .map(TIER_ORDER)
        .mean()
    )


    tier_distribution = (
        experiment[
            "selected_tier"
        ]
        .value_counts(
            normalize=True
        )
    )


    print("\n")
    print("=" * 70)

    print(
        f"ROUTING THRESHOLD = "
        f"{threshold:.2f}"
    )

    print("=" * 70)


    print(
        f"Overall success rate: "
        f"{overall_success:.2%}"
    )

    print(
        f"Success on solvable prompts: "
        f"{solvable_success:.2%}"
    )

    print(
        f"Exact oracle match: "
        f"{exact_match:.2%}"
    )

    print(
        f"Under-routing rate: "
        f"{under_routing:.2%}"
    )

    print(
        f"Over-routing rate: "
        f"{over_routing:.2%}"
    )

    print(
        f"Average selected tier: "
        f"{average_selected_tier:.3f}"
    )


    print("\nTier usage:")

    for tier in [
        "tier_1",
        "tier_2",
        "tier_3",
    ]:

        usage = (
            tier_distribution.get(
                tier,
                0
            )
        )

        print(
            f"{tier}: "
            f"{usage:.2%}"
        )


def print_baselines(df):

    print("\n")
    print("=" * 70)
    print("BASELINES")
    print("=" * 70)


    for tier, score_column in (
        TIER_SCORE_COLUMNS.items()
    ):

        success = (
            df[score_column]
            >= QUALITY_THRESHOLD
        ).mean()

        print(
            f"Always {tier}: "
            f"{success:.2%} success"
        )


    oracle = df.apply(
        get_oracle_tier,
        axis=1,
    )


    oracle_success = (
        oracle != "unresolved"
    ).mean()


    print(
        f"\nOracle maximum success: "
        f"{oracle_success:.2%}"
    )


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
    # Shared feature extraction
    # ==================================================

    feature_transformer = (
        build_feature_transformer()
    )


    print(
        "\nBuilding features..."
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


    # ==================================================
    # Train per-tier predictors
    # ==================================================

    for (
        tier,
        score_column
    ) in TIER_SCORE_COLUMNS.items():


        print(
            f"\nTraining {tier} predictor..."
        )


        y_train = build_binary_target(
            train_df,
            score_column,
        )


        base_model = LogisticRegression(
            max_iter=2000,
            solver="saga",
            class_weight="balanced",
            random_state=42,
        )


        model = CalibratedClassifierCV(
            estimator=base_model,
            method="sigmoid",
            cv=3,
        )


        model.fit(
            X_train,
            y_train,
        )

        model.predict_proba(
            X_validation
            )[:, 1]


        probabilities = (
            model.predict_proba(
                X_validation
            )[:, 1]
        )


        validation_df[
            f"{tier}_probability"
        ] = probabilities


    # ==================================================
    # Baselines
    # ==================================================

    print_baselines(
        validation_df
    )


    # ==================================================
    # Threshold experiments
    # ==================================================

    thresholds = [
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
    ]


    for threshold in thresholds:

        evaluate_threshold(
            validation_df,
            threshold,
        )


if __name__ == "__main__":
    main()
from pathlib import Path

import pandas as pd

from app.classifier.success_router import (
    build_feature_transformer,
    build_success_model,
    build_binary_target,
    TIER_SCORE_COLUMNS,
    QUALITY_THRESHOLD,
)

DATA_DIR = Path(
    "data/classifier/sprout"
)

TRAIN_PATH = (
    DATA_DIR / "routing_train.csv"
)

TEST_PATH = (
    DATA_DIR / "routing_test.csv"
)


TIER_1_THRESHOLD = 0.80
TIER_2_THRESHOLD = 0.70

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

def route_prompt(row):

    if (
        row["tier_1_probability"]
        >= TIER_1_THRESHOLD
    ):
        return "tier_1"

    if (
        row["tier_2_probability"]
        >= TIER_2_THRESHOLD
    ):
        return "tier_2"

    return "tier_3"

def get_oracle_tier(row):

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

def selected_model_succeeded(row):

    selected = row[
        "selected_tier"
    ]

    score_column = (
        TIER_SCORE_COLUMNS[
            selected
        ]
    )

    return (
        row[score_column]
        >= QUALITY_THRESHOLD
    )

def main():

    train_df = load_data(
        TRAIN_PATH
    )

    test_df = load_data(
        TEST_PATH
    )


    print(
        f"Training rows: "
        f"{len(train_df)}"
    )

    print(
        f"Test rows: "
        f"{len(test_df)}"
    )


    # =========================================
    # Shared features
    # =========================================

    features = (
        build_feature_transformer()
    )


    print(
        "\nBuilding features..."
    )


    X_train = (
        features.fit_transform(
            train_df["prompt"]
        )
    )


    X_test = (
        features.transform(
            test_df["prompt"]
        )
    )


    print(
        "Feature extraction complete."
    )


    # =========================================
    # Train predictors
    # =========================================

    for (
        tier,
        score_column
    ) in TIER_SCORE_COLUMNS.items():


        print(
            f"Training {tier}..."
        )


        y_train = build_binary_target(
            train_df,
            score_column,
        )


        model = (
            build_success_model()
        )


        model.fit(
            X_train,
            y_train,
        )


        probabilities = (
            model.predict_proba(
                X_test
            )[:, 1]
        )


        test_df[
            f"{tier}_probability"
        ] = probabilities

    test_df[
        "selected_tier"
    ] = test_df.apply(
        route_prompt,
        axis=1,
    )


    test_df[
        "success"
    ] = test_df.apply(
        selected_model_succeeded,
        axis=1,
    )


    test_df[
        "oracle_tier"
    ] = test_df.apply(
        get_oracle_tier,
        axis=1,
    )

    solvable = test_df[
        test_df["oracle_tier"]
        != "unresolved"
    ].copy()


    overall_success = (
        test_df[
            "success"
        ].mean()
    )


    solvable_success = (
        solvable[
            "success"
        ].mean()
    )

    tier_order = {
        "tier_1": 1,
        "tier_2": 2,
        "tier_3": 3,
    }


    selected_numbers = (
        solvable[
            "selected_tier"
        ].map(
            tier_order
        )
    )


    oracle_numbers = (
        solvable[
            "oracle_tier"
        ].map(
            tier_order
        )
    )


    under_routing = (
        selected_numbers
        <
        oracle_numbers
    ).mean()


    over_routing = (
        selected_numbers
        >
        oracle_numbers
    ).mean()

    average_tier = (
        test_df[
            "selected_tier"
        ]
        .map(
            tier_order
        )
        .mean()
    )


    tier_usage = (
        test_df[
            "selected_tier"
        ]
        .value_counts(
            normalize=True
        )
    )

    average_tier = (
        test_df[
            "selected_tier"
        ]
        .map(
            tier_order
        )
        .mean()
    )


    tier_usage = (
        test_df[
            "selected_tier"
        ]
        .value_counts(
            normalize=True
        )
    )

    print("\n")
    print("=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)


    print(
        f"Overall success: "
        f"{overall_success:.2%}"
    )

    print(
        f"Solvable success: "
        f"{solvable_success:.2%}"
    )

    print(
        f"Under-routing: "
        f"{under_routing:.2%}"
    )

    print(
        f"Over-routing: "
        f"{over_routing:.2%}"
    )

    print(
        f"Average selected tier: "
        f"{average_tier:.3f}"
    )


    print("\nTier usage:")


    for tier in [
        "tier_1",
        "tier_2",
        "tier_3",
    ]:

        print(
            f"{tier}: "
            f"{tier_usage.get(tier, 0):.2%}"
        )


if __name__ == "__main__":
    main()
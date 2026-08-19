from pathlib import Path
from itertools import product

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

VALIDATION_PATH = (
    DATA_DIR / "routing_validation.csv"
)


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


def train_predictors(
    train_df,
    validation_df,
):

    features = (
        build_feature_transformer()
    )


    print(
        "Building shared features..."
    )


    X_train = (
        features.fit_transform(
            train_df["prompt"]
        )
    )


    X_validation = (
        features.transform(
            validation_df["prompt"]
        )
    )


    probabilities = {}


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


        model = build_success_model()


        model.fit(
            X_train,
            y_train,
        )


        probabilities[tier] = (
            model.predict_proba(
                X_validation
            )[:, 1]
        )


    return probabilities

def route_prompt(
    row,
    threshold_1,
    threshold_2,
    threshold_3,
):

    if (
        row["tier_1_probability"]
        >= threshold_1
    ):
        return "tier_1"


    if (
        row["tier_2_probability"]
        >= threshold_2
    ):
        return "tier_2"


    if (
        row["tier_3_probability"]
        >= threshold_3
    ):
        return "tier_3"


    # Safety fallback.
    return "tier_3"

def model_succeeded(
    row,
):

    selected = (
        row["selected_tier"]
    )


    score_column = (
        TIER_SCORE_COLUMNS[
            selected
        ]
    )


    return (
        row[score_column]
        >= QUALITY_THRESHOLD
    )

def get_oracle_tier(
    row,
):

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

def evaluate_configuration(
    df,
    threshold_1,
    threshold_2,
    threshold_3,
):

    experiment = df.copy()


    experiment[
        "selected_tier"
    ] = experiment.apply(

        lambda row: route_prompt(

            row,

            threshold_1,
            threshold_2,
            threshold_3,
        ),

        axis=1,
    )


    experiment[
        "success"
    ] = experiment.apply(
        model_succeeded,
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


    overall_success = (
        experiment["success"]
        .mean()
    )


    solvable_success = (
        solvable["success"]
        .mean()
    )


    selected_numbers = (
        solvable["selected_tier"]
        .map(TIER_ORDER)
    )


    oracle_numbers = (
        solvable["oracle_tier"]
        .map(TIER_ORDER)
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
        experiment["selected_tier"]
        .map(TIER_ORDER)
        .mean()
    )


    tier_usage = (
        experiment["selected_tier"]
        .value_counts(
            normalize=True
        )
    )


    return {

        "threshold_1":
            threshold_1,

        "threshold_2":
            threshold_2,

        "threshold_3":
            threshold_3,

        "overall_success":
            overall_success,

        "solvable_success":
            solvable_success,

        "under_routing":
            under_routing,

        "over_routing":
            over_routing,

        "average_tier":
            average_tier,

        "tier_1_usage":
            tier_usage.get(
                "tier_1",
                0
            ),

        "tier_2_usage":
            tier_usage.get(
                "tier_2",
                0
            ),

        "tier_3_usage":
            tier_usage.get(
                "tier_3",
                0
            ),
    }

def main():

    train_df = load_data(
        TRAIN_PATH
    )


    validation_df = load_data(
        VALIDATION_PATH
    )


    probabilities = train_predictors(
    train_df,
    validation_df,
)


    validation_df = add_probabilities(
    validation_df,
    probabilities,
)


    threshold_values = [
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
    ]


    results = []


    combinations = product(
        threshold_values,
        threshold_values,
        threshold_values,
    )


    for (
        t1,
        t2,
        t3
    ) in combinations:

        result = (
            evaluate_configuration(

                validation_df,

                threshold_1=t1,
                threshold_2=t2,
                threshold_3=t3,
            )
        )


        results.append(
            result
        )


    results_df = pd.DataFrame(
        results
    )

    acceptable = results_df[

        results_df[
            "solvable_success"
        ] >= 0.88

    ].copy()





    acceptable = (
        acceptable.sort_values(

            by=[
                "average_tier",
                "under_routing",
            ],

            ascending=[
                True,
                True,
            ],
        )
    )

    print("\n")
    print("=" * 90)

    print(
        "BEST CONFIGURATIONS "
        "WITH SOLVABLE SUCCESS >= 88%"
    )

    print("=" * 90)


    columns = [

        "threshold_1",
        "threshold_2",
        "threshold_3",

        "overall_success",
        "solvable_success",

        "under_routing",
        "over_routing",

        "average_tier",

        "tier_1_usage",
        "tier_2_usage",
        "tier_3_usage",
    ]


    print(

        acceptable[
            columns
        ].head(10)
        .to_string(
            index=False
        )

    )

def add_probabilities(
    df,
    probabilities,
):

    df = df.copy()

    for tier, values in probabilities.items():

        df[
            f"{tier}_probability"
        ] = values

    return df


if __name__ == "__main__":
    main()
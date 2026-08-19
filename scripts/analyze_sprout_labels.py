from pathlib import Path

import pandas as pd


DATA_PATH = Path(
    "data/classifier/sprout/routing_train.csv"
)


QUALITY_THRESHOLD = 0.8


def load_data():

    df = pd.read_csv(DATA_PATH)

    score_columns = [
        "tier_1_score",
        "tier_2_score",
        "tier_3_score",
    ]

    df = df.dropna(
        subset=score_columns
    ).copy()

    return df


def analyze_monotonicity(df):

    print("\n")
    print("=" * 70)
    print("MONOTONICITY ANALYSIS")
    print("=" * 70)

    s1 = df["tier_1_score"]
    s2 = df["tier_2_score"]
    s3 = df["tier_3_score"]


    # Ideal assumption:
    #
    # tier_1_score <= tier_2_score <= tier_3_score

    monotonic = (
        (s1 <= s2)
        &
        (s2 <= s3)
    )


    violation_1_to_2 = (
        s1 > s2
    )

    violation_2_to_3 = (
        s2 > s3
    )

    violation_1_to_3 = (
        s1 > s3
    )


    print(
        f"Valid rows: "
        f"{len(df)}"
    )

    print(
        f"Fully monotonic: "
        f"{monotonic.sum()} "
        f"({monotonic.mean():.2%})"
    )

    print(
        f"Tier 1 > Tier 2: "
        f"{violation_1_to_2.sum()} "
        f"({violation_1_to_2.mean():.2%})"
    )

    print(
        f"Tier 2 > Tier 3: "
        f"{violation_2_to_3.sum()} "
        f"({violation_2_to_3.mean():.2%})"
    )

    print(
        f"Tier 1 > Tier 3: "
        f"{violation_1_to_3.sum()} "
        f"({violation_1_to_3.mean():.2%})"
    )


def get_pass_pattern(row, threshold):

    tier_1_pass = int(
        row["tier_1_score"] >= threshold
    )

    tier_2_pass = int(
        row["tier_2_score"] >= threshold
    )

    tier_3_pass = int(
        row["tier_3_score"] >= threshold
    )

    return (
        f"{tier_1_pass}"
        f"{tier_2_pass}"
        f"{tier_3_pass}"
    )


def analyze_pass_patterns(df):

    print("\n")
    print("=" * 70)
    print(
        f"PASS / FAIL PATTERNS "
        f"AT THRESHOLD {QUALITY_THRESHOLD}"
    )
    print("=" * 70)


    df = df.copy()

    df["pass_pattern"] = df.apply(
        lambda row: get_pass_pattern(
            row,
            QUALITY_THRESHOLD
        ),
        axis=1,
    )


    counts = (
        df["pass_pattern"]
        .value_counts()
    )


    descriptions = {

        "111":
            "All three pass",

        "011":
            "Tier 1 fails; Tier 2+3 pass",

        "001":
            "Only Tier 3 passes",

        "000":
            "Nobody passes",

        "101":
            "Tier 1 + Tier 3 pass; Tier 2 fails",

        "110":
            "Tier 1 + Tier 2 pass; Tier 3 fails",

        "010":
            "Only Tier 2 passes",

        "100":
            "Only Tier 1 passes",
    }


    total = len(df)


    for pattern in [
        "111",
        "011",
        "001",
        "000",
        "101",
        "110",
        "010",
        "100",
    ]:

        count = counts.get(
            pattern,
            0
        )

        percentage = (
            count / total
            if total > 0
            else 0
        )


        print(
            f"\n{pattern}: "
            f"{count} "
            f"({percentage:.2%})"
        )

        print(
            descriptions[pattern]
        )


def derive_label(
    row,
    threshold,
):

    if (
        row["tier_1_score"]
        >= threshold
    ):
        return "tier_1"

    if (
        row["tier_2_score"]
        >= threshold
    ):
        return "tier_2"

    if (
        row["tier_3_score"]
        >= threshold
    ):
        return "tier_3"

    return "unresolved"


def analyze_thresholds(df):

    print("\n")
    print("=" * 70)
    print("THRESHOLD SENSITIVITY")
    print("=" * 70)


    thresholds = [
        0.60,
        0.70,
        0.80,
        0.90,
    ]


    labels_by_threshold = {}


    for threshold in thresholds:

        column_name = (
            f"label_{threshold}"
        )


        df[column_name] = df.apply(
            lambda row: derive_label(
                row,
                threshold
            ),
            axis=1,
        )


        labels_by_threshold[
            threshold
        ] = column_name


        print(
            f"\nThreshold = "
            f"{threshold:.2f}"
        )

        distribution = (
            df[column_name]
            .value_counts(
                normalize=True
            )
            .sort_index()
        )


        for (
            label,
            proportion
        ) in distribution.items():

            print(
                f"{label}: "
                f"{proportion:.2%}"
            )


    # ----------------------------------------
    # How many labels change?
    # ----------------------------------------

    print("\n")
    print(
        "LABEL CHANGES BETWEEN THRESHOLDS"
    )
    print("-" * 70)


    comparisons = [
        (0.60, 0.70),
        (0.70, 0.80),
        (0.80, 0.90),
    ]


    for first, second in comparisons:

        first_column = (
            labels_by_threshold[first]
        )

        second_column = (
            labels_by_threshold[second]
        )


        changed = (
            df[first_column]
            !=
            df[second_column]
        )


        print(
            f"{first:.2f} → "
            f"{second:.2f}: "
            f"{changed.mean():.2%} "
            f"changed labels"
        )


def analyze_borderline_scores(df):

    print("\n")
    print("=" * 70)
    print("BORDERLINE SCORE ANALYSIS")
    print("=" * 70)


    margin = 0.10


    lower = (
        QUALITY_THRESHOLD
        - margin
    )

    upper = (
        QUALITY_THRESHOLD
        + margin
    )


    score_columns = [
        "tier_1_score",
        "tier_2_score",
        "tier_3_score",
    ]


    borderline = pd.Series(
        False,
        index=df.index,
    )


    for column in score_columns:

        score_is_borderline = (
            (df[column] >= lower)
            &
            (df[column] <= upper)
        )

        borderline = (
            borderline
            |
            score_is_borderline
        )


    print(
        f"Rows with at least one "
        f"score between "
        f"{lower:.2f} and {upper:.2f}:"
    )

    print(
        f"{borderline.sum()} "
        f"({borderline.mean():.2%})"
    )


    print(
        "\nNon-borderline rows:"
    )

    print(
        f"{(~borderline).sum()} "
        f"({(~borderline).mean():.2%})"
    )


def main():

    df = load_data()


    analyze_monotonicity(
        df
    )


    analyze_pass_patterns(
        df
    )


    analyze_thresholds(
        df
    )


    analyze_borderline_scores(
        df
    )


if __name__ == "__main__":
    main()
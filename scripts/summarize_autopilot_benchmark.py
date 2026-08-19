from pathlib import Path

import pandas as pd


BENCHMARK_PATH = Path(
    "data/benchmarks/autopilot_benchmark.csv"
)


def main():

    if not BENCHMARK_PATH.exists():

        raise FileNotFoundError(
            f"Benchmark file not found: "
            f"{BENCHMARK_PATH}"
        )


    df = pd.read_csv(
        BENCHMARK_PATH
    )


    if df.empty:

        raise ValueError(
            "Benchmark CSV is empty."
        )


    # ======================================================
    # Convert boolean columns safely
    # ======================================================

    boolean_columns = [

        "quality_passed",

        "verification_performed",

        "escalated",

    ]


    for column in boolean_columns:

        if df[column].dtype == "object":

            df[column] = (

                df[column]
                .astype(str)
                .str.lower()
                .map({
                    "true": True,
                    "false": False,
                })
            )


    # ======================================================
    # Strategy summary
    # ======================================================

    summary = (

        df.groupby(
            "strategy"
        )

        .agg(

            requests=(
                "prompt_id",
                "count",
            ),

            total_cost_usd=(
                "strategy_cost_usd",
                "sum",
            ),

            average_cost_usd=(
                "strategy_cost_usd",
                "mean",
            ),

            average_latency_ms=(
                "strategy_latency_ms",
                "mean",
            ),

            quality_pass_rate=(
                "quality_passed",
                "mean",
            ),

            average_quality_score=(
                "quality_score",
                "mean",
            ),

            verification_rate=(
                "verification_performed",
                "mean",
            ),

            escalation_rate=(
                "escalated",
                "mean",
            ),

            average_attempts=(
                "attempt_count",
                "mean",
            ),

        )

        .reset_index()
    )


    # ======================================================
    # Tier 3 baseline
    # ======================================================

    baseline_rows = (
        summary[
            summary["strategy"]
            ==
            "always_tier_3"
        ]
    )


    if baseline_rows.empty:

        raise ValueError(
            "always_tier_3 baseline "
            "was not found."
        )


    baseline_cost = (

        baseline_rows[
            "total_cost_usd"
        ]
        .iloc[0]
    )


    # ======================================================
    # Cost savings
    # ======================================================

    summary[
        "cost_savings_vs_tier3"
    ] = (

        (
            baseline_cost
            -
            summary[
                "total_cost_usd"
            ]
        )

        /

        baseline_cost
    )


    # ======================================================
    # Format percentages
    # ======================================================

    formatted = (
        summary.copy()
    )


    percentage_columns = [

        "quality_pass_rate",

        "verification_rate",

        "escalation_rate",

        "cost_savings_vs_tier3",

    ]


    for column in percentage_columns:

        formatted[column] = (

            formatted[column]
            * 100
        ).round(2)


    formatted[
        "total_cost_usd"
    ] = (

        formatted[
            "total_cost_usd"
        ]
        .round(8)
    )


    formatted[
        "average_cost_usd"
    ] = (

        formatted[
            "average_cost_usd"
        ]
        .round(8)
    )


    formatted[
        "average_latency_ms"
    ] = (

        formatted[
            "average_latency_ms"
        ]
        .round(2)
    )


    formatted[
        "average_quality_score"
    ] = (

        formatted[
            "average_quality_score"
        ]
        .round(3)
    )


    formatted[
        "average_attempts"
    ] = (

        formatted[
            "average_attempts"
        ]
        .round(2)
    )


    # ======================================================
    # Print main results
    # ======================================================

    print("\n")
    print("=" * 140)

    print(
        "LLM COST AUTOPILOT BENCHMARK SUMMARY"
    )

    print("=" * 140)


    print(

        formatted.to_string(
            index=False
        )

    )


    # ======================================================
    # Tier usage
    # ======================================================

    print("\n")
    print("=" * 100)

    print(
        "FINAL TIER DISTRIBUTION"
    )

    print("=" * 100)


    tier_distribution = (

        df.groupby(
            [
                "strategy",
                "final_tier",
            ]
        )

        .size()

        .groupby(
            level=0
        )

        .apply(
            lambda x:
                x / x.sum()
        )

    )


    print(
        tier_distribution
    )


    # ======================================================
    # Category breakdown
    # ======================================================

    print("\n")
    print("=" * 100)

    print(
        "CATEGORY BREAKDOWN"
    )

    print("=" * 100)


    category_summary = (

        df.groupby(
            [
                "category",
                "strategy",
            ]
        )

        .agg(

            cost=(
                "strategy_cost_usd",
                "mean",
            ),

            quality=(
                "quality_score",
                "mean",
            ),

            passed=(
                "quality_passed",
                "mean",
            ),

        )

        .reset_index()
    )


    print(

        category_summary.to_string(
            index=False
        )

    )


if __name__ == "__main__":

    main()
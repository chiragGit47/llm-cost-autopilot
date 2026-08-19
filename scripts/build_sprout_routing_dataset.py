import csv
from collections import Counter
from pathlib import Path

from datasets import load_dataset


QUALITY_THRESHOLD = 0.8


TIER_MODELS = {
    "tier_1": "wxai-llama-3-2-1b-instruct",
    "tier_2": "wxai-llama-3-1-8b-instruct",
    "tier_3": "wxai-llama-3-1-70b-instruct",
}


OUTPUT_DIR = Path(
    "data/classifier/sprout"
)


def get_score(model_result):

    if not isinstance(model_result, dict):
        return None

    score = model_result.get("score")

    if score is None:
        return None

    try:
        return float(score)

    except (TypeError, ValueError):
        return None


def derive_routing_label(row):

    tier_1_score = get_score(
        row[TIER_MODELS["tier_1"]]
    )

    tier_2_score = get_score(
        row[TIER_MODELS["tier_2"]]
    )

    tier_3_score = get_score(
        row[TIER_MODELS["tier_3"]]
    )


    if (
        tier_1_score is not None
        and tier_1_score >= QUALITY_THRESHOLD
    ):
        label = "tier_1"

    elif (
        tier_2_score is not None
        and tier_2_score >= QUALITY_THRESHOLD
    ):
        label = "tier_2"

    elif (
        tier_3_score is not None
        and tier_3_score >= QUALITY_THRESHOLD
    ):
        label = "tier_3"

    else:
        label = "unresolved"


    return (
        label,
        tier_1_score,
        tier_2_score,
        tier_3_score,
    )


def process_split(
    dataset,
    split_name
):

    output_path = (
        OUTPUT_DIR
        / f"routing_{split_name}.csv"
    )

    counts = Counter()


    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as csvfile:

        fieldnames = [
            "key",
            "source_dataset",
            "dataset_level",
            "prompt",
            "routing_label",
            "tier_1_score",
            "tier_2_score",
            "tier_3_score",
        ]


        writer = csv.DictWriter(
            csvfile,
            fieldnames=fieldnames,
        )

        writer.writeheader()


        for row in dataset:

            (
                label,
                score_1,
                score_2,
                score_3,
            ) = derive_routing_label(row)


            counts[label] += 1


            writer.writerow({
                "key":
                    row["key"],

                "source_dataset":
                    row["dataset"],

                "dataset_level":
                    row["dataset_level"],

                "prompt":
                    row["prompt"],

                "routing_label":
                    label,

                "tier_1_score":
                    score_1,

                "tier_2_score":
                    score_2,

                "tier_3_score":
                    score_3,
            })


    print(
        f"\n{split_name.upper()}"
    )

    print("-" * 50)

    for label, count in counts.items():

        print(
            f"{label}: {count}"
        )

    print(
        f"Total: {sum(counts.values())}"
    )

    print(
        f"Saved: {output_path}"
    )


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    dataset = load_dataset(
        "CARROT-LLM-Routing/SPROUT"
    )


    for split_name in [
        "train",
        "validation",
        "test",
    ]:

        process_split(
            dataset[split_name],
            split_name,
        )


if __name__ == "__main__":
    main()
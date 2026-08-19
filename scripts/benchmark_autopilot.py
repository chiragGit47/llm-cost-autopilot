import asyncio
import csv

from pathlib import Path


from app.core.registry import (
    MODEL_REGISTRY,
)

from app.providers.gemini_provider import (
    GeminiProvider,
)

from app.services.router_service import (
    RouterService,
)

from app.services.generation_service import (
    GenerationService,
)

from app.services.benchmark_evaluator_service import (
    BenchmarkEvaluatorService,
)


# ==========================================================
# Output
# ==========================================================

OUTPUT_PATH = Path(
    "data/benchmarks/autopilot_benchmark.csv"
)


# ==========================================================
# Small quota-friendly benchmark
# ==========================================================

PROMPTS = [

    {
        "category": "simple",

        "prompt":
            "What is Python?",
    },

    {
        "category": "moderate",

        "prompt": (
            "Explain authentication and "
            "authorization with examples."
        ),
    },

    {
        "category": "complex",

        "prompt": (
            "Design a globally distributed payment "
            "processing platform. Explain idempotency, "
            "duplicate requests, retries, database "
            "failures, race conditions, consistency, "
            "horizontal scaling and disaster recovery."
        ),
    },

]


STRATEGIES = [

    "always_tier_3",

    "ml_only",

    "autopilot",

]


# ==========================================================
# Strategy A
#
# Always use strongest model
# ==========================================================

async def run_always_tier_3(
    prompt: str,
    provider: GeminiProvider,
) -> dict:

    response = (
        await provider.send_request(

            prompt=prompt,

            model_config=
                MODEL_REGISTRY["tier_3"],
        )
    )

    return {

        "text":
            response.text,

        "initial_tier":
            "tier_3",

        "final_tier":
            "tier_3",

        "verification_performed":
            False,

        "escalated":
            False,

        "attempt_count":
            1,

        "strategy_cost_usd":
            response.estimated_cost_usd,

        "strategy_latency_ms":
            response.latency_ms,
    }


# ==========================================================
# Strategy B
#
# ML router only
#
# No runtime verification.
# No escalation.
# ==========================================================

async def run_ml_only(
    prompt: str,
    provider: GeminiProvider,
    router: RouterService,
) -> dict:

    decision = (
        router.route(
            prompt
        )
    )

    selected_tier = (
        decision.selected_tier
    )

    model_config = (
        MODEL_REGISTRY[
            selected_tier
        ]
    )

    response = (
        await provider.send_request(

            prompt=prompt,

            model_config=
                model_config,
        )
    )

    return {

        "text":
            response.text,

        "initial_tier":
            selected_tier,

        "final_tier":
            selected_tier,

        "verification_performed":
            False,

        "escalated":
            False,

        "attempt_count":
            1,

        "strategy_cost_usd":
            response.estimated_cost_usd,

        "strategy_latency_ms":
            response.latency_ms,
    }


# ==========================================================
# Strategy C
#
# Full Cost Autopilot
#
# ML routing
# +
# selective verification
# +
# escalation
# ==========================================================

async def run_autopilot(
    prompt: str,
    service: GenerationService,
) -> dict:

    result = (
        await service.generate(
            prompt
        )
    )

    # We care whether verification happened
    # ANYWHERE in the request, not just on
    # the final attempt.

    verification_performed = any(

        attempt.verification_performed

        for attempt
        in result.attempts
    )

    return {

        "text":
            result.text,

        "initial_tier":
            result.initial_tier,

        "final_tier":
            result.final_tier,

        "verification_performed":
            verification_performed,

        "escalated":
            result.escalated,

        "attempt_count":
            len(
                result.attempts
            ),

        "strategy_cost_usd":
            result.total_estimated_cost_usd,

        "strategy_latency_ms":
            result.total_latency_ms,
    }


# ==========================================================
# Strategy dispatcher
# ==========================================================

async def run_strategy(
    strategy_name: str,
    prompt: str,
    provider: GeminiProvider,
    router: RouterService,
    generation_service: GenerationService,
) -> dict:

    if strategy_name == "always_tier_3":

        return await run_always_tier_3(
            prompt=prompt,
            provider=provider,
        )

    if strategy_name == "ml_only":

        return await run_ml_only(
            prompt=prompt,
            provider=provider,
            router=router,
        )

    if strategy_name == "autopilot":

        return await run_autopilot(
            prompt=prompt,
            service=generation_service,
        )

    raise ValueError(
        f"Unknown benchmark strategy: "
        f"{strategy_name}"
    )


# ==========================================================
# Save CSV
# ==========================================================

def save_results(
    rows: list[dict],
):

    if not rows:
        return

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(

            file,

            fieldnames=
                rows[0].keys(),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ==========================================================
# Pretty print one strategy result
# ==========================================================

def print_strategy_result(
    strategy: str,
    result: dict,
):

    print(
        f"\n{strategy.upper()}"
    )

    print(
        f"Initial tier: "
        f"{result['initial_tier']}"
    )

    print(
        f"Final tier: "
        f"{result['final_tier']}"
    )

    print(
        f"Verification performed: "
        f"{result['verification_performed']}"
    )

    print(
        f"Escalated: "
        f"{result['escalated']}"
    )

    print(
        f"Attempts: "
        f"{result['attempt_count']}"
    )

    print(
        f"Cost: "
        f"${result['strategy_cost_usd']:.8f}"
    )

    print(
        f"Latency: "
        f"{result['strategy_latency_ms']:.2f} ms"
    )


# ==========================================================
# Main benchmark
# ==========================================================

async def main():

    print(
        "\nStarting LLM Cost Autopilot benchmark..."
    )

    print(
        f"Prompts: {len(PROMPTS)}"
    )

    print(
        f"Strategies: {len(STRATEGIES)}"
    )


    # ------------------------------------------------------
    # Provider used by:
    #
    # - always Tier 3
    # - ML-only
    # - benchmark evaluator
    #
    # GenerationService currently owns another provider.
    # We will refactor that later.
    # ------------------------------------------------------

    provider = GeminiProvider()

    router = RouterService()

    generation_service = (
        GenerationService()
    )

    evaluator = (
        BenchmarkEvaluatorService(
            provider=provider
        )
    )


    rows = []

    total_benchmark_evaluation_cost = 0.0
    total_benchmark_evaluation_latency = 0.0


    try:

        # ==================================================
        # Loop through benchmark prompts
        # ==================================================

        for index, item in enumerate(
            PROMPTS,
            start=1,
        ):

            prompt = (
                item["prompt"]
            )

            category = (
                item["category"]
            )


            print("\n")
            print("=" * 90)

            print(
                f"PROMPT "
                f"{index}/{len(PROMPTS)}"
            )

            print(
                f"CATEGORY: "
                f"{category.upper()}"
            )

            print("=" * 90)

            print(prompt)


            # ==============================================
            # 1. Generate answers using all strategies
            # ==============================================

            strategy_results = {}


            for strategy in STRATEGIES:

                print(
                    f"\nRunning strategy: "
                    f"{strategy}"
                )


                result = (
                    await run_strategy(

                        strategy_name=
                            strategy,

                        prompt=
                            prompt,

                        provider=
                            provider,

                        router=
                            router,

                        generation_service=
                            generation_service,
                    )
                )


                strategy_results[
                    strategy
                ] = result


                print_strategy_result(
                    strategy,
                    result,
                )


            # ==============================================
            # 2. ONE benchmark judge call
            # ==============================================

            print(
                "\nEvaluating all three answers "
                "with ONE benchmark judge call..."
            )


            evaluation = (
                await evaluator.evaluate(

                    user_prompt=
                        prompt,

                    answers={

                        "always_tier_3":
                            strategy_results[
                                "always_tier_3"
                            ]["text"],

                        "ml_only":
                            strategy_results[
                                "ml_only"
                            ]["text"],

                        "autopilot":
                            strategy_results[
                                "autopilot"
                            ]["text"],
                    },
                )
            )


            total_benchmark_evaluation_cost += (
                evaluation[
                    "evaluation_cost_usd"
                ]
            )

            total_benchmark_evaluation_latency += (
                evaluation[
                    "evaluation_latency_ms"
                ]
            )


            print(
                "\nQUALITY RESULTS"
            )


            for strategy in STRATEGIES:

                quality = (
                    evaluation[
                        strategy
                    ]
                )

                print(
                    f"{strategy}: "
                    f"score={quality['score']:.2f}, "
                    f"passed={quality['passed']}"
                )


            print(
                "\nBenchmark evaluator cost "
                "(NOT production cost): "
                f"${evaluation['evaluation_cost_usd']:.8f}"
            )


            # ==============================================
            # 3. Create CSV rows
            # ==============================================

            for strategy in STRATEGIES:

                result = (
                    strategy_results[
                        strategy
                    ]
                )

                quality = (
                    evaluation[
                        strategy
                    ]
                )


                row = {

                    "prompt_id":
                        index,

                    "category":
                        category,

                    "strategy":
                        strategy,

                    "initial_tier":
                        result[
                            "initial_tier"
                        ],

                    "final_tier":
                        result[
                            "final_tier"
                        ],

                    "verification_performed":
                        result[
                            "verification_performed"
                        ],

                    "escalated":
                        result[
                            "escalated"
                        ],

                    "attempt_count":
                        result[
                            "attempt_count"
                        ],

                    "strategy_cost_usd":
                        result[
                            "strategy_cost_usd"
                        ],

                    "strategy_latency_ms":
                        result[
                            "strategy_latency_ms"
                        ],

                    "quality_passed":
                        quality[
                            "passed"
                        ],

                    "quality_score":
                        quality[
                            "score"
                        ],
                }


                rows.append(
                    row
                )


            # ==============================================
            # 4. Checkpoint after EVERY completed prompt
            # ==============================================

            save_results(
                rows
            )


            print(
                f"\nCheckpoint saved: "
                f"{OUTPUT_PATH}"
            )


        # ==================================================
        # Completed
        # ==================================================

        print("\n")
        print("=" * 90)
        print("BENCHMARK COMPLETE")
        print("=" * 90)


        print(
            f"\nRaw benchmark results saved to:\n"
            f"{OUTPUT_PATH}"
        )


        print(
            "\nTotal benchmark evaluator cost "
            "(excluded from strategy costs):"
        )

        print(
            f"${total_benchmark_evaluation_cost:.8f}"
        )


        print(
            "\nTotal benchmark evaluator latency:"
        )

        print(
            f"{total_benchmark_evaluation_latency:.2f} ms"
        )


    finally:

        await provider.close()

        await generation_service.close()


if __name__ == "__main__":

    asyncio.run(
        main()
    )
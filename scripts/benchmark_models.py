import asyncio
import csv
from pathlib import Path

from app.core.registry import MODEL_REGISTRY
from app.providers.gemini_provider import GeminiProvider


OUTPUT_PATH = Path(
    "data/benchmarks/model_benchmark.csv"
)


PROMPTS = [

    # ------------------------
    # SIMPLE
    # ------------------------

    {
        "id": "P01",
        "complexity": "simple",
        "prompt": """
Explain what a Python list is in two sentences.
""",
    },

    {
        "id": "P02",
        "complexity": "simple",
        "prompt": """
What does HTTP status code 404 mean?
Answer in one sentence.
""",
    },

    {
        "id": "P03",
        "complexity": "simple",
        "prompt": """
Rewrite this sentence professionally:

"the server broke so we fixed it"
""",
    },


    # ------------------------
    # MODERATE
    # ------------------------

    {
        "id": "P04",
        "complexity": "moderate",
        "prompt": """
Explain three differences between REST APIs
and GraphQL APIs.

Keep the response under 150 words.
""",
    },

    {
        "id": "P05",
        "complexity": "moderate",
        "prompt": """
Write a Python function that accepts a list
of integers and returns the second largest
unique number.

Explain the time complexity.
""",
    },

    {
        "id": "P06",
        "complexity": "moderate",
        "prompt": """
Find the problem in this Python code and
explain how to fix it:

numbers = [1, 2, 3, 4]

for i in range(len(numbers)):
    print(numbers[i + 1])
""",
    },


    # ------------------------
    # COMPLEX
    # ------------------------

    {
        "id": "P07",
        "complexity": "complex",
        "prompt": """
Design a scalable API rate-limiting system
for a distributed FastAPI application.

Explain:

1. architecture
2. Redis usage
3. race conditions
4. failure handling
5. horizontal scaling

Keep the explanation concise.
""",
    },

    {
        "id": "P08",
        "complexity": "complex",
        "prompt": """
A Python application has two asynchronous
workers updating the same account balance.

Explain how a race condition could occur,
show a simple example scenario, and propose
two ways to prevent it.
""",
    },

    {
        "id": "P09",
        "complexity": "complex",
        "prompt": """
You are designing a production RAG system
for 5 million documents.

Compare FAISS, a managed vector database,
and PostgreSQL with pgvector.

Discuss scalability, operational complexity,
latency, filtering, and cost trade-offs.
""",
    },

    {
        "id": "P10",
        "complexity": "complex",
        "prompt": """
Design an LLM routing system that minimizes
API cost while maintaining response quality.

The system has access to three models with
different prices and capabilities.

Explain how you would classify requests,
route them, measure quality, detect bad
routing decisions, and continuously improve
the router.
""",
    },
]


async def main():

    provider = GeminiProvider()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = [
        "prompt_id",
        "expected_complexity",
        "prompt",
        "model",
        "input_tokens",
        "output_tokens",
        "thinking_tokens",
        "latency_ms",
        "estimated_cost_usd",
        "response",
        "success",
        "error",
    ]

    try:

        with open(
            OUTPUT_PATH,
            "w",
            newline="",
            encoding="utf-8"
        ) as csvfile:

            writer = csv.DictWriter(
                csvfile,
                fieldnames=fieldnames
            )

            writer.writeheader()

            for prompt_data in PROMPTS:

                prompt_id = prompt_data["id"]
                complexity = prompt_data["complexity"]
                prompt = prompt_data["prompt"].strip()

                print("\n")
                print("#" * 70)
                print(
                    f"{prompt_id} | "
                    f"{complexity.upper()}"
                )
                print("#" * 70)

                for (
                    model_name,
                    model_config
                ) in MODEL_REGISTRY.items():

                    print(
                        f"\nTesting {model_name}..."
                    )

                    try:

                        result = (
                            await provider.send_request(
                                prompt=prompt,
                                model_config=model_config,
                            )
                        )

                        writer.writerow({
                            "prompt_id": prompt_id,
                            "expected_complexity": complexity,
                            "prompt": prompt,
                            "model": model_name,
                            "input_tokens":
                                result.input_tokens,
                            "output_tokens":
                                result.output_tokens,
                            "thinking_tokens":
                                result.thinking_tokens,
                            "latency_ms":
                                round(
                                    result.latency_ms,
                                    2
                                ),
                            "estimated_cost_usd":
                                result.estimated_cost_usd,
                            "response":
                                result.text,
                            "success": True,
                            "error": "",
                        })

                        csvfile.flush()

                        print(
                            f"  Input: "
                            f"{result.input_tokens}"
                        )

                        print(
                            f"  Output: "
                            f"{result.output_tokens}"
                        )

                        print(
                            f"  Thinking: "
                            f"{result.thinking_tokens}"
                        )

                        print(
                            f"  Latency: "
                            f"{result.latency_ms:.2f} ms"
                        )

                        print(
                            f"  Cost: "
                            f"${result.estimated_cost_usd:.8f}"
                        )

                    except Exception as error:

                        writer.writerow({
                            "prompt_id": prompt_id,
                            "expected_complexity":
                                complexity,
                            "prompt": prompt,
                            "model": model_name,
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "thinking_tokens": 0,
                            "latency_ms": 0,
                            "estimated_cost_usd": 0,
                            "response": "",
                            "success": False,
                            "error": str(error),
                        })

                        csvfile.flush()

                        print(
                            f"  FAILED: {error}"
                        )

                    # Small pause between requests.
                    await asyncio.sleep(1)

    finally:

        await provider.close()

    print("\nBenchmark complete.")
    print(
        f"Results saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    asyncio.run(main())
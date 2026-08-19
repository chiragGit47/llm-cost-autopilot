from app.core.models import (
    GenerationAttempt,
    GenerationResult,
)

from app.services.usage_logger_service import (
    UsageLoggerService,
)


def make_fake_result():

    attempt = GenerationAttempt(

        tier=
            "tier_1",

        model_id=
            "fake-tier-1",

        text=
            "Python is a programming language.",

        verification_performed=
            False,

        verification_passed=
            None,

        verification_score=
            None,

        verification_reason=
            "Economy mode test.",

        input_tokens=
            10,

        output_tokens=
            100,

        thinking_tokens=
            0,

        generation_latency_ms=
            500.0,

        verification_latency_ms=
            0.0,

        generation_cost_usd=
            0.0001,

        verification_cost_usd=
            0.0,

        total_attempt_cost_usd=
            0.0001,
    )


    return GenerationResult(

        text=
            attempt.text,

        mode=
            "economy",

        initial_tier=
            "tier_1",

        final_tier=
            "tier_1",

        model_id=
            "fake-tier-1",

        routing_scores={
            "tier_1": 0.95,
            "tier_2": 0.98,
            "tier_3": 0.99,
        },

        verification_performed=
            False,

        verification_passed=
            None,

        verification_score=
            None,

        verification_reason=
            "Economy mode test.",

        escalated=
            False,

        attempts=[
            attempt
        ],

        total_estimated_cost_usd=
            0.0001,

        total_latency_ms=
            500.0,
    )


def test_generation_is_logged(
    tmp_path,
):

    database = (
        tmp_path
        /
        "test.db"
    )


    logger = UsageLoggerService(
        database_path=database
    )


    result = make_fake_result()


    request_id = (
        logger.log_generation(
            result
        )
    )


    history = (
        logger.get_request_history(
            limit=10
        )
    )


    assert (
        len(history)
        ==
        1
    )


    assert (
        history[0]["request_id"]
        ==
        request_id
    )


    assert (
        history[0]["mode"]
        ==
        "economy"
    )


    assert (
        history[0]["final_tier"]
        ==
        "tier_1"
    )


    assert (
        history[0]["input_tokens"]
        ==
        10
    )


    assert (
        history[0]["output_tokens"]
        ==
        100
    )


def test_summary_is_correct(
    tmp_path,
):

    logger = UsageLoggerService(

        database_path=
            tmp_path
            /
            "summary.db"
    )


    logger.log_generation(
        make_fake_result()
    )


    summary = (
        logger.get_summary()
    )


    assert (
        summary["total_requests"]
        ==
        1
    )


    assert (
        summary["total_cost_usd"]
        ==
        0.0001
    )


    assert (
        summary["tokens"]["input"]
        ==
        10
    )


    assert (
        summary["tokens"]["output"]
        ==
        100
    )


    assert (
        summary[
            "tier_usage"
        ][
            "tier_1"
        ][
            "count"
        ]
        ==
        1
    )


    assert (
        summary[
            "mode_usage"
        ][
            "economy"
        ][
            "count"
        ]
        ==
        1
    )


def test_savings_calculation(
    tmp_path,
):

    logger = UsageLoggerService(

        database_path=
            tmp_path
            /
            "savings.db"
    )


    logger.log_generation(
        make_fake_result()
    )


    savings = (
        logger.get_savings_summary()
    )


    assert (
        savings["total_requests"]
        ==
        1
    )


    assert (
        savings["actual_cost_usd"]
        ==
        0.0001
    )


    assert (
        savings[
            "tier_3_equivalent_cost_usd"
        ]
        >
        0
    )


    # Fundamental accounting identity:
    #
    # baseline - actual = savings

    calculated_savings = (

        savings[
            "tier_3_equivalent_cost_usd"
        ]

        -

        savings[
            "actual_cost_usd"
        ]
    )


    assert abs(

        calculated_savings

        -

        savings[
            "estimated_savings_usd"
        ]

    ) < 1e-12


def test_empty_database_summary(
    tmp_path,
):

    logger = UsageLoggerService(

        database_path=
            tmp_path
            /
            "empty.db"
    )


    summary = (
        logger.get_summary()
    )


    assert (
        summary["total_requests"]
        ==
        0
    )


    assert (
        summary["total_cost_usd"]
        ==
        0
    )


    assert (
        summary["verification_rate"]
        ==
        0
    )
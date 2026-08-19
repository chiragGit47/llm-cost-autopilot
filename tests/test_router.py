from app.services.router_service import (
    RouterService,
)


def test_simple_prompt_routes_to_tier_1():

    router = RouterService()

    decision = router.route(
        "What is Python?"
    )

    assert (
        decision.selected_tier
        ==
        "tier_1"
    )

    assert (
        decision.scores["tier_1"]
        >=
        decision.thresholds["tier_1"]
    )

    assert (
        decision.fallback_used
        is False
    )


def test_moderate_prompt_routes_to_tier_2():

    router = RouterService()

    prompt = (
        "Explain authentication and "
        "authorization with examples."
    )

    decision = router.route(
        prompt
    )

    assert (
        decision.selected_tier
        ==
        "tier_2"
    )

    assert (
        decision.scores["tier_1"]
        <
        decision.thresholds["tier_1"]
    )

    assert (
        decision.scores["tier_2"]
        >=
        decision.thresholds["tier_2"]
    )


def test_router_returns_all_three_scores():

    router = RouterService()

    decision = router.route(
        "Explain REST APIs."
    )

    assert set(
        decision.scores.keys()
    ) == {
        "tier_1",
        "tier_2",
        "tier_3",
    }


def test_router_rejects_empty_prompt():

    router = RouterService()

    try:

        router.route("")

    except ValueError:

        pass

    else:

        raise AssertionError(
            "Empty prompt should raise ValueError."
        )
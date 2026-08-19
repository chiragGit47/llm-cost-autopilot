from app.core.models import (
    RoutingDecision,
)

from app.services.verification_policy_service import (
    VerificationPolicyService,
)


def make_decision(
    selected_tier: str,
    selected_score: float,
):

    scores = {
        "tier_1": 0.50,
        "tier_2": 0.50,
        "tier_3": 0.95,
    }

    scores[
        selected_tier
    ] = selected_score


    return RoutingDecision(

        selected_tier=
            selected_tier,

        scores=
            scores,

        thresholds={
            "tier_1": 0.80,
            "tier_2": 0.70,
        },

        fallback_used=
            False,

        reason=
            "Test routing decision.",
    )


def test_high_confidence_skips_verification():

    policy = VerificationPolicyService(
        auto_accept_score=0.90
    )

    decision = make_decision(
        selected_tier="tier_1",
        selected_score=0.95,
    )

    should_verify, reason = (
        policy.should_verify(

            routing_decision=
                decision,

            current_tier=
                "tier_1",

            is_escalated=
                False,
        )
    )

    assert (
        should_verify
        is False
    )

    assert (
        "0.95"
        in reason
    )


def test_lower_confidence_requires_verification():

    policy = VerificationPolicyService(
        auto_accept_score=0.90
    )

    decision = make_decision(
        selected_tier="tier_2",
        selected_score=0.84,
    )

    should_verify, _ = (
        policy.should_verify(

            routing_decision=
                decision,

            current_tier=
                "tier_2",

            is_escalated=
                False,
        )
    )

    assert (
        should_verify
        is True
    )


def test_escalated_attempt_requires_verification():

    policy = VerificationPolicyService(
        auto_accept_score=0.90
    )

    decision = make_decision(
        selected_tier="tier_1",
        selected_score=0.95,
    )

    should_verify, _ = (
        policy.should_verify(

            routing_decision=
                decision,

            current_tier=
                "tier_2",

            is_escalated=
                True,
        )
    )

    assert (
        should_verify
        is True
    )


def test_tier_3_skips_verification():

    policy = VerificationPolicyService(
        auto_accept_score=0.90
    )

    decision = make_decision(
        selected_tier="tier_3",
        selected_score=0.70,
    )

    should_verify, _ = (
        policy.should_verify(

            routing_decision=
                decision,

            current_tier=
                "tier_3",

            is_escalated=
                False,
        )
    )

    assert (
        should_verify
        is False
    )
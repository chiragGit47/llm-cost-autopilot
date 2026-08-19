from app.core.models import (
    RoutingDecision,
)


class VerificationPolicyService:

    def __init__(
        self,
        auto_accept_score: float = 0.90,
    ):

        if not 0.0 <= auto_accept_score <= 1.0:

            raise ValueError(
                "auto_accept_score must be "
                "between 0 and 1."
            )


        self.auto_accept_score = (
            auto_accept_score
        )


    def should_verify(
        self,
        routing_decision: RoutingDecision,
        current_tier: str,
        is_escalated: bool,
    ) -> tuple[bool, str]:

        # ==========================================
        # Tier 3 cannot escalate any further.
        # ==========================================

        if current_tier == "tier_3":

            return (
                False,
                "Tier 3 is the strongest available "
                "tier, so verification would not "
                "enable further escalation."
            )


        # ==========================================
        # If an earlier model already FAILED,
        # verify the escalated model too.
        # ==========================================

        if is_escalated:

            return (
                True,
                "This model was reached through "
                "escalation, so its answer must "
                "be verified before acceptance."
            )


        # ==========================================
        # Initial routing decision
        # ==========================================

        selected_score = (
            routing_decision
            .scores[current_tier]
        )


        if (
            selected_score
            >=
            self.auto_accept_score
        ):

            return (
                False,

                f"Routing confidence "
                f"{selected_score:.4f} reached "
                f"the auto-accept threshold "
                f"{self.auto_accept_score:.2f}."
            )


        return (
            True,

            f"Routing confidence "
            f"{selected_score:.4f} is below "
            f"the auto-accept threshold "
            f"{self.auto_accept_score:.2f}."
        )
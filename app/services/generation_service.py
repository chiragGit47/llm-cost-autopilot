from app.core.models import (
    GenerationAttempt,
    GenerationResult,
)

from app.core.registry import (
    MODEL_REGISTRY,
)

from app.services.router_service import (
    RouterService,
)

from app.services.quality_verifier_service import (
    QualityVerifierService,
)

from app.services.verification_policy_service import (
    VerificationPolicyService,
)

from app.providers.gemini_provider import (
    GeminiProvider,
)

from app.core.config import (
    get_settings,
)

class GenerationService:

    VALID_MODES = {
        "economy",
        "balanced",
    }


    TIER_SEQUENCE = [
        "tier_1",
        "tier_2",
        "tier_3",
    ]


    def __init__(self):

        # ML router
        self.router = RouterService()

        settings = get_settings()


        # Shared Gemini provider
        self.provider = GeminiProvider()


        # LLM-based quality verifier
        self.verifier = (
            QualityVerifierService(
                provider=self.provider
            )
        )


        # Determines when verification
        # is worth paying for.
        self.verification_policy = (
            VerificationPolicyService(

                auto_accept_score=
                    settings
                    .verification_auto_accept_score

            )
)


    # ======================================================
    # Mode validation
    # ======================================================

    def _validate_mode(
        self,
        mode: str,
    ) -> str:

        if not isinstance(
            mode,
            str,
        ):

            raise TypeError(
                "Generation mode must be a string."
            )


        mode = (
            mode
            .strip()
            .lower()
        )


        if mode not in self.VALID_MODES:

            raise ValueError(
                f"Unsupported generation mode: "
                f"{mode}. "
                f"Supported modes are: "
                f"{sorted(self.VALID_MODES)}"
            )


        return mode


    # ======================================================
    # Determine escalation path
    # ======================================================

    def _get_tiers_to_try(
        self,
        initial_tier: str,
    ) -> list[str]:

        try:

            start_index = (
                self.TIER_SEQUENCE.index(
                    initial_tier
                )
            )

        except ValueError as exc:

            raise ValueError(
                f"Unknown routing tier: "
                f"{initial_tier}"
            ) from exc


        return self.TIER_SEQUENCE[
            start_index:
        ]


    # ======================================================
    # ECONOMY MODE
    #
    # Router
    #   ↓
    # selected model
    #   ↓
    # return immediately
    #
    # No verification.
    # No escalation.
    # ======================================================

    async def _generate_economy(
        self,
        prompt: str,
        routing_decision,
    ) -> GenerationResult:

        selected_tier = (
            routing_decision.selected_tier
        )


        model_config = (
            MODEL_REGISTRY[
                selected_tier
            ]
        )


        model_response = (
            await self.provider.send_request(

                prompt=prompt,

                model_config=
                    model_config,
            )
        )


        generation_cost = (
            model_response
            .estimated_cost_usd
        )


        generation_latency = (
            model_response
            .latency_ms
        )


        reason = (
            "Economy mode trusts the ML "
            "router and skips runtime "
            "verification."
        )


        attempt = GenerationAttempt(

            tier=
                selected_tier,

            model_id=
                model_response.model_id,

            text=
                model_response.text,

            verification_performed=
                False,

            verification_passed=
                None,

            verification_score=
                None,

            verification_reason=
                reason,

            input_tokens=
                model_response.input_tokens,

            output_tokens=
                model_response.output_tokens,

            thinking_tokens=
                model_response.thinking_tokens,

            generation_latency_ms=
                generation_latency,

            verification_latency_ms=
                0.0,

            generation_cost_usd=
                generation_cost,

            verification_cost_usd=
                0.0,

            total_attempt_cost_usd=
                generation_cost,
        )


        return GenerationResult(

            text=
                model_response.text,

            mode=
                "economy",

            initial_tier=
                selected_tier,

            final_tier=
                selected_tier,

            model_id=
                model_response.model_id,

            routing_scores=
                routing_decision.scores,

            verification_performed=
                False,

            verification_passed=
                None,

            verification_score=
                None,

            verification_reason=
                reason,

            escalated=
                False,

            attempts=[
                attempt
            ],

            total_estimated_cost_usd=
                generation_cost,

            total_latency_ms=
                generation_latency,
        )


    # ======================================================
    # BALANCED MODE
    #
    # Router
    #   ↓
    # Generate
    #   ↓
    # Risk policy
    #   ↓
    # verify only if needed
    #   ↓
    # escalate on failure
    # ======================================================

    async def _generate_balanced(
        self,
        prompt: str,
        routing_decision,
    ) -> GenerationResult:

        initial_tier = (
            routing_decision.selected_tier
        )


        tiers_to_try = (
            self._get_tiers_to_try(
                initial_tier
            )
        )


        attempts = []

        total_cost = 0.0
        total_latency = 0.0


        # --------------------------------------------------
        # Try initial tier and escalate when necessary
        # --------------------------------------------------

        for tier in tiers_to_try:

            model_config = (
                MODEL_REGISTRY[
                    tier
                ]
            )


            # ==============================================
            # Generate answer
            # ==============================================

            model_response = (
                await self.provider.send_request(

                    prompt=prompt,

                    model_config=
                        model_config,
                )
            )


            is_escalated_attempt = (
                tier != initial_tier
            )


            # ==============================================
            # Decide whether answer needs verification
            # ==============================================

            (
                should_verify,
                policy_reason,
            ) = (
                self.verification_policy
                .should_verify(

                    routing_decision=
                        routing_decision,

                    current_tier=
                        tier,

                    is_escalated=
                        is_escalated_attempt,
                )
            )


            # ==============================================
            # CASE 1:
            # Verification skipped
            # ==============================================

            if not should_verify:

                generation_cost = (
                    model_response
                    .estimated_cost_usd
                )


                generation_latency = (
                    model_response
                    .latency_ms
                )


                total_cost += (
                    generation_cost
                )


                total_latency += (
                    generation_latency
                )


                attempt = GenerationAttempt(

                    tier=
                        tier,

                    model_id=
                        model_response.model_id,

                    text=
                        model_response.text,

                    verification_performed=
                        False,

                    verification_passed=
                        None,

                    verification_score=
                        None,

                    verification_reason=
                        policy_reason,

                    input_tokens=
                        model_response.input_tokens,

                    output_tokens=
                        model_response.output_tokens,

                    thinking_tokens=
                        model_response.thinking_tokens,

                    generation_latency_ms=
                        generation_latency,

                    verification_latency_ms=
                        0.0,

                    generation_cost_usd=
                        generation_cost,

                    verification_cost_usd=
                        0.0,

                    total_attempt_cost_usd=
                        generation_cost,
                )


                attempts.append(
                    attempt
                )


                return GenerationResult(

                    text=
                        model_response.text,

                    mode=
                        "balanced",

                    initial_tier=
                        initial_tier,

                    final_tier=
                        tier,

                    model_id=
                        model_response.model_id,

                    routing_scores=
                        routing_decision.scores,

                    verification_performed=
                        False,

                    verification_passed=
                        None,

                    verification_score=
                        None,

                    verification_reason=
                        policy_reason,

                    escalated=(
                        tier != initial_tier
                    ),

                    attempts=
                        attempts,

                    total_estimated_cost_usd=
                        total_cost,

                    total_latency_ms=
                        total_latency,
                )


            # ==============================================
            # CASE 2:
            # Verification required
            # ==============================================

            verification = (
                await self.verifier.verify(

                    user_prompt=
                        prompt,

                    answer=
                        model_response.text,
                )
            )


            generation_cost = (
                model_response
                .estimated_cost_usd
            )


            verification_cost = (
                verification
                .estimated_cost_usd
            )


            attempt_cost = (
                generation_cost
                +
                verification_cost
            )


            generation_latency = (
                model_response
                .latency_ms
            )


            verification_latency = (
                verification
                .latency_ms
            )


            attempt_latency = (
                generation_latency
                +
                verification_latency
            )


            # ----------------------------------------------
            # Accounting
            # ----------------------------------------------

            total_cost += (
                attempt_cost
            )


            total_latency += (
                attempt_latency
            )


            # Safety check
            assert abs(

                attempt_cost
                -
                (
                    generation_cost
                    +
                    verification_cost
                )

            ) < 1e-12


            attempt = GenerationAttempt(

                tier=
                    tier,

                model_id=
                    model_response.model_id,

                text=
                    model_response.text,

                verification_performed=
                    True,

                verification_passed=
                    verification.passed,

                verification_score=
                    verification.score,

                verification_reason=
                    verification.reason,

                input_tokens=
                    model_response.input_tokens,

                output_tokens=
                    model_response.output_tokens,

                thinking_tokens=
                    model_response.thinking_tokens,

                generation_latency_ms=
                    generation_latency,

                verification_latency_ms=
                    verification_latency,

                generation_cost_usd=
                    generation_cost,

                verification_cost_usd=
                    verification_cost,

                total_attempt_cost_usd=
                    attempt_cost,
            )


            attempts.append(
                attempt
            )


            # ==============================================
            # Verification passed
            # ==============================================

            if verification.passed:

                return GenerationResult(

                    text=
                        model_response.text,

                    mode=
                        "balanced",

                    initial_tier=
                        initial_tier,

                    final_tier=
                        tier,

                    model_id=
                        model_response.model_id,

                    routing_scores=
                        routing_decision.scores,

                    verification_performed=
                        True,

                    verification_passed=
                        True,

                    verification_score=
                        verification.score,

                    verification_reason=
                        verification.reason,

                    escalated=(
                        tier != initial_tier
                    ),

                    attempts=
                        attempts,

                    total_estimated_cost_usd=
                        total_cost,

                    total_latency_ms=
                        total_latency,
                )


            # If verification failed,
            # the loop continues to the
            # next tier.


        # ==================================================
        # Safety fallback
        # ==================================================

        last_attempt = (
            attempts[-1]
        )


        return GenerationResult(

            text=
                last_attempt.text,

            mode=
                "balanced",

            initial_tier=
                initial_tier,

            final_tier=
                last_attempt.tier,

            model_id=
                last_attempt.model_id,

            routing_scores=
                routing_decision.scores,

            verification_performed=
                last_attempt
                .verification_performed,

            verification_passed=
                last_attempt
                .verification_passed,

            verification_score=
                last_attempt
                .verification_score,

            verification_reason=
                last_attempt
                .verification_reason,

            escalated=(
                last_attempt.tier
                !=
                initial_tier
            ),

            attempts=
                attempts,

            total_estimated_cost_usd=
                total_cost,

            total_latency_ms=
                total_latency,
        )


    # ======================================================
    # Public API
    # ======================================================

    async def generate(
        self,
        prompt: str,
        mode: str = "balanced",
    ) -> GenerationResult:

        # ----------------------------------------------
        # Validate prompt
        # ----------------------------------------------

        if not isinstance(
            prompt,
            str,
        ):

            raise TypeError(
                "Prompt must be a string."
            )


        prompt = (
            prompt.strip()
        )


        if not prompt:

            raise ValueError(
                "Prompt cannot be empty."
            )


        # ----------------------------------------------
        # Validate requested generation mode
        # ----------------------------------------------

        mode = (
            self._validate_mode(
                mode
            )
        )


        # ----------------------------------------------
        # ML routing happens for BOTH modes
        # ----------------------------------------------

        routing_decision = (
            self.router.route(
                prompt
            )
        )


        # ----------------------------------------------
        # Economy
        # ----------------------------------------------

        if mode == "economy":

            return await self._generate_economy(

                prompt=
                    prompt,

                routing_decision=
                    routing_decision,
            )


        # ----------------------------------------------
        # Balanced
        # ----------------------------------------------

        return await self._generate_balanced(

            prompt=
                prompt,

            routing_decision=
                routing_decision,
        )


    # ======================================================
    # Cleanup
    # ======================================================

    async def close(
        self,
    ):

        await self.provider.close()
from fastapi import (
    APIRouter,
    HTTPException,
    Request,
)


from app.api.schemas.generation import (
    GenerationRequest,
    GenerationResponse,
)

from app.core.exceptions import (
    ProviderError,
    ProviderQuotaError,
    ProviderUnavailableError,
)

import logging
import uuid

logger = logging.getLogger(
    __name__
)


router = APIRouter(
    tags=["Generation"]
)


@router.post(
    "/generate",
    response_model=GenerationResponse,
)
async def generate(
    body: GenerationRequest,
    request: Request,
):

    # ======================================================
    # Get the already-loaded GenerationService
    # ======================================================

    service = (
        request.app.state
        .generation_service
    )


    try:

        # ==================================================
        # Generate using economy / balanced mode
        # ==================================================

        result = await service.generate(

            prompt=
                body.prompt,

            mode=
                body.mode,
        )


# ==========================================================
# Create request ID independently of persistence
# ==========================================================

        request_id = str(
            uuid.uuid4()
        )


        # ==========================================================
        # Usage logging should NOT destroy a successful generation
        # ==========================================================

        usage_logger = (
            request.app.state
            .usage_logger
        )


        try:

            usage_logger.log_generation(

                result,

                request_id=
                    request_id,
            )


        except Exception:

            logger.exception(

                "Failed to persist generation analytics. "
                "request_id=%s",

                request_id,
            )


        # ==================================================
        # Convert our internal dataclasses
        # into API response structure
        # ==================================================

        attempts = [

            {

                "tier":
                    attempt.tier,

                "model_id":
                    attempt.model_id,

                "verification_performed":
                    attempt.verification_performed,

                "verification_passed":
                    attempt.verification_passed,

                "verification_score":
                    attempt.verification_score,

                "verification_reason":
                    attempt.verification_reason,

                "input_tokens":
                    attempt.input_tokens,

                "output_tokens":
                    attempt.output_tokens,

                "thinking_tokens":
                    attempt.thinking_tokens,

                "generation_latency_ms":
                    attempt.generation_latency_ms,

                "verification_latency_ms":
                    attempt.verification_latency_ms,

                "generation_cost_usd":
                    attempt.generation_cost_usd,

                "verification_cost_usd":
                    attempt.verification_cost_usd,

                "total_attempt_cost_usd":
                    attempt.total_attempt_cost_usd,
            }

            for attempt
            in result.attempts

        ]


        return GenerationResponse(

            request_id=
                request_id,

            text=
                result.text,

            mode=
                result.mode,

            initial_tier=
                result.initial_tier,

            final_tier=
                result.final_tier,

            model_id=
                result.model_id,

            routing_scores=
                result.routing_scores,

            verification_performed=
                result.verification_performed,

            verification_passed=
                result.verification_passed,

            verification_score=
                result.verification_score,

            verification_reason=
                result.verification_reason,

            escalated=
                result.escalated,

            attempts=
                attempts,

            total_estimated_cost_usd=
                result.total_estimated_cost_usd,

            total_latency_ms=
                result.total_latency_ms,
        )


    except ValueError as exc:

        raise HTTPException(

            status_code=400,

            detail=str(
                exc
            ),

        ) from exc


    except TypeError as exc:

        raise HTTPException(

            status_code=400,

            detail=str(
                exc
            ),

        ) from exc
    
    except ProviderQuotaError as exc:

        raise HTTPException(
            status_code=429,
            detail={
                "error": "provider_quota_exceeded",
                "message": str(exc),
            },
        ) from exc


    except ProviderUnavailableError as exc:

        raise HTTPException(
            status_code=503,
            detail={
                "error": "provider_unavailable",
                "message": str(exc),
            },
        ) from exc


    except ProviderError as exc:

        raise HTTPException(
            status_code=502,
            detail={
                "error": "provider_error",
                "message": str(exc),
            },
        ) from exc
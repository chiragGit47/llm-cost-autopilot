from typing import Literal

from pydantic import (
    BaseModel,
    Field,
)


# ==========================================================
# REQUEST
# ==========================================================

class GenerationRequest(BaseModel):

    prompt: str = Field(
        min_length=1,
        max_length=20000,
    )

    mode: Literal[
        "economy",
        "balanced",
    ] = "balanced"


# ==========================================================
# ONE GENERATION ATTEMPT
# ==========================================================

class GenerationAttemptResponse(BaseModel):

    tier: str

    model_id: str

    verification_performed: bool

    verification_passed: bool | None

    verification_score: float | None

    verification_reason: str | None

    input_tokens: int
    output_tokens: int
    thinking_tokens: int

    generation_latency_ms: float
    verification_latency_ms: float

    generation_cost_usd: float
    verification_cost_usd: float

    total_attempt_cost_usd: float


# ==========================================================
# FINAL RESPONSE
# ==========================================================

class GenerationResponse(BaseModel):

    request_id: str

    text: str

    mode: str

    initial_tier: str
    final_tier: str

    model_id: str

    routing_scores: dict[str, float]

    verification_performed: bool

    verification_passed: bool | None
    verification_score: float | None
    verification_reason: str | None

    escalated: bool

    attempts: list[
        GenerationAttemptResponse
    ]

    total_estimated_cost_usd: float
    total_latency_ms: float
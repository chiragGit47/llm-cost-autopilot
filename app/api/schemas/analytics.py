from pydantic import (
    BaseModel,
)


class UsageBucket(BaseModel):

    count: int
    percentage: float


class TokenSummary(BaseModel):

    input: int
    output: int
    thinking: int
    total: int


class AnalyticsSummaryResponse(BaseModel):

    total_requests: int

    total_cost_usd: float
    average_cost_usd: float

    average_latency_ms: float

    tokens: TokenSummary

    average_attempt_count: float

    verification_rate: float
    escalation_rate: float

    tier_usage: dict[
        str,
        UsageBucket,
    ]

    mode_usage: dict[
        str,
        UsageBucket,
    ]

from pydantic import (
    BaseModel,
)


class UsageBucket(
    BaseModel
):

    count: int

    percentage: float


class TokenSummary(
    BaseModel
):

    input: int

    output: int

    thinking: int

    total: int


class AnalyticsSummaryResponse(
    BaseModel
):

    total_requests: int


    total_cost_usd: float

    average_cost_usd: float

    average_latency_ms: float


    tokens: TokenSummary


    average_attempt_count: float


    verification_rate: float

    escalation_rate: float


    tier_usage: dict[
        str,
        UsageBucket,
    ]


    mode_usage: dict[
        str,
        UsageBucket,
    ]

class SavingsTokenSummary(BaseModel):

    input: int

    output: int

    thinking: int

    billable_output: int


class Tier3PricingSummary(BaseModel):

    input_cost_per_million: float

    output_cost_per_million: float


class SavingsSummaryResponse(BaseModel):

    baseline_type: str

    total_requests: int

    actual_cost_usd: float

    tier_3_equivalent_cost_usd: float

    estimated_savings_usd: float

    estimated_savings_percentage: float

    average_actual_cost_usd: float

    average_tier_3_equivalent_cost_usd: float

    generation_tokens: SavingsTokenSummary

    tier_3_pricing: Tier3PricingSummary

    note: str

class RequestHistoryItem(BaseModel):

    request_id: str

    created_at: str

    mode: str

    initial_tier: str
    final_tier: str

    model_id: str

    routing_scores: dict[
        str,
        float,
    ]

    verification_performed: bool

    verification_passed: bool | None

    verification_score: float | None

    escalated: bool

    attempt_count: int

    input_tokens: int
    output_tokens: int
    thinking_tokens: int

    total_cost_usd: float

    total_latency_ms: float
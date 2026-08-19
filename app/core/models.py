from dataclasses import dataclass


@dataclass
class ModelConfig:
    provider: str
    model_id: str
    input_cost_per_million: float
    output_cost_per_million: float
    quality_tier: str
    average_latency_ms: float | None = None


@dataclass
class ModelResponse:
    text: str
    model_id: str

    input_tokens: int
    output_tokens: int
    thinking_tokens: int

    latency_ms: float
    estimated_cost_usd: float


@dataclass
class RoutingDecision:
    selected_tier: str

    scores: dict[str, float]

    thresholds: dict[str, float]

    fallback_used: bool

    reason: str


@dataclass
class GenerationAttempt:
    tier: str
    model_id: str

    text: str

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

@dataclass
class GenerationResult:
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

    attempts: list[GenerationAttempt]

    total_estimated_cost_usd: float
    total_latency_ms: float

@dataclass
class VerificationResult:
    passed: bool
    score: float
    reason: str

    verifier_model_id: str

    latency_ms: float
    estimated_cost_usd: float


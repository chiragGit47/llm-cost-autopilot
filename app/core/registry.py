from app.core.models import ModelConfig


MODEL_REGISTRY = {

    "tier_1": ModelConfig(
        provider="gemini",
        model_id="gemini-3.1-flash-lite",
        input_cost_per_million=0.25,
        output_cost_per_million=1.50,
        quality_tier="tier_1",
    ),

    "tier_2": ModelConfig(
        provider="gemini",
        model_id="gemini-3.5-flash-lite",
        input_cost_per_million=0.30,
        output_cost_per_million=2.50,
        quality_tier="tier_2",
    ),

    "tier_3": ModelConfig(
        provider="gemini",
        model_id="gemini-3.7-flash",
        input_cost_per_million=0.75,
        output_cost_per_million=3.75,
        quality_tier="tier_3",
    ),
}
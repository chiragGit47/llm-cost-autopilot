from fastapi.testclient import TestClient


import app.main as main_module


from app.core.models import (
    GenerationAttempt,
    GenerationResult,
)


from app.core.exceptions import (
    ProviderError,
    ProviderQuotaError,
    ProviderUnavailableError,
)


# ==========================================================
# FAKE GENERATION RESULT
# ==========================================================

def make_fake_generation_result(
    mode: str = "economy",
) -> GenerationResult:

    attempt = GenerationAttempt(

        tier=
            "tier_1",

        model_id=
            "fake-gemini-tier-1",

        text=
            "Python is a programming language.",

        verification_performed=
            False,

        verification_passed=
            None,

        verification_score=
            None,

        verification_reason=
            "Mock API test.",

        input_tokens=
            5,

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
            mode,

        initial_tier=
            "tier_1",

        final_tier=
            "tier_1",

        model_id=
            attempt.model_id,

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
            "Mock API test.",

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


# ==========================================================
# FAKE USAGE LOGGER
#
# We don't want API tests touching:
#
# data/autopilot.db
# ==========================================================

class FakeUsageLoggerService:

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        pass


    def log_generation(
    self,
    result,
    request_id=None,
) -> str:

        return request_id


# ==========================================================
# SUCCESSFUL FAKE GENERATION SERVICE
# ==========================================================

class FakeSuccessGenerationService:

    async def generate(
        self,
        prompt: str,
        mode: str = "balanced",
    ):

        return (
            make_fake_generation_result(
                mode=mode
            )
        )


    async def close(
        self,
    ):
        pass


# ==========================================================
# QUOTA FAILURE
# ==========================================================

class FakeQuotaGenerationService:

    async def generate(
        self,
        prompt: str,
        mode: str = "balanced",
    ):

        raise ProviderQuotaError(
            "Gemini API quota or rate limit "
            "has been exceeded."
        )


    async def close(
        self,
    ):
        pass


# ==========================================================
# PROVIDER UNAVAILABLE
# ==========================================================

class FakeUnavailableGenerationService:

    async def generate(
        self,
        prompt: str,
        mode: str = "balanced",
    ):

        raise ProviderUnavailableError(
            "Gemini is temporarily unavailable."
        )


    async def close(
        self,
    ):
        pass


# ==========================================================
# GENERIC PROVIDER FAILURE
# ==========================================================

class FakeProviderErrorGenerationService:

    async def generate(
        self,
        prompt: str,
        mode: str = "balanced",
    ):

        raise ProviderError(
            "Gemini API request failed."
        )


    async def close(
        self,
    ):
        pass


# ==========================================================
# HELPER
#
# Replace the real startup services before TestClient
# runs the FastAPI lifespan.
# ==========================================================

def install_fake_services(
    monkeypatch,
    generation_service_class,
):

    monkeypatch.setattr(

        main_module,

        "GenerationService",

        generation_service_class,
    )


    monkeypatch.setattr(

        main_module,

        "UsageLoggerService",

        FakeUsageLoggerService,
    )


# ==========================================================
# HEALTH ENDPOINT
# ==========================================================

def test_health_endpoint(
    monkeypatch,
):

    install_fake_services(

        monkeypatch,

        FakeSuccessGenerationService,
    )


    with TestClient(
        main_module.app
    ) as client:

        response = client.get(
            "/health"
        )


    assert (
        response.status_code
        ==
        200
    )


    body = response.json()


    assert (
        body["status"]
        ==
        "ok"
    )


    assert (
        body["service"]
        ==
        "llm-cost-autopilot"
    )


# ==========================================================
# SUCCESSFUL /generate
# ==========================================================

def test_generate_success(
    monkeypatch,
):

    install_fake_services(

        monkeypatch,

        FakeSuccessGenerationService,
    )


    with TestClient(
        main_module.app
    ) as client:

        response = client.post(

            "/generate",

            json={

                "prompt":
                    "What is Python?",

                "mode":
                    "economy",
            },
        )


    assert (
        response.status_code
        ==
        200
    )


    body = response.json()


    assert (
        body["request_id"]
        ==
        "test-request-id-123"
    )


    assert (
        body["mode"]
        ==
        "economy"
    )


    assert (
        body["initial_tier"]
        ==
        "tier_1"
    )


    assert (
        body["final_tier"]
        ==
        "tier_1"
    )


    assert (
        body["model_id"]
        ==
        "fake-gemini-tier-1"
    )


    assert (
        body[
            "verification_performed"
        ]
        is False
    )


    assert (
        body["escalated"]
        is False
    )


    assert (
        body[
            "total_estimated_cost_usd"
        ]
        ==
        0.0001
    )


    assert (
        len(
            body["attempts"]
        )
        ==
        1
    )


# ==========================================================
# INVALID MODE
#
# This should fail during Pydantic validation,
# before GenerationService is called.
# ==========================================================

def test_invalid_generation_mode(
    monkeypatch,
):

    install_fake_services(

        monkeypatch,

        FakeSuccessGenerationService,
    )


    with TestClient(
        main_module.app
    ) as client:

        response = client.post(

            "/generate",

            json={

                "prompt":
                    "What is Python?",

                "mode":
                    "super_turbo_mode",
            },
        )


    assert (
        response.status_code
        ==
        422
    )


# ==========================================================
# EMPTY PROMPT
# ==========================================================

def test_empty_prompt_is_rejected(
    monkeypatch,
):

    install_fake_services(

        monkeypatch,

        FakeSuccessGenerationService,
    )


    with TestClient(
        main_module.app
    ) as client:

        response = client.post(

            "/generate",

            json={

                "prompt":
                    "",

                "mode":
                    "economy",
            },
        )


    assert (
        response.status_code
        ==
        422
    )


# ==========================================================
# QUOTA ERROR -> HTTP 429
# ==========================================================

def test_provider_quota_error_returns_429(
    monkeypatch,
):

    install_fake_services(

        monkeypatch,

        FakeQuotaGenerationService,
    )


    with TestClient(

        main_module.app,

        raise_server_exceptions=
            False,

    ) as client:

        response = client.post(

            "/generate",

            json={

                "prompt":
                    "Explain Python.",

                "mode":
                    "economy",
            },
        )


    assert (
        response.status_code
        ==
        429
    )


    body = response.json()


    assert (
        body["detail"]["error"]
        ==
        "provider_quota_exceeded"
    )


# ==========================================================
# PROVIDER UNAVAILABLE -> HTTP 503
# ==========================================================

def test_provider_unavailable_returns_503(
    monkeypatch,
):

    install_fake_services(

        monkeypatch,

        FakeUnavailableGenerationService,
    )


    with TestClient(

        main_module.app,

        raise_server_exceptions=
            False,

    ) as client:

        response = client.post(

            "/generate",

            json={

                "prompt":
                    "Explain Python.",

                "mode":
                    "balanced",
            },
        )


    assert (
        response.status_code
        ==
        503
    )


    body = response.json()


    assert (
        body["detail"]["error"]
        ==
        "provider_unavailable"
    )


# ==========================================================
# GENERIC PROVIDER ERROR -> HTTP 502
# ==========================================================

def test_generic_provider_error_returns_502(
    monkeypatch,
):

    install_fake_services(

        monkeypatch,

        FakeProviderErrorGenerationService,
    )


    with TestClient(

        main_module.app,

        raise_server_exceptions=
            False,

    ) as client:

        response = client.post(

            "/generate",

            json={

                "prompt":
                    "Explain Python.",

                "mode":
                    "economy",
            },
        )


    assert (
        response.status_code
        ==
        502
    )


    body = response.json()


    assert (
        body["detail"]["error"]
        ==
        "provider_error"
    )

class FakeFailingUsageLoggerService:

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        pass


    def log_generation(
        self,
        result,
        request_id=None,
    ):

        raise RuntimeError(
            "Fake database failure."
        )
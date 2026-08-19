from contextlib import (
    asynccontextmanager,
)


from fastapi import (
    FastAPI,
)


from app.api.routes.health import (
    router as health_router,
)

from app.api.routes.generation import (
    router as generation_router,
)

from app.services.generation_service import (
    GenerationService,
)

from app.services.usage_logger_service import (
    UsageLoggerService,
)

from app.api.routes.analytics import (
    router as analytics_router,
)

from app.core.config import (
    get_settings,
)

# ==========================================================
# APPLICATION LIFESPAN
# ==========================================================

@asynccontextmanager
async def lifespan(
    app: FastAPI,
):

    # ======================================================
    # STARTUP
    # ======================================================

    print(
        "Starting LLM Cost Autopilot..."
    )


    generation_service = (
        GenerationService()
    )


    app.state.generation_service = (
        generation_service
    )


    print(
        "GenerationService loaded."
    )

    usage_logger = (
    UsageLoggerService(

        database_path=
            settings.database_path

    )
)


    app.state.usage_logger = (
        usage_logger
    )


    # ------------------------------------------------------
    # Application runs while execution is paused at yield.
    # ------------------------------------------------------

    yield


    # ======================================================
    # SHUTDOWN
    # ======================================================

    print(
        "Shutting down LLM Cost Autopilot..."
    )


    await generation_service.close()


    print(
        "GenerationService closed."
    )


# ==========================================================
# FASTAPI APPLICATION
# ==========================================================
settings = get_settings()

app = FastAPI(

    title=
        settings.app_name,

    version=
        settings.app_version,

    description=(
        "Cost-aware LLM routing API using "
        "machine-learning model selection, "
        "selective quality verification, "
        "and automatic escalation."
    ),

    lifespan=
        lifespan,
)


# ==========================================================
# ROUTES
# ==========================================================

app.include_router(
    health_router
)


app.include_router(
    generation_router
)

app.include_router(
    analytics_router
)



from app.api.schemas.analytics import (
    AnalyticsSummaryResponse,
)



from app.api.schemas.analytics import (
    AnalyticsSummaryResponse,
    SavingsSummaryResponse,
    RequestHistoryItem,
)

from fastapi import (
    APIRouter,
    Request,
    Query,
)

router = APIRouter(

    prefix="/analytics",

    tags=[
        "Analytics"
    ],
)


@router.get(
    "/summary",

    response_model=
        AnalyticsSummaryResponse,
)
async def analytics_summary(
    request: Request,
):

    usage_logger = (

        request.app.state
        .usage_logger

    )


    summary = (
        usage_logger
        .get_summary()
    )


    return (
        AnalyticsSummaryResponse(
            **summary
        )
    )

@router.get(
    "/savings",
    response_model=
        SavingsSummaryResponse,
)
async def analytics_savings(
    request: Request,
):

    usage_logger = (

        request.app.state
        .usage_logger

    )


    savings = (

        usage_logger
        .get_savings_summary()

    )


    return SavingsSummaryResponse(
        **savings
    )

@router.get(
    "/requests",
    response_model=
        list[RequestHistoryItem],
)
async def analytics_requests(

    request: Request,

    limit: int = Query(
        default=50,
        ge=1,
        le=100,
    ),
):

    usage_logger = (

        request.app.state
        .usage_logger

    )


    history = (

        usage_logger
        .get_request_history(
            limit=limit
        )

    )


    return [

        RequestHistoryItem(
            **item
        )

        for item
        in history

    ]
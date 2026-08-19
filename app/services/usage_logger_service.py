import json
import sqlite3
import uuid

from datetime import (
    datetime,
    timezone,
)

from pathlib import Path


from app.core.models import (
    GenerationResult,
)

from app.core.registry import (
    MODEL_REGISTRY,
)


class UsageLoggerService:

    # ======================================================
    # INITIALIZATION
    # ======================================================

    def __init__(
        self,
        database_path: str | Path | None = None,
    ):

        project_root = (
            Path(__file__)
            .resolve()
            .parents[2]
        )

        if database_path is None:

            database_path = (
                project_root
                / "data"
                / "autopilot.db"
            )

        self.database_path = Path(
            database_path
        )

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._create_tables()


    # ======================================================
    # DATABASE CONNECTION
    # ======================================================

    def _connect(
        self,
    ):

        connection = sqlite3.connect(
            self.database_path
        )

        connection.row_factory = (
            sqlite3.Row
        )

        return connection


    # ======================================================
    # CREATE DATABASE TABLES
    # ======================================================

    def _create_tables(
        self,
    ):

        with self._connect() as connection:

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS generation_logs (

                    request_id TEXT PRIMARY KEY,

                    created_at TEXT NOT NULL,

                    mode TEXT NOT NULL,

                    initial_tier TEXT NOT NULL,

                    final_tier TEXT NOT NULL,

                    model_id TEXT NOT NULL,

                    routing_scores TEXT NOT NULL,

                    verification_performed INTEGER NOT NULL,

                    verification_passed INTEGER,

                    verification_score REAL,

                    escalated INTEGER NOT NULL,

                    attempt_count INTEGER NOT NULL,

                    input_tokens INTEGER NOT NULL,

                    output_tokens INTEGER NOT NULL,

                    thinking_tokens INTEGER NOT NULL,

                    total_cost_usd REAL NOT NULL,

                    total_latency_ms REAL NOT NULL
                )
                """
            )

            connection.commit()


    # ======================================================
    # LOG ONE GENERATION
    # ======================================================

    def log_generation(
    self,
    result: GenerationResult,
    request_id: str | None = None,
    ) -> str:

        if request_id is None:

            request_id = str(
                uuid.uuid4()
            )

        created_at = (
            datetime.now(
                timezone.utc
            ).isoformat()
        )


        # --------------------------------------------------
        # Aggregate token usage across attempts
        # --------------------------------------------------

        total_input_tokens = sum(

            attempt.input_tokens

            for attempt
            in result.attempts
        )


        total_output_tokens = sum(

            attempt.output_tokens

            for attempt
            in result.attempts
        )


        total_thinking_tokens = sum(

            attempt.thinking_tokens

            for attempt
            in result.attempts
        )


        routing_scores_json = json.dumps(
            result.routing_scores
        )


        with self._connect() as connection:

            connection.execute(
                """
                INSERT INTO generation_logs (

                    request_id,
                    created_at,

                    mode,

                    initial_tier,
                    final_tier,

                    model_id,

                    routing_scores,

                    verification_performed,
                    verification_passed,
                    verification_score,

                    escalated,

                    attempt_count,

                    input_tokens,
                    output_tokens,
                    thinking_tokens,

                    total_cost_usd,
                    total_latency_ms
                )

                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,

                (
                    request_id,

                    created_at,

                    result.mode,

                    result.initial_tier,
                    result.final_tier,

                    result.model_id,

                    routing_scores_json,

                    int(
                        result.verification_performed
                    ),

                    (
                        None

                        if result.verification_passed
                        is None

                        else int(
                            result.verification_passed
                        )
                    ),

                    result.verification_score,

                    int(
                        result.escalated
                    ),

                    len(
                        result.attempts
                    ),

                    total_input_tokens,

                    total_output_tokens,

                    total_thinking_tokens,

                    result.total_estimated_cost_usd,

                    result.total_latency_ms,
                ),
            )

            connection.commit()


        return request_id


    # ======================================================
    # BASIC ANALYTICS SUMMARY
    # ======================================================

    def get_summary(
        self,
    ) -> dict:

        with self._connect() as connection:

            # --------------------------------------------------
            # Overall statistics
            # --------------------------------------------------

            overall = connection.execute(
                """
                SELECT

                    COUNT(*) AS total_requests,

                    COALESCE(
                        SUM(total_cost_usd),
                        0
                    ) AS total_cost_usd,

                    COALESCE(
                        AVG(total_cost_usd),
                        0
                    ) AS average_cost_usd,

                    COALESCE(
                        AVG(total_latency_ms),
                        0
                    ) AS average_latency_ms,

                    COALESCE(
                        SUM(input_tokens),
                        0
                    ) AS total_input_tokens,

                    COALESCE(
                        SUM(output_tokens),
                        0
                    ) AS total_output_tokens,

                    COALESCE(
                        SUM(thinking_tokens),
                        0
                    ) AS total_thinking_tokens,

                    COALESCE(
                        AVG(attempt_count),
                        0
                    ) AS average_attempt_count,

                    COALESCE(
                        AVG(verification_performed),
                        0
                    ) AS verification_rate,

                    COALESCE(
                        AVG(escalated),
                        0
                    ) AS escalation_rate

                FROM generation_logs
                """
            ).fetchone()


            # --------------------------------------------------
            # Tier usage
            # --------------------------------------------------

            tier_rows = connection.execute(
                """
                SELECT

                    final_tier,

                    COUNT(*) AS request_count

                FROM generation_logs

                GROUP BY final_tier
                """
            ).fetchall()


            # --------------------------------------------------
            # Mode usage
            # --------------------------------------------------

            mode_rows = connection.execute(
                """
                SELECT

                    mode,

                    COUNT(*) AS request_count

                FROM generation_logs

                GROUP BY mode
                """
            ).fetchall()


        total_requests = int(
            overall["total_requests"]
        )


        # ==================================================
        # Tier distribution
        # ==================================================

        tier_usage = {

            "tier_1": {
                "count": 0,
                "percentage": 0.0,
            },

            "tier_2": {
                "count": 0,
                "percentage": 0.0,
            },

            "tier_3": {
                "count": 0,
                "percentage": 0.0,
            },
        }


        for row in tier_rows:

            tier = row[
                "final_tier"
            ]

            count = int(
                row[
                    "request_count"
                ]
            )

            percentage = (

                count
                /
                total_requests
                *
                100

                if total_requests > 0

                else 0.0
            )


            tier_usage[tier] = {

                "count":
                    count,

                "percentage":
                    percentage,
            }


        # ==================================================
        # Mode distribution
        # ==================================================

        mode_usage = {}


        for row in mode_rows:

            mode = row[
                "mode"
            ]

            count = int(
                row[
                    "request_count"
                ]
            )

            percentage = (

                count
                /
                total_requests
                *
                100

                if total_requests > 0

                else 0.0
            )


            mode_usage[mode] = {

                "count":
                    count,

                "percentage":
                    percentage,
            }


        # ==================================================
        # Token totals
        # ==================================================

        input_tokens = int(
            overall[
                "total_input_tokens"
            ]
        )

        output_tokens = int(
            overall[
                "total_output_tokens"
            ]
        )

        thinking_tokens = int(
            overall[
                "total_thinking_tokens"
            ]
        )


        total_tokens = (

            input_tokens
            +
            output_tokens
            +
            thinking_tokens
        )


        # ==================================================
        # Return summary
        # ==================================================

        return {

            "total_requests":
                total_requests,

            "total_cost_usd":
                float(
                    overall[
                        "total_cost_usd"
                    ]
                ),

            "average_cost_usd":
                float(
                    overall[
                        "average_cost_usd"
                    ]
                ),

            "average_latency_ms":
                float(
                    overall[
                        "average_latency_ms"
                    ]
                ),

            "tokens": {

                "input":
                    input_tokens,

                "output":
                    output_tokens,

                "thinking":
                    thinking_tokens,

                "total":
                    total_tokens,
            },

            "average_attempt_count":
                float(
                    overall[
                        "average_attempt_count"
                    ]
                ),

            "verification_rate":
                float(
                    overall[
                        "verification_rate"
                    ]
                )
                * 100,

            "escalation_rate":
                float(
                    overall[
                        "escalation_rate"
                    ]
                )
                * 100,

            "tier_usage":
                tier_usage,

            "mode_usage":
                mode_usage,
        }
    

    def get_request_history(
        self,
        limit: int = 50,
        ) -> list[dict]:

            # ======================================================
            # Validate limit
            # ======================================================

            if limit < 1:
                raise ValueError(
                    "Limit must be at least 1."
                )

            if limit > 100:
                limit = 100


            # ======================================================
            # Fetch newest requests first
            # ======================================================

            with self._connect() as connection:

                rows = connection.execute(
                    """
                    SELECT

                        request_id,
                        created_at,

                        mode,

                        initial_tier,
                        final_tier,

                        model_id,

                        routing_scores,

                        verification_performed,
                        verification_passed,
                        verification_score,

                        escalated,

                        attempt_count,

                        input_tokens,
                        output_tokens,
                        thinking_tokens,

                        total_cost_usd,
                        total_latency_ms

                    FROM generation_logs

                    ORDER BY created_at DESC

                    LIMIT ?
                    """,

                    (
                        limit,
                    ),
                ).fetchall()


            # ======================================================
            # Convert SQLite rows into normal dictionaries
            # ======================================================

            history = []


            for row in rows:

                routing_scores = json.loads(
                    row["routing_scores"]
                )


                history.append(
                    {

                        "request_id":
                            row["request_id"],

                        "created_at":
                            row["created_at"],

                        "mode":
                            row["mode"],

                        "initial_tier":
                            row["initial_tier"],

                        "final_tier":
                            row["final_tier"],

                        "model_id":
                            row["model_id"],

                        "routing_scores":
                            routing_scores,

                        "verification_performed":
                            bool(
                                row[
                                    "verification_performed"
                                ]
                            ),

                        "verification_passed":
                            (
                                None

                                if row[
                                    "verification_passed"
                                ] is None

                                else bool(
                                    row[
                                        "verification_passed"
                                    ]
                                )
                            ),

                        "verification_score":
                            row[
                                "verification_score"
                            ],

                        "escalated":
                            bool(
                                row[
                                    "escalated"
                                ]
                            ),

                        "attempt_count":
                            row[
                                "attempt_count"
                            ],

                        "input_tokens":
                            row[
                                "input_tokens"
                            ],

                        "output_tokens":
                            row[
                                "output_tokens"
                            ],

                        "thinking_tokens":
                            row[
                                "thinking_tokens"
                            ],

                        "total_cost_usd":
                            row[
                                "total_cost_usd"
                            ],

                        "total_latency_ms":
                            row[
                                "total_latency_ms"
                            ],
                    }
                )


            return history


        # ======================================================
        # COST SAVINGS ANALYTICS
        # ======================================================

    def get_savings_summary(
            self,
        ) -> dict:

            # --------------------------------------------------
            # Tier 3 pricing
            # --------------------------------------------------

            tier_3 = MODEL_REGISTRY[
                "tier_3"
            ]


            with self._connect() as connection:

                totals = connection.execute(
                    """
                    SELECT

                        COUNT(*) AS total_requests,

                        COALESCE(
                            SUM(total_cost_usd),
                            0
                        ) AS actual_cost_usd,

                        COALESCE(
                            SUM(input_tokens),
                            0
                        ) AS input_tokens,

                        COALESCE(
                            SUM(output_tokens),
                            0
                        ) AS output_tokens,

                        COALESCE(
                            SUM(thinking_tokens),
                            0
                        ) AS thinking_tokens

                    FROM generation_logs
                    """
                ).fetchone()


            total_requests = int(
                totals[
                    "total_requests"
                ]
            )


            actual_cost = float(
                totals[
                    "actual_cost_usd"
                ]
            )


            input_tokens = int(
                totals[
                    "input_tokens"
                ]
            )


            output_tokens = int(
                totals[
                    "output_tokens"
                ]
            )


            thinking_tokens = int(
                totals[
                    "thinking_tokens"
                ]
            )


            # ==================================================
            # Tier-3 token-equivalent input cost
            # ==================================================

            tier_3_input_cost = (

                input_tokens

                /
                1_000_000

                *
                tier_3.input_cost_per_million
            )


            # ==================================================
            # Output + thinking token equivalent
            # ==================================================

            billable_output_tokens = (

                output_tokens
                +
                thinking_tokens
            )


            tier_3_output_cost = (

                billable_output_tokens

                /
                1_000_000

                *
                tier_3.output_cost_per_million
            )


            # ==================================================
            # Baseline
            # ==================================================

            tier_3_equivalent_cost = (

                tier_3_input_cost
                +
                tier_3_output_cost
            )


            # ==================================================
            # Estimated savings
            # ==================================================

            estimated_savings = (

                tier_3_equivalent_cost
                -
                actual_cost
            )


            if tier_3_equivalent_cost > 0:

                estimated_savings_percentage = (

                    estimated_savings

                    /
                    tier_3_equivalent_cost

                    *
                    100
                )

            else:

                estimated_savings_percentage = 0.0


            # ==================================================
            # Per-request averages
            # ==================================================

            if total_requests > 0:

                average_actual_cost = (

                    actual_cost
                    /
                    total_requests
                )


                average_tier_3_equivalent_cost = (

                    tier_3_equivalent_cost
                    /
                    total_requests
                )

            else:

                average_actual_cost = 0.0

                average_tier_3_equivalent_cost = 0.0


            # ==================================================
            # Return savings report
            # ==================================================

            return {

                "baseline_type":
                    "tier_3_token_equivalent",

                "total_requests":
                    total_requests,

                "actual_cost_usd":
                    actual_cost,

                "tier_3_equivalent_cost_usd":
                    tier_3_equivalent_cost,

                "estimated_savings_usd":
                    estimated_savings,

                "estimated_savings_percentage":
                    estimated_savings_percentage,

                "average_actual_cost_usd":
                    average_actual_cost,

                "average_tier_3_equivalent_cost_usd":
                    average_tier_3_equivalent_cost,

                "generation_tokens": {

                    "input":
                        input_tokens,

                    "output":
                        output_tokens,

                    "thinking":
                        thinking_tokens,

                    "billable_output":
                        billable_output_tokens,
                },

                "tier_3_pricing": {

                    "input_cost_per_million":
                        tier_3
                        .input_cost_per_million,

                    "output_cost_per_million":
                        tier_3
                        .output_cost_per_million,
                },

                "note": (
                    "This is a token-equivalent Tier-3 "
                    "baseline, not an exact simulation "
                    "of always using Tier 3. Different "
                    "models may generate different "
                    "numbers of tokens."
                ),
            }
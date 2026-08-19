import json
from pathlib import Path

import joblib

from app.core.models import RoutingDecision


class RouterService:

    def __init__(
        self,
        artifact_dir: str | Path | None = None,
    ):

        # ------------------------------------------
        # Find project root
        # ------------------------------------------

        project_root = (
            Path(__file__)
            .resolve()
            .parents[2]
        )


        if artifact_dir is None:

            artifact_dir = (
                project_root
                / "artifacts"
                / "router"
            )


        self.artifact_dir = Path(
            artifact_dir
        )


        # ------------------------------------------
        # Load configuration
        # ------------------------------------------

        config_path = (
            self.artifact_dir
            / "router_config.json"
        )


        self._require_file(
            config_path
        )


        with open(
            config_path,
            "r",
            encoding="utf-8",
        ) as file:

            self.config = json.load(
                file
            )


        # ------------------------------------------
        # Load feature transformer
        # ------------------------------------------

        transformer_path = (
            self.artifact_dir
            / "feature_transformer.joblib"
        )


        self._require_file(
            transformer_path
        )


        self.feature_transformer = (
            joblib.load(
                transformer_path
            )
        )


        # ------------------------------------------
        # Load success predictors
        # ------------------------------------------

        self.predictors = {}


        for tier in [
            "tier_1",
            "tier_2",
            "tier_3",
        ]:

            model_path = (
                self.artifact_dir
                / f"{tier}_predictor.joblib"
            )


            self._require_file(
                model_path
            )


            self.predictors[tier] = (
                joblib.load(
                    model_path
                )
            )


    def _require_file(
        self,
        path: Path,
    ) -> None:

        if not path.exists():

            raise FileNotFoundError(
                f"Required router artifact "
                f"not found: {path}"
            )


    def _get_pass_score(
        self,
        tier: str,
        transformed_prompt,
    ) -> float:

        model = self.predictors[
            tier
        ]


        probabilities = (
            model.predict_proba(
                transformed_prompt
            )
        )


        classes = list(
            model.classes_
        )


        pass_index = (
            classes.index(1)
        )


        pass_score = (
            probabilities[
                0,
                pass_index
            ]
        )


        return float(
            pass_score
        )


    def route(
        self,
        prompt: str,
    ) -> RoutingDecision:

        # ------------------------------------------
        # Validate input
        # ------------------------------------------

        if not isinstance(
            prompt,
            str,
        ):

            raise TypeError(
                "Prompt must be a string."
            )


        prompt = prompt.strip()


        if not prompt:

            raise ValueError(
                "Prompt cannot be empty."
            )


        # ------------------------------------------
        # Convert prompt into SAME features
        # used during training
        # ------------------------------------------

        transformed_prompt = (
            self.feature_transformer
            .transform(
                [prompt]
            )
        )


        # ------------------------------------------
        # Predict model success scores
        # ------------------------------------------

        scores = {

            tier:
                self._get_pass_score(
                    tier,
                    transformed_prompt,
                )

            for tier in [
                "tier_1",
                "tier_2",
                "tier_3",
            ]
        }


        # ------------------------------------------
        # Load frozen routing thresholds
        # ------------------------------------------

        tier_1_threshold = float(
            self.config[
                "tier_1_threshold"
            ]
        )


        tier_2_threshold = float(
            self.config[
                "tier_2_threshold"
            ]
        )


        thresholds = {
            "tier_1":
                tier_1_threshold,

            "tier_2":
                tier_2_threshold,
        }


        # ------------------------------------------
        # Routing policy
        # ------------------------------------------

        if (
            scores["tier_1"]
            >=
            tier_1_threshold
        ):

            selected_tier = (
                "tier_1"
            )

            fallback_used = False

            reason = (
                "Tier 1 predicted success "
                "score passed its threshold."
            )


        elif (
            scores["tier_2"]
            >=
            tier_2_threshold
        ):

            selected_tier = (
                "tier_2"
            )

            fallback_used = False

            reason = (
                "Tier 1 did not meet its "
                "threshold, but Tier 2 did."
            )


        else:

            selected_tier = (
                self.config[
                    "fallback_tier"
                ]
            )

            fallback_used = True

            reason = (
                "Neither Tier 1 nor Tier 2 "
                "met the required routing "
                "threshold, so the strongest "
                "tier was selected."
            )


        return RoutingDecision(

            selected_tier=
                selected_tier,

            scores=scores,

            thresholds=
                thresholds,

            fallback_used=
                fallback_used,

            reason=reason,
        )
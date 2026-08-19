
import time 

from app.core.config import (
    get_settings,
)
from google import genai
from google.genai import types

from google.genai import errors

from app.core.exceptions import (
    ProviderError,
    ProviderQuotaError,
    ProviderUnavailableError,
)



from app.core.exceptions import (
    ProviderError,
    ProviderQuotaError,
    ProviderUnavailableError,
)

from app.core.models import ModelConfig, ModelResponse
from app.providers.base import BaseProvider

settings = get_settings()



class GeminiProvider(BaseProvider):

    def __init__(self):

        settings = get_settings()

        self.api_key = (
            settings.gemini_api_key
        )


        # 3. Validate that it exists
        if not self.api_key:

            raise ValueError(
                "GEMINI_API_KEY was not found "
                "in the environment."
            )


        # 4. Only NOW create the Gemini client
        self.client = genai.Client(

            api_key=self.api_key,

            http_options=types.HttpOptions(

                retry_options=
                    types.HttpRetryOptions(

                        attempts=5,

                        initial_delay=1.0,

                        max_delay=16.0,

                        exp_base=2,

                        jitter=1.0,

                        http_status_codes=[
                            429,
                            500,
                            502,
                            503,
                            504,
                        ],
                    )
            ),
        )


        # 5. Async version of the same client
        self.async_client = (
            self.client.aio
        )

    async def send_request(
    self,
    prompt: str,
    model_config: ModelConfig
    ) -> ModelResponse:

        start_time = time.perf_counter()

        chat = self.async_client.chats.create(
            model=model_config.model_id
        )

        try:

            response = await chat.send_message(
                prompt
            )


        except errors.APIError as exc:

            # ==============================================
            # 429
            #
            # Rate limit / quota exhaustion
            # ==============================================

            if exc.code == 429:

                raise ProviderQuotaError(
                    "Gemini API quota or rate limit "
                    "has been exceeded."
                ) from exc


            # ==============================================
            # Temporary provider/server failures
            # ==============================================

            if exc.code in {
                500,
                502,
                503,
                504,
            }:

                raise ProviderUnavailableError(
                    "Gemini is temporarily unavailable."
                ) from exc


            # ==============================================
            # Other Gemini API failure
            # ==============================================

            raise ProviderError(
                "Gemini API request failed."
            ) from exc

        end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000

        usage = response.usage_metadata

        input_tokens = usage.prompt_token_count or 0
        output_tokens = usage.candidates_token_count or 0
        thinking_tokens = usage.thoughts_token_count or 0

        billable_output_tokens = (
            output_tokens + thinking_tokens
        )

        input_cost = (
            input_tokens / 1_000_000
        ) * model_config.input_cost_per_million

        output_cost = (
            billable_output_tokens / 1_000_000
        ) * model_config.output_cost_per_million

        estimated_cost = input_cost + output_cost

        return ModelResponse(
            text=response.text or "",
            model_id=model_config.model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            thinking_tokens=thinking_tokens,
            latency_ms=latency_ms,
            estimated_cost_usd=estimated_cost,
        )
    
    async def close(self):
        await self.async_client.aclose()
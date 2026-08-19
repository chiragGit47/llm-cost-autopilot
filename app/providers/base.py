from abc import ABC, abstractmethod

from app.core.models import ModelConfig, ModelResponse

class BaseProvider(ABC):

     @abstractmethod
     async def send_request(
        self,
        prompt: str,
        model_config: ModelConfig
    ) -> ModelResponse:
        pass
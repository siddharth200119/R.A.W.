from typing import List, Union, Dict, Optional, AsyncGenerator
from abc import ABC, abstractmethod
from RAW.models import Image, LLMOutput, LLMCapability, Message
from pydantic import BaseModel
from RAW.utils import Logger

class LLMOptions(BaseModel):
    ...

class LLM(ABC):
    def __init__(self, logger: Logger, model: str):
        self.capabilities: List[LLMCapability] = []
        self.logger = logger
        self.model = model
        pass
    
    @abstractmethod
    def generate(self, prompt: str, images: List[Image] = [], think: bool = False, format: Optional[Union[str, Dict]] = None, stream=False) -> Union[AsyncGenerator[LLMOutput, None], LLMOutput]:
        ...

    @abstractmethod
    def chat(self, message: Message, think: bool = False, stream: bool = False, format: Optional[Union[str, Dict]] = None) -> Union[AsyncGenerator[Message, None], Message]:
        ...

    def _validate_thinking(self, think: bool) -> bool:
        if think and LLMCapability.THINKING not in self.capabilities:
            self.logger.warning_sync(f"The chosen model {self.model} does not support thinking, passing think=false to prevent errors")
            return False
        return think

    def _validate_images(self, images: List[Image]) -> List[Image]:
        if images and LLMCapability.VISION not in self.capabilities:
            self.logger.warning_sync(f"The chosen model {self.model} does not support vision, skipping images")
            return []
        return images
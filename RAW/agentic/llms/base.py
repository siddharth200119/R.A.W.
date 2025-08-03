from abc import ABC, abstractmethod
from typing import List, Literal, Optional, Dict, Union
import uuid

from RAW.utils import Logger
from RAW.models import Image, Message, Tool

class LLM(ABC):
    @abstractmethod
    def __init__(self, logger: Logger,):
        super().__init__()
        self.logger = logger if logger else Logger()
        self.capabilities: List[Literal['thinking', 'tools', 'vision', 'embedding']] = []

    @abstractmethod
    def generate(self, prompt: str, images: List[Image] = [], format: Optional[Union[str, Dict]] = None, think: bool = False, stream: bool = False):
        ...

    @abstractmethod
    def chat(self, messages: List[Message], think: bool = False, stream: bool = False, tools: List[Tool] = []):
        ...

    @abstractmethod
    def embed(self, content):
        ...

    def validate_think(self, think: bool) -> bool:
        if 'thinking' in self.capabilities:
            return think
        elif think:
            self.logger.warning(content="The LLM selected does not support thinking automatically switching to non thinking mode")
            return False
        else:
            return False
        
    def validate_image(self, image: Image) -> Optional[Image]:
        if('vision' in self.capabilities):
            return image
        else:
            self.logger.warning(content="The LLM selected does not support vision automatically skipping images")
            return
from pydantic import BaseModel
from typing import Literal, Union, List
from .llm_output import LLMOutput
from .image import Image

class Message(BaseModel):
    role: Literal['system', 'tool', 'user', 'assistant']
    content: Union[LLMOutput, str]
    images: List[Image] = None
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from .image import Image

class ToolCall(BaseModel):
    function_name: str
    arguments: Dict[str, Any]

class LLMOutput(BaseModel):
    thought: Optional[str] = None
    content: Optional[str] = None
    images: List[Image] = None
    tool_calls: Optional[List[ToolCall]] = None
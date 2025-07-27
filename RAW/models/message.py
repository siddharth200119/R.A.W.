from pydantic import BaseModel
from typing import Literal, List, Optional, Dict, Any

from RAW.models.image import Image
from RAW.models.tool import Tool

class toolCall(BaseModel):
    tool: Tool
    arguments: Dict[str, Any]

class Message(BaseModel):
    role: Literal['system', 'user', 'assistant', 'tool']
    thought: Optional[str] = None
    content: str
    images: List[Image] = []
    tool_calls: Optional[List[toolCall]] = None
    tool_call_id: Optional[str] = None
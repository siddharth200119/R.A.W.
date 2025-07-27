from pydantic import BaseModel
from typing import Callable, Literal, List, Union, Awaitable, Generator, AsyncGenerator, Any

class ToolParam(BaseModel):
    name: str
    type: Literal['string', 'number', 'integer', 'boolean', 'array', 'object', 'null']
    description: str
    required: bool

class Tool(BaseModel):
    name: str
    description: str
    parameters: List[ToolParam]
    function: Union[
        Callable[..., str],
        Callable[..., Awaitable[str]],
        Callable[..., Generator[Any, None, str]],
        Callable[..., AsyncGenerator[str, None]],
    ]
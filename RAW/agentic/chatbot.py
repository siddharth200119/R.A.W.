from RAW.llms import LLM
from RAW.models import LLMCapability, Message
from RAW.utils import Logger
from typing import List, Union, Optional

class ChatBot():
    def __init__(self, name: str, llm: LLM, system_prompt: str, logger: Optional[Logger]):
        self.name = name
        self.messages: List[Message] = [
            Message(
                role='system',
                content=system_prompt
            )
        ]
        self.logger = logger if isinstance(logger, Logger) else Logger()

    def __call__(self, message: Union[str, Message]) -> Optional[Message]:
        if isinstance(message, str):
            message = Message(
                role='user',
                content=message
            )
        elif isinstance(message, Message):
            if message.role is not 'user':
                message.role = 'user'
        else:
            self.logger.warning(f'Message can only be str or Message received: {type(message)}: {message}')
            return
    
from typing import Literal, List
import re

from RAW.models import Message
from RAW.utils import Logger

class ShortTermMemory():
    def __init__(
            self, 
            logger: Logger = Logger(),
            actionable_unit: Literal['length', 'context'] = 'length',
            actionable_quantity: int = 10
        ):
        self.actionable_unit = actionable_unit
        self.actionable_quantity = actionable_quantity
        self.logger = logger

    def __call__(self, messages: List[Message]) -> List[Message]:
        if(self.actionable_unit == 'length'):
            if(len(messages) < self.actionable_quantity):
                return messages
            system_message = messages[0]
            memory_messages = messages[(-1 * self.actionable_quantity):]
            while memory_messages[0].role != 'user':
                memory_messages.pop(0)

            if(len(memory_messages) == 0):
                self.logger.warning(f'no user message found while compressing messages using Unit = {self.actionable_unit}, quantity = {self.actionable_quantity}')

            return [system_message] + memory_messages
                
        else:
            num_tokens = self._count_tokens(messages=messages)
            if(num_tokens < self.actionable_quantity):
                return messages
            
            system_message = messages.pop(0)
            first_message = messages.pop(0)

            while(num_tokens < self.actionable_quantity and first_message.role != 'user'):
                messages.pop(0)

                first_message = messages[0]
                num_tokens = self._count_tokens(messages=[system_message] + messages)

            return [system_message] + messages

    def _count_tokens(self, messages: List[Message]):
        text = "\n".join(
            f"{m['role']}: {m['content']}" for m in messages
        )

        return len(re.findall(r"\S+", text)) * 1.3
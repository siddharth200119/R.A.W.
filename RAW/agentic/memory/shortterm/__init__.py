from typing import Literal, List, Optional
import re
import json

from RAW.models import Message
from RAW.utils import Logger
from RAW.agentic.llms import LLM

class ShortTermMemory():
    def __init__(
            self, 
            llm: Optional[LLM] = None,
            logger: Logger = Logger(),
            actionable_unit: Literal['length', 'context'] = 'length',
            actionable_quantity: int = 10,
            summarize_to_system_prompt: bool = False
        ):
        self.llm = llm
        self.actionable_unit = actionable_unit
        self.actionable_quantity = actionable_quantity
        self.logger = logger
        self.summarize_to_system_prompt = summarize_to_system_prompt

    async def __call__(self, messages: List[Message]) -> tuple[List[Message], List[Message]]:
        if not messages:
            return messages, []
    
        system_message: Optional[Message] = None
        preserved_messages: List[Message] = messages.copy()
        removed_messages: List[Message] = []
        
        if messages[0].role == 'system':
            system_message = preserved_messages.pop(0)

        if self.actionable_unit == 'length':
            if len(preserved_messages) >= self.actionable_quantity:
                removed_messages = preserved_messages[:-self.actionable_quantity]
                preserved_messages = preserved_messages[-self.actionable_quantity:]
                while preserved_messages and preserved_messages[0].role != 'user':
                    removed_messages.append(preserved_messages.pop(0))

                if len(preserved_messages) == 0:
                    self.logger.warning(f'No user message found while compressing messages using Unit = {self.actionable_unit}, quantity = {self.actionable_quantity}')
                    
        else:
            current_tokens = self._count_tokens(messages=preserved_messages)
            
            if current_tokens >= self.actionable_quantity:
                while (preserved_messages and 
                       self._count_tokens(messages=preserved_messages) >= self.actionable_quantity):
                    removed_messages.append(preserved_messages.pop(0))
                
                while preserved_messages and preserved_messages[0].role != 'user':
                    removed_messages.append(preserved_messages.pop(0))
                
                if len(preserved_messages) == 0:
                    self.logger.warning(f'No user message found while compressing messages using Unit = {self.actionable_unit}, quantity = {self.actionable_quantity}')

        result: List[Message] = []
        if system_message:
            if(self.summarize_to_system_prompt and self.llm and len(removed_messages) > 0):
                system_message.content += f"""

Summary of the earliest {len(removed_messages)} which were removed to save context:

{(await self.llm.generate(prompt=f"generate a summary for these messsages of an llm: {json.dumps([message.model_dump() for message in removed_messages], indent=2)}")).content}

"""
            result.append(system_message)
        result.extend(preserved_messages)
        
        return result

    def _count_tokens(self, messages: List[Message]) -> int:
        text = "\n".join(
            f"{m.role}: {m.content}" for m in messages
        )
        return int(len(re.findall(r"\S+", text)) * 1.3)
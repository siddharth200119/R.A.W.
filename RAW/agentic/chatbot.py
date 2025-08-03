from RAW.agentic.llms import LLM
from RAW.models import Tool, Message
from RAW.utils import Logger
from RAW.agentic.memory import ShortTermMemory

from typing import List, Optional

class ChatBot():
    def __init__(
            self, 
            llm: LLM, 
            short_term_memory: Optional[ShortTermMemory],
            tools: List[Tool] = [], 
            logger: Logger = Logger(), 
            system_prompt: str = '',
        ):
        self.logger = logger
        if(not llm):
            self.logger.error("Please provide and LLM")
            raise RuntimeError("Please provide and LLM")
        
        self.llm = llm
        self.tools = tools
        self.short_term_memory = short_term_memory

        if(len(tools) > 0 and 'tools' not in self.llm.capabilities):
            self.logger.warning('the chosen LLM does not support tool calling skipping tools')
            self.tools = []

        self.messages: List[Message] = [
            Message(
                role='system',
                content=system_prompt
            )
        ]

    def __call__(self, message: Message, stream: bool = False, think: bool = False) -> Optional[Message]:
        if(not message):
            self.logger.error("Message cannot be empty")

        self.messages.append(message)

        if(self.short_term_memory):
            self.messages = self.short_term_memory(messages=self.messages)

        if(stream):
            return self._call_stream(think)
        else:
            return self._call_no_stream(think)
        
    
    async def _call_stream(self, think: bool = False):
        async for message in self.llm.chat(messages=self.messages, think=think, stream=True, tools=self.tools):
            yield message

        self.messages.append(message)
        return

    async def _call_no_stream(self, think: bool = False):
        message = await self.llm.chat(messages=self.messages, think=think, stream=False, tools=self.tools)
        self.messages.append(message)
        return message
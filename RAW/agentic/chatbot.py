from RAW.agentic.llms import LLM
from RAW.models import Tool, Message
from RAW.utils import Logger
from RAW.agentic.memory import ShortTermMemory
from RAW.agentic.memory import LongTermMemory
from typing import List, Optional
import inspect
import json

class ChatBot():
    def __init__(
            self, 
            llm: LLM, 
            user_id: str,
            short_term_memory: Optional[ShortTermMemory],
            tools: List[Tool] = [], 
            logger: Logger = Logger(), 
            system_prompt: str = '',
            long_term_memory: Optional[LongTermMemory] = None,

        ):
        self.logger = logger
        if(not llm):
            self.logger.error("Please provide and LLM")
            raise RuntimeError("Please provide and LLM")
        
        self.llm = llm
        self.user_id = user_id
        self.tools = tools
        self.short_term_memory = short_term_memory
        self.long_term_memory = long_term_memory
        if self.long_term_memory:
            update_memory_tool = self.long_term_memory.create_update_memory_tool(self.user_id)
            self.add_tool(update_memory_tool)


        if(len(tools) > 0 and 'tools' not in self.llm.capabilities):
            self.logger.warning('the chosen LLM does not support tool calling skipping tools')
            self.tools = []

        self.messages: List[Message] = [
            Message(
                role='system',
                content=system_prompt
            )
        ]

    def add_tool(self, tool: Tool):
        self.tools.append(tool)
        
    def __call__(self, message: Message, stream: bool = False, think: bool = False) -> Optional[Message]:
        if(not message):
            self.logger.error("Message cannot be empty")

        self.messages.append(message)

        if(stream):
            return self._call_stream(think)
        else:
            return self._call_no_stream(think)
        
    
    async def _call_stream(self, think: bool = False):
        if(self.short_term_memory):
            self.messages = await self.short_term_memory(messages=self.messages)
        
        if self.long_term_memory:
            self.messages = await self.long_term_memory(self.user_id, self.messages[-1])

        async for message in self.llm.chat(messages=self.messages, think=think, stream=True, tools=self.tools):
            yield message

        self.messages.append(message)
        return

    async def _call_no_stream(self, think: bool = False):
        if self.short_term_memory:
            self.messages = await self.short_term_memory(messages=self.messages)

        if self.long_term_memory:
            self.messages = await self.long_term_memory(self.user_id, self.messages[-1])

        message = await self.llm.chat(messages=self.messages, think=think, stream=False, tools=self.tools)

        if hasattr(message, "tool_calls") and message.tool_calls:
            for tool_call in message.tool_calls:
                tool_func = tool_call.tool.function
                if tool_func:
                    try:
                        if inspect.isasyncgenfunction(tool_func):
                            async for result in tool_func(**tool_call.arguments):
                                self.messages.append(Message(role='assistant', content=result))
                                if self.long_term_memory:
                                    tool_params_json = json.dumps(tool_call.arguments) if tool_call.arguments else None
                                    tool_output_json = json.dumps(result) if not isinstance(result, str) else result

                                    self.long_term_memory.storage.store_message(
                                        self.user_id,
                                        Message(role='assistant', content=result if isinstance(result, str) else json.dumps(result)),
                                        is_tool_call=1,
                                        tool_name=tool_call.tool.name,
                                        tool_params=tool_params_json,
                                        tool_output=tool_output_json
                                    )
                                return Message(role='assistant', content=result)

                        elif inspect.iscoroutinefunction(tool_func):
                            result = await tool_func(**tool_call.arguments)
                            self.messages.append(Message(role='assistant', content=result))
                            if self.long_term_memory:
                                tool_params_json = json.dumps(tool_call.arguments) if tool_call.arguments else None
                                tool_output_json = json.dumps(result) if not isinstance(result, str) else result

                                self.long_term_memory.storage.store_message(
                                    self.user_id,
                                    Message(role='assistant', content=result if isinstance(result, str) else json.dumps(result)),
                                    is_tool_call=1,
                                    tool_name=tool_call.tool.name,
                                    tool_params=tool_params_json,
                                    tool_output=tool_output_json
                                )
                            return Message(role='assistant', content=result)

                        else:
                            result = tool_func(**tool_call.arguments)
                            self.messages.append(Message(role='assistant', content=result))
                            if self.long_term_memory:
                                tool_params_json = json.dumps(tool_call.arguments) if tool_call.arguments else None
                                tool_output_json = json.dumps(result) if not isinstance(result, str) else result

                                self.long_term_memory.storage.store_message(
                                    self.user_id,
                                    Message(role='assistant', content=result if isinstance(result, str) else json.dumps(result)),
                                    is_tool_call=1,
                                    tool_name=tool_call.tool.name,
                                    tool_params=tool_params_json,
                                    tool_output=tool_output_json
                                )
                            return Message(role='assistant', content=result)

                    except Exception as e:
                        error_msg = f"Error calling tool {tool_call.tool.name}: {str(e)}"
                        self.messages.append(Message(role='assistant', content=error_msg))
                        if self.long_term_memory:
                            self.long_term_memory.storage.store_message(
                                self.user_id,
                                Message(role='assistant', content=error_msg),
                                is_tool_call=1,
                                tool_name=tool_call.tool.name,
                                tool_params=None,
                                tool_output=error_msg
                            )
                        return Message(role='assistant', content=error_msg)

        self.messages.append(message)
        if self.long_term_memory:
            self.long_term_memory.storage.store_message(self.user_id, message)
        return message

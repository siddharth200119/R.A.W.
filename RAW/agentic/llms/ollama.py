from RAW.agentic.llms.base import LLM
from RAW.utils import HTTPClient
from RAW.models import Message, Tool, toolCall

from typing import Dict, List, Literal
import json

class Ollama(LLM):
    def __init__(self, logger, track_id, model: str, base_url: str = "http://127.0.0.1:11434", timeout: int = 500, capabilities: List[Literal['thinking', 'tools', 'vision', 'embedding']] = []):
        super().__init__(logger, track_id)
        self.client = HTTPClient(base_url=base_url, timeout=timeout)
        self.model = model
        self.capabilities = capabilities

    def generate(self, prompt, images = [], format = None, think = False, stream = False):
        think = self.validate_think(think)
        for index, image in enumerate(images):
            image = self.validate_image(image)
            if(image is None):
                images.pop(index)

        body = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "images": [image.to_base64() for image in images],
            "think": think
        }

        if(format):
            body["format"] = 'json' if isinstance(format, str) else format

        if(stream):
            return self._generate_stream(body)
        else:
            return self.__generate_no_stream(body)
        
    def chat(self, messages: List[Message], think: bool = False, stream: bool = False, tools: List[Tool] = []):
        for message in messages:
            if(message.images is not None and len(message.images) != 0):
                for index, image in enumerate(message.images):
                    image = self.validate_image(image)
                    if(image is None):
                        message.images.pop(index)
        think = self.validate_think(think)

        body = {
            "model": self.model,
            "stream": stream,
            "messages": [{
                "role": message.role,
                "content": message.content,
                "images": [image.to_base64() for image in message.images],
                "tool_calls": [{
                    "function": {
                        "name": tool_call.tool.name,
                        "arguments": tool_call.arguments
                    }
                } for tool_call in message.tool_calls] if message.tool_calls is not None else []
            } for message in messages],
            "think": think,
            'tools': [{
                "type": "function",
                "function":{
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            param.name: {
                                "type": param.type,
                                "description": param.description
                            } for param in tool.parameters
                        },
                        "required": [param.name for param in tool.parameters if param.required]
                    }
                }
            } for tool in tools]
        }

        if(stream):
            return self._chat_stream(body, tools)
        else:
            return self._chat_no_stream(body, tools)
        
    async def __generate_no_stream(self, body: Dict):
        response = await self.client.apost("/api/generate", json=body)
        response = response.json()
        return Message(
            thought=response.get("thinking", ""),
            content=response.get("response", ""),
            role='assistant'
        )
    
    async def _generate_stream(self, body: Dict):
        full_thought = ""
        full_response = ""
        
        async for chunk in self.client.apost_stream(url="/api/generate", json=body):
            response = json.loads(chunk.decode())
            to_yield = Message(
                thought=response.get("thinking", ""),
                content=response.get("response", ""),
                role='assistant'
            )
            full_thought += to_yield.thought
            full_response += to_yield.content
            yield to_yield

        yield Message(
                thought=full_thought,
                content=full_response,
                role='assistant'
            )
        return
    
    async def _chat_no_stream(self, body: Dict, tools: List[Tool]):
        response = await self.client.apost('/api/chat', json=body)
        response = response.json()
        message = response.get('message')
        if(not message):
            self.logger.warning(content=f'no message received from the API, {response}')
        return Message(
            role='assistant',
            content=message.get('content', ''),
            thought=message.get('thinking', ''),
            images=[],
            tool_calls=[toolCall(
                tool=next((tool for tool in tools if tool.name == tool_call.get('function', {}).get('name')), None),
                arguments=tool_call.get('function', {}).get('arguments', {})
            ) for tool_call in message.get('tool_calls', [])],
            tool_call_id=''
        )
    
    async def _chat_stream(self, body: Dict, tools: List[Tool]):
        full_thought = ""
        full_response = ""
        
        async for chunk in self.client.apost_stream(url="/api/chat", json=body):
            response = json.loads(chunk.decode())
            message = response.get('message')
            if(not message):
                self.logger.warning(content=f'no message received from the API, {response}')
            to_yield = Message(
                role='assistant',
                content=message.get('content', ''),
                thought=message.get('thinking', ''),
                images=[],
                tool_calls=[toolCall(
                    tool=next((tool for tool in tools if tool.name == tool_call.get('function', {}).get('name')), None),
                    arguments=tool_call.get('function', {}).get('arguments', {})
                ) for tool_call in message.get('tool_calls', [])],
                tool_call_id=''
            )

            full_response += to_yield.content
            full_thought += to_yield.thought
            
            yield to_yield
        
        yield Message(
                role='assistant',
                content=full_response,
                thought=full_thought,
                images=[],
                tool_calls=[toolCall(
                    tool=next((tool for tool in tools if tool.name == tool_call.get('function', {}).get('name')), None),
                    arguments=tool_call.get('function', {}).get('arguments', {})
                ) for tool_call in message.get('tool_calls', [])],
                tool_call_id=''
            )
        return
        
    
    def embed(self, content):
        return super().embed(content)
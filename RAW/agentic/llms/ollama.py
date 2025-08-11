from RAW.agentic.llms.base import LLM
from RAW.utils import HTTPClient
from RAW.models import Message, Tool, toolCall

from typing import Dict, List, Literal, Optional
import json
import numpy as np
from pydantic import BaseModel

class OllamaOptions(BaseModel):
    num_keep: Optional[int] = None
    seed: Optional[int] = None
    num_predict: Optional[int] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    min_p: Optional[float] = None
    typical_p: Optional[float] = None
    repeat_last_n: Optional[int] = None
    temperature: Optional[float] = None
    repeat_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    penalize_newline: Optional[bool] = None
    stop: Optional[List[str]] = None
    numa: Optional[bool] = None
    num_ctx: Optional[int] = None
    num_batch: Optional[int] = None
    num_gpu: Optional[int] = None
    main_gpu: Optional[int] = None
    use_mmap: Optional[bool] = None
    num_thread: Optional[int] = None

class Ollama(LLM):
    def __init__(
            self, 
            logger, 
            model: str, 
            base_url: str = "http://127.0.0.1:11434", 
            timeout: int = 500, 
            capabilities: List[Literal['thinking', 'tools', 'vision', 'embedding']] = [],
            options: Optional[OllamaOptions] = None
        ):
        super().__init__(logger)
        self.client = HTTPClient(base_url=base_url, timeout=timeout)
        self.model = model
        self.capabilities = capabilities
        self.options = options

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

        if self.options:
            body["options"] = self.options.model_dump(exclude_none=True)

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

        if self.options:
            body["options"] = self.options.model_dump(exclude_none=True)

        if(stream):
            return self._chat_stream(body, tools)
        else:
            return self._chat_no_stream(body, tools)
        
    async def __generate_no_stream(self, body: Dict):
        try:
            response = await self.client.apost("/api/generate", json=body)
            response = response.json()
            return Message(
                thought=response.get("thinking", ""),
                content=response.get("response", ""),
                role='assistant'
            )
        except Exception as e:
            self.logger.error(f"Error in generating response: {e}", error=e)
            return
    
    async def _generate_stream(self, body: Dict):
        full_thought = ""
        full_response = ""
        
        try:
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
        except Exception as e:
            self.logger.error(f"Error in generating response: {e}", error=e)
            return
    
    async def _chat_no_stream(self, body: Dict, tools: List[Tool]):
        try:
            response = await self.client.apost('/api/chat', json=body)
            response = response.json()
            # print("response: ", response)
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
        except Exception as e:
            self.logger.error(f"Error in generating response: {e}", error=e)
            return
    
    async def _chat_stream(self, body: Dict, tools: List[Tool]):
        full_thought = ""
        full_response = ""
        try:
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
        except Exception as e:
            self.logger.error(f"Error in generating response: {e}", error=e)
            return
    
    async def embed(self, content) -> np.ndarray:
        if('embedding' not in self.capabilities):
            self.logger.warning(f"the model chosen model {self.model} does not support embeddings")
            return
        
        body = {
            "model": self.model,
            "prompt": content
        }

        if self.options:
            body["options"] = self.options.model_dump(exclude_none=True)

        try:
            response = await self.client.apost("/api/embeddings", json=body)
            response.raise_for_status()
            response = response.json()
            embedding = response.get("embedding")
            if(not embedding):
                self.logger.error("Embeddings not generated")
                return
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            self.logger.error(f"Error in generating response: {e}", error=e)
            return
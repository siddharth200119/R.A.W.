from ..request import HTTPClient
from ..logger import Logger
from .base import LLM, LLMOptions
from typing import Optional, List, Dict, Any, AsyncGenerator
from ..models import LLMCapability, LLMOutput,Message
import json

class OllamaOptions(LLMOptions):
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

class OllamaLLM(LLM):
    def __init__(self, model: str, host_url: str = "http://127.0.0.1:11434", options: Optional[OllamaOptions] = None, logger: Optional[Logger] = None, capabilities: List[LLMCapability] = []):
        super().__init__(model=model, logger=logger)
        if model == None:
            raise RuntimeError("Model cannot be none")
        self.model: str = model
        self.logger: Logger = logger if logger else Logger(name=f"{model}_ollama_logger")
        self.messages: List[Message] = []
        self.client = HTTPClient(
            base_url=host_url,
            logger=self.logger,
            timeout=100
        )
        if(len(capabilities) == 0):
            try:
                model_details_response = self.client.post("/api/show", json={"model": model})
                model_details = model_details_response.json()
                # Fixed typo: capabilites -> capabilities
                self.capabilities = [LLMCapability(cap) for cap in model_details.get("capabilities", [])]
            except Exception as e:
                raise RuntimeError(f"Error occured while getting llm capabilities: {repr(e)}")
        else:
            self.capabilities = capabilities
        
        self.options: OllamaOptions = options if options else OllamaOptions()

    def _convert_messages(self):
        return [{
            "role": message.role,
            "content": message.content if isinstance(message.content, str) else message.content.content,
            "images": [image.to_base64() for image in message.images] if message.images else [],
            "tool_calls": [{
                "function": {
                    "name": tool_call.function_name,
                    "arguments": tool_call.arguments
                }
            } for tool_call in message.content.tool_calls] if isinstance(message.content, LLMOutput) else []
        } for message in self.messages]

    def _prepare_generate_body(self, prompt, images, think, format, stream) -> Dict:
        body: Dict[str, Any] = {
            "model": self.model
        }

        if(images):
            body["images"] = [image.to_base64() for image in images]

        body["think"] = think
        
        if(format):
            body["format"] = "json" if isinstance(format, str) else format

        body["prompt"] = prompt
        body["stream"] = stream

        body["options"] = self.options.model_dump(exclude_none=True)
        return body

    def generate(self, prompt, images = [], think = False, format = None, stream = False):
        think = self._validate_thinking(think)
        images = self._validate_images(images)
        
        body = self._prepare_generate_body(prompt, images, think, format, stream=stream)

        if stream:
            return self._generate_stream(body)
        else:
            return self._generate_no_stream(body)
    
    def _perpare_chat_body(self, think, format, stream) -> Dict:
        return {
            "model": self.model,
            "messages": self._convert_messages(),
            "think": think,
            "format": "json" if isinstance(format, str) else format if format else None,
            "options": self.options.model_dump(exclude_none=True),
            "stream": stream
        }
        
    def chat(self, message, think = False, stream = False, format = None):
        think = self._validate_thinking(think)
        if(message.images):
            message.images = self._validate_images(images=message.images)
            
        self.messages.append(message)

        body = self._perpare_chat_body(think, format, stream)

        return self._chat_stream(body) if stream else self._chat_no_stream(body)

    async def _generate_no_stream(self, body) -> LLMOutput:
        response = (await self.client.apost("/api/generate", json=body)).json()
        return LLMOutput(
            thought=response.get("thinking", ""),
            content=response.get("response", '')
        )
    
    async def _generate_stream(self, body) -> AsyncGenerator[LLMOutput, None]:
        final_thought: str = ""
        final_content: str = ""

        async for item in self.client.apost_stream("/api/generate", json=body):
            string = item.decode('utf-8')
            output: Dict[str, str] = json.loads(string)
            content_chunk = output.get("response", "")
            thought_chunk = output.get("thinking", '')

            final_thought += thought_chunk
            final_content += content_chunk
            
            yield LLMOutput(
                content=content_chunk,
                thought=thought_chunk
            )
    
    async def _chat_no_stream(self, body: Dict) -> Message:
        response: Dict[str, Any] = (await self.client.apost("/api/chat", json=body)).json()
        message: Dict[str, str] = response.get("message", {})
        return Message(
            role='assistant',
            content=LLMOutput(
                content=message.get("content", ""),
                thought=message.get("thinking", "")
            )
        )
    
    async def _chat_stream(self, body: Dict) -> AsyncGenerator[Message, None]:
        final_thought: str = ""
        final_content: str = ""

        async for item in self.client.apost_stream("/api/chat", json=body):
            string = item.decode('utf-8')
            message: Dict[str, str] = json.loads(string).get("message", {})
            content_chunk = message.get("content", "")
            thought_chunk = message.get("thinking", "")
            final_thought += thought_chunk
            final_content += content_chunk
            yield Message(
                role='assistant',
                content=LLMOutput(
                    content=content_chunk,
                    thought=thought_chunk
                )
            )
        yield Message(
            role='assistant',
            content=LLMOutput(
                content=final_content,
                thought=final_thought
            )
        )
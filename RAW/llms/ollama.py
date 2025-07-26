from RAW.utils import HTTPClient, Logger
from .base import LLM, LLMOptions
from typing import Optional, List, Dict, Any, AsyncGenerator
from RAW.models import LLMCapability, LLMOutput, Message
import json
from pathlib import Path

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
    def __init__(self, model: str, host_url: str = "http://127.0.0.1:11434", 
                 options: Optional[OllamaOptions] = None, logger: Optional[Logger] = None, 
                 capabilities: List[LLMCapability] = []):
        super().__init__(model=model, logger=logger)
        
        if model == None:
            raise RuntimeError("Model cannot be none")
            
        self.model: str = model
        self.logger: Logger = logger if logger else Logger(log_file=Path("ollama_logger.log"))
        self.messages: List[Message] = []
        
        self.logger.info(f"Initializing OllamaLLM with model: {model}", tags=["init", "ollama"])
        self.logger.debug(f"Host URL: {host_url}, Options: {options}", tags=["init", "ollama", "config"])
        
        self.client = HTTPClient(
            base_url=host_url,
            logger=self.logger,
            timeout=100
        )
        
        if len(capabilities) == 0:
            self.logger.debug("No capabilities provided, fetching from model", tags=["init", "capabilities"])
            try:
                self.logger.info(f"Fetching capabilities for model: {model}", tags=["capabilities", "api"])
                model_details_response = self.client.post("/api/show", json={"model": model})
                model_details = model_details_response.json()
                
                capabilities_data = model_details.get("capabilities", [])
                self.logger.debug(f"Raw capabilities data: {capabilities_data}", tags=["capabilities", "debug"])
                
                self.capabilities = [LLMCapability(cap) for cap in capabilities_data]
                self.logger.info(f"Successfully fetched {len(self.capabilities)} capabilities", 
                               tags=["capabilities", "success"])
                self.logger.debug(f"Capabilities: {[cap.value for cap in self.capabilities]}", 
                                tags=["capabilities", "debug"])
                                
            except Exception as e:
                self.logger.error(f"Error occurred while getting LLM capabilities for model {model}", 
                                error=e, tags=["capabilities", "error", "init"])
                raise RuntimeError(f"Error occurred while getting llm capabilities: {repr(e)}")
        else:
            self.logger.info(f"Using provided capabilities: {len(capabilities)} items", 
                           tags=["init", "capabilities"])
            self.logger.debug(f"Provided capabilities: {[cap.value for cap in capabilities]}", 
                            tags=["init", "capabilities", "debug"])
            self.capabilities = capabilities
        
        self.options: OllamaOptions = options if options else OllamaOptions()
        self.logger.debug(f"Final options: {self.options.model_dump(exclude_none=True)}", 
                        tags=["init", "options"])
        self.logger.info("OllamaLLM initialization completed successfully", tags=["init", "ollama", "success"])

    def _convert_messages(self):
        self.logger.debug(f"Converting {len(self.messages)} messages for API", 
                        tags=["convert", "messages"])
        
        converted_messages = []
        for i, message in enumerate(self.messages):
            self.logger.debug(f"Converting message {i}: role={message.role}, has_images={bool(message.images)}", 
                            tags=["convert", "messages", "debug"])
            
            # Handle content conversion
            content = message.content if isinstance(message.content, str) else message.content.content
            
            # Handle images
            images = []
            if message.images:
                self.logger.debug(f"Converting {len(message.images)} images for message {i}", 
                                tags=["convert", "images"])
                try:
                    images = [image.to_base64() for image in message.images]
                    self.logger.debug(f"Successfully converted {len(images)} images to base64", 
                                    tags=["convert", "images", "success"])
                except Exception as e:
                    self.logger.error(f"Error converting images to base64 for message {i}", 
                                    error=e, tags=["convert", "images", "error"])
                    raise
            
            # Handle tool calls
            tool_calls = []
            if isinstance(message.content, LLMOutput) and hasattr(message.content, 'tool_calls') and message.content.tool_calls:
                self.logger.debug(f"Converting {len(message.content.tool_calls)} tool calls for message {i}", 
                                tags=["convert", "tool_calls"])
                try:
                    tool_calls = [{
                        "function": {
                            "name": tool_call.function_name,
                            "arguments": tool_call.arguments
                        }
                    } for tool_call in message.content.tool_calls]
                    self.logger.debug(f"Successfully converted {len(tool_calls)} tool calls", 
                                    tags=["convert", "tool_calls", "success"])
                except Exception as e:
                    self.logger.error(f"Error converting tool calls for message {i}", 
                                    error=e, tags=["convert", "tool_calls", "error"])
                    raise
            
            converted_message = {
                "role": message.role,
                "content": content,
                "images": images,
                "tool_calls": tool_calls
            }
            converted_messages.append(converted_message)
        
        self.logger.info(f"Successfully converted {len(converted_messages)} messages", 
                       tags=["convert", "messages", "success"])
        return converted_messages

    def _prepare_generate_body(self, prompt, images, think, format, stream) -> Dict:
        self.logger.debug(f"Preparing generate body - prompt_length={len(prompt)}, images={len(images) if images else 0}, think={think}, format={format}, stream={stream}", 
                        tags=["prepare", "generate"])
        
        body: Dict[str, Any] = {
            "model": self.model
        }

        if images:
            self.logger.debug(f"Adding {len(images)} images to generate body", tags=["prepare", "generate", "images"])
            try:
                body["images"] = [image.to_base64() for image in images]
                self.logger.debug("Successfully converted images to base64 for generate", 
                                tags=["prepare", "generate", "images", "success"])
            except Exception as e:
                self.logger.error("Error converting images to base64 for generate", 
                                error=e, tags=["prepare", "generate", "images", "error"])
                raise

        body["think"] = think
        
        if format:
            format_value = "json" if isinstance(format, str) else format
            body["format"] = format_value
            self.logger.debug(f"Added format to generate body: {format_value}", 
                            tags=["prepare", "generate", "format"])

        body["prompt"] = prompt
        body["stream"] = stream

        options_dict = self.options.model_dump(exclude_none=True)
        body["options"] = options_dict
        
        self.logger.debug(f"Generate body prepared with options: {options_dict}", 
                        tags=["prepare", "generate", "options"])
        self.logger.info("Generate request body prepared successfully", 
                       tags=["prepare", "generate", "success"])
        
        return body

    def generate(self, prompt, images=[], think=False, format=None, stream=False):
        self.logger.info(f"Starting generate request - prompt_length={len(prompt)}, stream={stream}", 
                       tags=["generate", "start"])
        self.logger.debug(f"Generate parameters - images={len(images)}, think={think}, format={format}", 
                        tags=["generate", "params"])
        
        try:
            think = self._validate_thinking(think)
            self.logger.debug(f"Validated thinking parameter: {think}", tags=["generate", "validation"])
            
            images = self._validate_images(images)
            self.logger.debug(f"Validated images: {len(images)} items", tags=["generate", "validation"])
            
            body = self._prepare_generate_body(prompt, images, think, format, stream=stream)
            
            if stream:
                self.logger.info("Initiating streaming generate request", tags=["generate", "stream"])
                return self._generate_stream(body)
            else:
                self.logger.info("Initiating non-streaming generate request", tags=["generate", "no_stream"])
                return self._generate_no_stream(body)
                
        except Exception as e:
            self.logger.error("Error in generate method", error=e, tags=["generate", "error"])
            raise
    
    def _perpare_chat_body(self, think, format, stream) -> Dict:
        self.logger.debug(f"Preparing chat body - think={think}, format={format}, stream={stream}", 
                        tags=["prepare", "chat"])
        
        format_value = "json" if isinstance(format, str) else format if format else None
        
        body = {
            "model": self.model,
            "messages": self._convert_messages(),
            "think": think,
            "format": format_value,
            "options": self.options.model_dump(exclude_none=True),
            "stream": stream
        }
        
        self.logger.info(f"Chat body prepared with {len(body['messages'])} messages", 
                       tags=["prepare", "chat", "success"])
        self.logger.debug(f"Chat body format: {format_value}, options: {body['options']}", 
                        tags=["prepare", "chat", "debug"])
        
        return body
        
    def chat(self, message, think=False, stream=False, format=None):
        self.logger.info(f"Starting chat request - role={message.role}, stream={stream}", 
                       tags=["chat", "start"])
        self.logger.debug(f"Chat parameters - think={think}, format={format}, has_images={bool(message.images)}", 
                        tags=["chat", "params"])
        
        try:
            think = self._validate_thinking(think)
            self.logger.debug(f"Validated thinking parameter: {think}", tags=["chat", "validation"])
            
            if message.images:
                self.logger.debug(f"Validating {len(message.images)} images in message", 
                                tags=["chat", "validation", "images"])
                message.images = self._validate_images(images=message.images)
                self.logger.debug("Images validated successfully", tags=["chat", "validation", "images"])
                
            self.messages.append(message)
            self.logger.info(f"Added message to conversation. Total messages: {len(self.messages)}", 
                           tags=["chat", "messages"])

            body = self._perpare_chat_body(think, format, stream)

            if stream:
                self.logger.info("Initiating streaming chat request", tags=["chat", "stream"])
                return self._chat_stream(body)
            else:
                self.logger.info("Initiating non-streaming chat request", tags=["chat", "no_stream"])
                return self._chat_no_stream(body)
                
        except Exception as e:
            self.logger.error("Error in chat method", error=e, tags=["chat", "error"])
            raise

    async def _generate_no_stream(self, body) -> LLMOutput:
        self.logger.info("Executing non-streaming generate request", tags=["generate", "no_stream", "api"])
        self.logger.debug(f"Generate request body keys: {list(body.keys())}", 
                        tags=["generate", "no_stream", "debug"])
        
        try:
            response = await self.client.apost("/api/generate", json=body)
            self.logger.info(f"Generate API response received - status: {response.status_code}", 
                           tags=["generate", "no_stream", "response"])
            
            response_data = response.json()
            self.logger.debug(f"Generate response keys: {list(response_data.keys())}", 
                            tags=["generate", "no_stream", "debug"])
            
            thought = response_data.get("thinking", "")
            content = response_data.get("response", '')
            
            self.logger.info(f"Generate completed - content_length={len(content)}, has_thought={bool(thought)}", 
                           tags=["generate", "no_stream", "success"])
            
            result = LLMOutput(
                thought=thought,
                content=content
            )
            
            return result
            
        except Exception as e:
            self.logger.error("Error in non-streaming generate request", 
                            error=e, tags=["generate", "no_stream", "error"])
            raise
    
    async def _generate_stream(self, body) -> AsyncGenerator[LLMOutput, None]:
        self.logger.info("Starting streaming generate request", tags=["generate", "stream", "api"])
        self.logger.debug(f"Generate stream request body keys: {list(body.keys())}", 
                        tags=["generate", "stream", "debug"])
        
        final_thought: str = ""
        final_content: str = ""
        chunk_count = 0

        try:
            async for item in self.client.apost_stream("/api/generate", json=body):
                chunk_count += 1
                
                try:
                    string = item.decode('utf-8')
                    output: Dict[str, str] = json.loads(string)
                    
                    content_chunk = output.get("response", "")
                    thought_chunk = output.get("thinking", '')

                    final_thought += thought_chunk
                    final_content += content_chunk
                    
                    if chunk_count % 10 == 0:  # Log every 10 chunks to avoid spam
                        self.logger.debug(f"Generate stream progress - chunk {chunk_count}, content_length={len(final_content)}", 
                                        tags=["generate", "stream", "progress"])
                    
                    yield LLMOutput(
                        content=content_chunk,
                        thought=thought_chunk
                    )
                    
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse JSON in generate stream chunk {chunk_count}: {string[:100]}", 
                                      tags=["generate", "stream", "json_error"])
                    continue
                except Exception as e:
                    self.logger.error(f"Error processing generate stream chunk {chunk_count}", 
                                    error=e, tags=["generate", "stream", "chunk_error"])
                    continue
            
            self.logger.info(f"Generate stream completed - {chunk_count} chunks, final_content_length={len(final_content)}, has_final_thought={bool(final_thought)}", 
                           tags=["generate", "stream", "success"])
                           
        except Exception as e:
            self.logger.error("Error in streaming generate request", 
                            error=e, tags=["generate", "stream", "error"])
            raise
    
    async def _chat_no_stream(self, body: Dict) -> Message:
        self.logger.info("Executing non-streaming chat request", tags=["chat", "no_stream", "api"])
        self.logger.debug(f"Chat request body keys: {list(body.keys())}, messages_count={len(body.get('messages', []))}", 
                        tags=["chat", "no_stream", "debug"])
        
        try:
            response = await self.client.apost("/api/chat", json=body)
            self.logger.info(f"Chat API response received - status: {response.status_code}", 
                           tags=["chat", "no_stream", "response"])
            
            response_data: Dict[str, Any] = response.json()
            self.logger.debug(f"Chat response keys: {list(response_data.keys())}", 
                            tags=["chat", "no_stream", "debug"])
            
            message_data: Dict[str, str] = response_data.get("message", {})
            content = message_data.get("content", "")
            thought = message_data.get("thinking", "")
            
            self.logger.info(f"Chat completed - content_length={len(content)}, has_thought={bool(thought)}", 
                           tags=["chat", "no_stream", "success"])
            
            result = Message(
                role='assistant',
                content=LLMOutput(
                    content=content,
                    thought=thought
                )
            )
            
            return result
            
        except Exception as e:
            self.logger.error("Error in non-streaming chat request", 
                            error=e, tags=["chat", "no_stream", "error"])
            raise
    
    async def _chat_stream(self, body: Dict) -> AsyncGenerator[Message, None]:
        self.logger.info("Starting streaming chat request", tags=["chat", "stream", "api"])
        self.logger.debug(f"Chat stream request body keys: {list(body.keys())}, messages_count={len(body.get('messages', []))}", 
                        tags=["chat", "stream", "debug"])
        
        final_thought: str = ""
        final_content: str = ""
        chunk_count = 0

        try:
            async for item in self.client.apost_stream("/api/chat", json=body):
                chunk_count += 1
                
                try:
                    string = item.decode('utf-8')
                    response_data = json.loads(string)
                    message: Dict[str, str] = response_data.get("message", {})
                    
                    content_chunk = message.get("content", "")
                    thought_chunk = message.get("thinking", "")
                    
                    final_thought += thought_chunk
                    final_content += content_chunk
                    
                    if chunk_count % 10 == 0:  # Log every 10 chunks to avoid spam
                        self.logger.debug(f"Chat stream progress - chunk {chunk_count}, content_length={len(final_content)}", 
                                        tags=["chat", "stream", "progress"])
                    
                    yield Message(
                        role='assistant',
                        content=LLMOutput(
                            content=content_chunk,
                            thought=thought_chunk
                        )
                    )
                    
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse JSON in chat stream chunk {chunk_count}: {string[:100]}", 
                                      tags=["chat", "stream", "json_error"])
                    continue
                except Exception as e:
                    self.logger.error(f"Error processing chat stream chunk {chunk_count}", 
                                    error=e, tags=["chat", "stream", "chunk_error"])
                    continue
            
            # Yield final message with complete content
            self.logger.info(f"Chat stream completed - {chunk_count} chunks, final_content_length={len(final_content)}, has_final_thought={bool(final_thought)}", 
                           tags=["chat", "stream", "success"])
            
            yield Message(
                role='assistant',
                content=LLMOutput(
                    content=final_content,
                    thought=final_thought
                )
            )
            
        except Exception as e:
            self.logger.error("Error in streaming chat request", 
                            error=e, tags=["chat", "stream", "error"])
            raise
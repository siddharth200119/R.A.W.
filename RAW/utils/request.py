import httpx
import asyncio
from typing import Any, Dict, Optional, Iterable, AsyncIterable, Set, Union
from types import TracebackType
from .logger import Logger
from pathlib import Path

class HTTPClient:
    HTTPStatusError = httpx.HTTPStatusError
    Timeout = httpx.Timeout

    def __init__(self, base_url: str = "", headers: Optional[Dict[str, str]] = None, 
                 timeout: Union[float, Timeout] = 10.0, logger: Optional[Logger] = None, 
                 **client_kwargs: Any):
        self.base_url = base_url.rstrip("/")
        self.headers = headers or {}
        self.timeout = timeout
        self.client_kwargs = client_kwargs
        self.logger: Logger = logger if logger else Logger(log_file=Path("requests.log"))
        self.sync_client: Optional[httpx.Client] = None
        self.async_client: Optional[httpx.AsyncClient] = None
        self._tasks: Set[asyncio.Task] = set()
        self._active_streaming_responses: Set[httpx.Response] = set()
        self._lock = asyncio.Lock()
        
        self.logger.info(f"HTTPClient initialized", tags=["init", "http_client"])
        self.logger.debug(f"Base URL: {self.base_url}, Timeout: {timeout}, Headers: {self.headers}", tags=["init", "config"])
        self.logger.debug(f"Client kwargs: {client_kwargs}", tags=["init", "config"])

    def __enter__(self) -> "HTTPClient":
        self.logger.debug("Entering sync context manager", tags=["context", "sync"])
        self._init_sync_client()
        return self

    def __exit__(self, exc_type: Optional[type[BaseException]], 
                 exc_val: Optional[BaseException], 
                 exc_tb: Optional[TracebackType]) -> None:
        self.logger.debug("Exiting sync context manager", tags=["context", "sync"])
        if exc_val:
            self.logger.error(f"Exception in sync context: {exc_val}", error=exc_val, tags=["context", "error"])
        self.close()

    async def __aenter__(self) -> "HTTPClient":
        self.logger.debug("Entering async context manager", tags=["context", "async"])
        await self._init_async_client()
        return self

    async def __aexit__(self, exc_type: Optional[type[BaseException]], 
                        exc_val: Optional[BaseException], 
                        exc_tb: Optional[TracebackType]) -> None:
        self.logger.debug("Exiting async context manager", tags=["context", "async"])
        if exc_val:
            self.logger.error(f"Exception in async context: {exc_val}", error=exc_val, tags=["context", "error"])
        self.logger.info("Closed Async HTTP Client", tags=["cleanup", "async"])
        await self.aclose()

    def _init_sync_client(self) -> None:
        if self.sync_client is None:
            self.logger.debug("Initializing sync HTTP client", tags=["init", "sync"])
            try:
                self.sync_client = httpx.Client(
                    base_url=self.base_url,
                    headers=self.headers,
                    timeout=self.timeout,
                    **self.client_kwargs
                )
                self.logger.info("Initialized sync HTTP client successfully", tags=["init", "sync"])
            except Exception as e:
                self.logger.error("Failed to initialize sync HTTP client", error=e, tags=["init", "sync", "error"])
                raise
        else:
            self.logger.debug("Sync HTTP client already initialized", tags=["init", "sync"])

    async def _init_async_client(self) -> None:
        if self.async_client is None:
            self.logger.debug("Initializing async HTTP client", tags=["init", "async"])
            try:
                self.async_client = httpx.AsyncClient(
                    base_url=self.base_url,
                    headers=self.headers,
                    timeout=self.timeout,
                    **self.client_kwargs
                )
                self.logger.info("Initialized async HTTP client successfully", tags=["init", "async"])
            except Exception as e:
                self.logger.error("Failed to initialize async HTTP client", error=e, tags=["init", "async", "error"])
                raise
        else:
            self.logger.debug("Async HTTP client already initialized", tags=["init", "async"])

    # ---------------------- Sync Methods ---------------------- #
    def request(self, method: str, url: str, raise_for_status: bool = True, **kwargs: Any) -> httpx.Response:
        self.logger.info(f"Making sync {method} request to {url}", tags=["request", "sync", method.lower()])
        self.logger.debug(f"Request kwargs: {kwargs}", tags=["request", "sync", "debug"])
        
        self._init_sync_client()
        assert self.sync_client is not None
        
        try:
            response = self.sync_client.request(method, url, **kwargs)
            self.logger.info(f"Received response: {response.status_code} for {method} {url}", 
                           tags=["response", "sync", method.lower()])
            self.logger.debug(f"Response headers: {dict(response.headers)}", tags=["response", "sync", "debug"])
            
            if raise_for_status:
                response.raise_for_status()
                self.logger.debug("Response status check passed", tags=["response", "sync"])
            
            return response
        except httpx.HTTPStatusError as e:
            self.logger.error(f"HTTP status error for {method} {url}: {e.response.status_code}", 
                            error=e, tags=["request", "sync", "http_error"])
            raise
        except Exception as e:
            self.logger.error(f"Request failed for {method} {url}", error=e, tags=["request", "sync", "error"])
            raise

    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        self.logger.debug(f"GET request wrapper called for {url}", tags=["get", "sync"])
        self.logger.debug(f"GET kwargs: {kwargs}", tags=["get", "sync", "debug"])
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        self.logger.debug(f"POST request wrapper called for {url}", tags=["post", "sync"])
        self.logger.debug(f"POST kwargs: {kwargs}", tags=["post", "sync", "debug"])
        return self.request("POST", url, **kwargs)

    def put(self, url: str, **kwargs: Any) -> httpx.Response:
        self.logger.debug(f"PUT request wrapper called for {url}", tags=["put", "sync"])
        self.logger.debug(f"PUT kwargs: {kwargs}", tags=["put", "sync", "debug"])
        return self.request("PUT", url, **kwargs)

    def patch(self, url: str, **kwargs: Any) -> httpx.Response:
        self.logger.debug(f"PATCH request wrapper called for {url}", tags=["patch", "sync"])
        self.logger.debug(f"PATCH kwargs: {kwargs}", tags=["patch", "sync", "debug"])
        return self.request("PATCH", url, **kwargs)

    def delete(self, url: str, **kwargs: Any) -> httpx.Response:
        self.logger.debug(f"DELETE request wrapper called for {url}", tags=["delete", "sync"])
        self.logger.debug(f"DELETE kwargs: {kwargs}", tags=["delete", "sync", "debug"])
        return self.request("DELETE", url, **kwargs)

    def head(self, url: str, **kwargs: Any) -> httpx.Response:
        self.logger.debug(f"HEAD request wrapper called for {url}", tags=["head", "sync"])
        self.logger.debug(f"HEAD kwargs: {kwargs}", tags=["head", "sync", "debug"])
        return self.request("HEAD", url, **kwargs)

    def options(self, url: str, **kwargs: Any) -> httpx.Response:
        self.logger.debug(f"OPTIONS request wrapper called for {url}", tags=["options", "sync"])
        self.logger.debug(f"OPTIONS kwargs: {kwargs}", tags=["options", "sync", "debug"])
        return self.request("OPTIONS", url, **kwargs)

    # ---------------------- Sync Streaming Methods ---------------------- #
    def stream(self, method: str, url: str, raise_for_status: bool = True, **kwargs: Any) -> Iterable[bytes]:
        self.logger.info(f"Starting sync {method} stream to {url}", tags=["stream", "sync", method.lower()])
        self.logger.debug(f"Stream kwargs: {kwargs}", tags=["stream", "sync", "debug"])
        
        self._init_sync_client()
        assert self.sync_client is not None
        
        response = None
        try:
            response = self.sync_client.stream(method, url, **kwargs)
            self.logger.info(f"Stream response received: {response.status_code} for {method} {url}", 
                           tags=["stream", "sync", "response"])
            
            if raise_for_status:
                response.raise_for_status()
                self.logger.debug("Stream response status check passed", tags=["stream", "sync"])
            
            chunk_count = 0
            for chunk in response.iter_bytes():
                chunk_count += 1
                if chunk_count % 100 == 0:  # Log every 100 chunks to avoid spam
                    self.logger.debug(f"Processed {chunk_count} chunks from stream", tags=["stream", "sync", "progress"])
                yield chunk
                
            self.logger.info(f"Stream completed: {chunk_count} total chunks from {method} {url}", 
                           tags=["stream", "sync", "complete"])
                           
        except httpx.HTTPStatusError as e:
            self.logger.error(f"HTTP status error in stream for {method} {url}: {e.response.status_code}", 
                            error=e, tags=["stream", "sync", "http_error"])
            raise
        except Exception as e:
            self.logger.error(f"Stream failed for {method} {url}", error=e, tags=["stream", "sync", "error"])
            raise
        finally:
            if response:
                response.close()
                self.logger.debug(f"Stream response closed for {method} {url}", tags=["stream", "sync", "cleanup"])

    def get_stream(self, url: str, **kwargs: Any) -> Iterable[bytes]:
        self.logger.debug(f"GET stream wrapper called for {url}", tags=["get_stream", "sync"])
        self.logger.debug(f"GET stream kwargs: {kwargs}", tags=["get_stream", "sync", "debug"])
        return self.stream("GET", url, **kwargs)

    def post_stream(self, url: str, **kwargs: Any) -> Iterable[bytes]:
        self.logger.debug(f"POST stream wrapper called for {url}", tags=["post_stream", "sync"])
        self.logger.debug(f"POST stream kwargs: {kwargs}", tags=["post_stream", "sync", "debug"])
        return self.stream("POST", url, **kwargs)

    def put_stream(self, url: str, **kwargs: Any) -> Iterable[bytes]:
        self.logger.debug(f"PUT stream wrapper called for {url}", tags=["put_stream", "sync"])
        self.logger.debug(f"PUT stream kwargs: {kwargs}", tags=["put_stream", "sync", "debug"])
        return self.stream("PUT", url, **kwargs)

    def patch_stream(self, url: str, **kwargs: Any) -> Iterable[bytes]:
        self.logger.debug(f"PATCH stream wrapper called for {url}", tags=["patch_stream", "sync"])
        self.logger.debug(f"PATCH stream kwargs: {kwargs}", tags=["patch_stream", "sync", "debug"])
        return self.stream("PATCH", url, **kwargs)

    def delete_stream(self, url: str, **kwargs: Any) -> Iterable[bytes]:
        self.logger.debug(f"DELETE stream wrapper called for {url}", tags=["delete_stream", "sync"])
        self.logger.debug(f"DELETE stream kwargs: {kwargs}", tags=["delete_stream", "sync", "debug"])
        return self.stream("DELETE", url, **kwargs)

    # ---------------------- Async Methods ---------------------- #
    async def arequest(self, method: str, url: str, raise_for_status: bool = True, **kwargs: Any) -> httpx.Response:
        self.logger.info(f"Making async {method} request to {url}", tags=["request", "async", method.lower()])
        self.logger.debug(f"Async request kwargs: {kwargs}", tags=["request", "async", "debug"])
        
        await self._init_async_client()
        assert self.async_client is not None
        
        task = asyncio.create_task(self.async_client.request(method, url, **kwargs))
        async with self._lock:
            self._tasks.add(task)
            self.logger.debug(f"Added task to active tasks set. Total tasks: {len(self._tasks)}", 
                            tags=["task", "async", "tracking"])
        
        try:
            response = await task
            self.logger.info(f"Received async response: {response.status_code} for {method} {url}", 
                           tags=["response", "async", method.lower()])
            self.logger.debug(f"Async response headers: {dict(response.headers)}", tags=["response", "async", "debug"])
            
            if raise_for_status:
                response.raise_for_status()
                self.logger.debug("Async response status check passed", tags=["response", "async"])
                
            return response
        except httpx.HTTPStatusError as e:
            self.logger.error(f"HTTP status error for async {method} {url}: {e.response.status_code}", 
                            error=e, tags=["request", "async", "http_error"])
            raise
        except Exception as e:
            self.logger.error(f"Async request failed for {method} {url}", error=e, tags=["request", "async", "error"])
            raise
        finally:
            async with self._lock:
                self._tasks.discard(task)
                self.logger.debug(f"Removed task from active tasks set. Remaining tasks: {len(self._tasks)}", 
                                tags=["task", "async", "tracking"])

    async def aget(self, url: str, **kwargs: Any) -> httpx.Response:
        self.logger.debug(f"Async GET request wrapper called for {url}", tags=["aget", "async"])
        self.logger.debug(f"Async GET kwargs: {kwargs}", tags=["aget", "async", "debug"])
        return await self.arequest("GET", url, **kwargs)

    async def apost(self, url: str, **kwargs: Any) -> httpx.Response:
        self.logger.debug(f"Async POST request wrapper called for {url}", tags=["apost", "async"])
        self.logger.debug(f"Async POST kwargs: {kwargs}", tags=["apost", "async", "debug"])
        return await self.arequest("POST", url, **kwargs)

    async def aput(self, url: str, **kwargs: Any) -> httpx.Response:
        self.logger.debug(f"Async PUT request wrapper called for {url}", tags=["aput", "async"])
        self.logger.debug(f"Async PUT kwargs: {kwargs}", tags=["aput", "async", "debug"])
        return await self.arequest("PUT", url, **kwargs)

    async def apatch(self, url: str, **kwargs: Any) -> httpx.Response:
        self.logger.debug(f"Async PATCH request wrapper called for {url}", tags=["apatch", "async"])
        self.logger.debug(f"Async PATCH kwargs: {kwargs}", tags=["apatch", "async", "debug"])
        return await self.arequest("PATCH", url, **kwargs)

    async def adelete(self, url: str, **kwargs: Any) -> httpx.Response:
        self.logger.debug(f"Async DELETE request wrapper called for {url}", tags=["adelete", "async"])
        self.logger.debug(f"Async DELETE kwargs: {kwargs}", tags=["adelete", "async", "debug"])
        return await self.arequest("DELETE", url, **kwargs)

    async def ahead(self, url: str, **kwargs: Any) -> httpx.Response:
        self.logger.debug(f"Async HEAD request wrapper called for {url}", tags=["ahead", "async"])
        self.logger.debug(f"Async HEAD kwargs: {kwargs}", tags=["ahead", "async", "debug"])
        return await self.arequest("HEAD", url, **kwargs)

    async def aoptions(self, url: str, **kwargs: Any) -> httpx.Response:
        self.logger.debug(f"Async OPTIONS request wrapper called for {url}", tags=["aoptions", "async"])
        self.logger.debug(f"Async OPTIONS kwargs: {kwargs}", tags=["aoptions", "async", "debug"])
        return await self.arequest("OPTIONS", url, **kwargs)

    # ---------------------- Async Streaming Methods ---------------------- #
    async def astream(self, method: str, url: str, raise_for_status: bool = True, **kwargs: Any) -> AsyncIterable[bytes]:
        self.logger.info(f"Starting async {method} stream to {url}", tags=["stream", "async", method.lower()])
        self.logger.debug(f"Async stream kwargs: {kwargs}", tags=["stream", "async", "debug"])
        
        await self._init_async_client()
        assert self.async_client is not None
        
        try:
            async with self.async_client.stream(method, url, **kwargs) as response:
                async with self._lock:
                    self._active_streaming_responses.add(response)
                    self.logger.debug(f"Added streaming response to active set. Total active: {len(self._active_streaming_responses)}", 
                                    tags=["stream", "async", "tracking"])
                
                try:
                    self.logger.info(f"Async stream response received: {response.status_code} for {method} {url}", 
                                   tags=["stream", "async", "response"])
                    
                    if raise_for_status:
                        response.raise_for_status()
                        self.logger.debug("Async stream response status check passed", tags=["stream", "async"])
                    
                    chunk_count = 0
                    async for chunk in response.aiter_bytes():
                        chunk_count += 1
                        if chunk_count % 100 == 0:  # Log every 100 chunks to avoid spam
                            self.logger.debug(f"Processed {chunk_count} chunks from async stream", 
                                            tags=["stream", "async", "progress"])
                        yield chunk
                        
                    self.logger.info(f"Async stream completed: {chunk_count} total chunks from {method} {url}", 
                                   tags=["stream", "async", "complete"])
                                   
                except httpx.HTTPStatusError as e:
                    self.logger.error(f"HTTP status error in async stream for {method} {url}: {e.response.status_code}", 
                                    error=e, tags=["stream", "async", "http_error"])
                    raise
                except Exception as e:
                    self.logger.error(f"Async stream failed for {method} {url}", error=e, tags=["stream", "async", "error"])
                    raise
                finally:
                    async with self._lock:
                        self._active_streaming_responses.discard(response)
                        self.logger.debug(f"Removed streaming response from active set. Remaining: {len(self._active_streaming_responses)}", 
                                        tags=["stream", "async", "tracking"])
        except Exception as e:
            self.logger.error(f"Failed to establish async stream for {method} {url}", error=e, tags=["stream", "async", "error"])
            raise

    async def aget_stream(self, url: str, **kwargs: Any) -> AsyncIterable[bytes]:
        self.logger.debug(f"Async GET stream wrapper called for {url}", tags=["aget_stream", "async"])
        self.logger.debug(f"Async GET stream kwargs: {kwargs}", tags=["aget_stream", "async", "debug"])
        async for chunk in self.astream("GET", url, **kwargs):
            yield chunk

    async def apost_stream(self, url: str, **kwargs: Any) -> AsyncIterable[bytes]:
        self.logger.debug(f"Async POST stream wrapper called for {url}", tags=["apost_stream", "async"])
        self.logger.debug(f"Async POST stream kwargs: {kwargs}", tags=["apost_stream", "async", "debug"])
        async for chunk in self.astream("POST", url, **kwargs):
            yield chunk

    async def aput_stream(self, url: str, **kwargs: Any) -> AsyncIterable[bytes]:
        self.logger.debug(f"Async PUT stream wrapper called for {url}", tags=["aput_stream", "async"])
        self.logger.debug(f"Async PUT stream kwargs: {kwargs}", tags=["aput_stream", "async", "debug"])
        async for chunk in self.astream("PUT", url, **kwargs):
            yield chunk

    async def apatch_stream(self, url: str, **kwargs: Any) -> AsyncIterable[bytes]:
        self.logger.debug(f"Async PATCH stream wrapper called for {url}", tags=["apatch_stream", "async"])
        self.logger.debug(f"Async PATCH stream kwargs: {kwargs}", tags=["apatch_stream", "async", "debug"])
        async for chunk in self.astream("PATCH", url, **kwargs):
            yield chunk

    async def adelete_stream(self, url: str, **kwargs: Any) -> AsyncIterable[bytes]:
        self.logger.debug(f"Async DELETE stream wrapper called for {url}", tags=["adelete_stream", "async"])
        self.logger.debug(f"Async DELETE stream kwargs: {kwargs}", tags=["adelete_stream", "async", "debug"])
        async for chunk in self.astream("DELETE", url, **kwargs):
            yield chunk

    # ---------------------- Cancel Async Requests ---------------------- #
    async def cancel_all_requests(self) -> None:
        """Cancel all pending async tasks and streaming responses"""
        self.logger.warning("Cancelling all active requests and streams", tags=["cancel", "async"])
        
        async with self._lock:
            # Cancel tasks
            task_count = len(self._tasks)
            for task in self._tasks:
                if not task.cancelled():
                    task.cancel()
                    self.logger.debug("Cancelled async task", tags=["cancel", "async", "task"])
            self._tasks.clear()
            
            # Close active streaming responses
            stream_count = len(self._active_streaming_responses)
            for response in self._active_streaming_responses:
                try:
                    await response.aclose()
                    self.logger.debug("Closed active streaming response", tags=["cancel", "async", "stream"])
                except Exception as e:
                    self.logger.warning(f"Error closing streaming response", error=e, tags=["cancel", "async", "stream", "error"])
            self._active_streaming_responses.clear()
            
            self.logger.info(f"Cancelled {task_count} tasks and {stream_count} streaming responses", 
                           tags=["cancel", "async", "complete"])

    # ---------------------- Cleanup ---------------------- #
    def close(self) -> None:
        self.logger.info("Closing sync HTTP client", tags=["cleanup", "sync"])
        if self.sync_client:
            try:
                self.sync_client.close()
                self.sync_client = None
                self.logger.info("Sync HTTP client closed successfully", tags=["cleanup", "sync"])
            except Exception as e:
                self.logger.error("Error closing sync HTTP client", error=e, tags=["cleanup", "sync", "error"])
        else:
            self.logger.debug("Sync HTTP client was not initialized, nothing to close", tags=["cleanup", "sync"])

    async def aclose(self) -> None:
        self.logger.info("Closing async HTTP client", tags=["cleanup", "async"])
        try:
            await self.cancel_all_requests()
            if self.async_client:
                await self.async_client.aclose()
                self.async_client = None
                self.logger.info("Async HTTP client closed successfully", tags=["cleanup", "async"])
            else:
                self.logger.debug("Async HTTP client was not initialized, nothing to close", tags=["cleanup", "async"])
        except Exception as e:
            self.logger.error("Error closing async HTTP client", error=e, tags=["cleanup", "async", "error"])
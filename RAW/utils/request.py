import httpx
import asyncio
from typing import Any, Dict, Optional, Iterable, AsyncIterable, Set, Union
from types import TracebackType
from .logger import Logger

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
        self.logger: Logger = logger if logger else Logger()
        self.sync_client: Optional[httpx.Client] = None
        self.async_client: Optional[httpx.AsyncClient] = None
        self._tasks: Set[asyncio.Task] = set()
        self._active_streaming_responses: Set[httpx.Response] = set()
        self._lock = asyncio.Lock()
        
        self.logger.info(f"HTTPClient initialized with base_url='{self.base_url}', timeout={timeout}")

    def __enter__(self) -> "HTTPClient":
        self._init_sync_client()
        return self

    def __exit__(self, exc_type: Optional[type[BaseException]], 
                 exc_val: Optional[BaseException], 
                 exc_tb: Optional[TracebackType]) -> None:
        if exc_val:
            self.logger.error(f"Context manager exiting with exception: {str(exc_val)}")
        self.close()

    async def __aenter__(self) -> "HTTPClient":
        await self._init_async_client()
        return self

    async def __aexit__(self, exc_type: Optional[type[BaseException]], 
                        exc_val: Optional[BaseException], 
                        exc_tb: Optional[TracebackType]) -> None:
        if exc_val:
            self.logger.error(f"Async context manager exiting with exception: {str(exc_val)}")
        await self.aclose()

    def _init_sync_client(self) -> None:
        if self.sync_client is None:
            try:
                self.sync_client = httpx.Client(
                    base_url=self.base_url,
                    headers=self.headers,
                    timeout=self.timeout,
                    **self.client_kwargs
                )
                self.logger.info("Initialized sync HTTP client")
            except Exception as e:
                self.logger.error(f"Failed to initialize sync HTTP client: {str(e)}", error=e)
                raise

    async def _init_async_client(self) -> None:
        if self.async_client is None:
            try:
                self.async_client = httpx.AsyncClient(
                    base_url=self.base_url,
                    headers=self.headers,
                    timeout=self.timeout,
                    **self.client_kwargs
                )
                self.logger.info("Initialized async HTTP client")
            except Exception as e:
                self.logger.error(f"Failed to initialize async HTTP client: {str(e)}", error=e)
                raise

    def _log_request(self, method: str, url: str, **kwargs: Any) -> None:
        """Log request details in a structured way"""
        log_data = {
            "method": method,
            "url": url,
            "headers": kwargs.get("headers", {}),
            "params": kwargs.get("params", {}),
        }
        
        # Only log body if it's text/JSON and not too large
        if "data" in kwargs and isinstance(kwargs["data"], (str, bytes)) and len(kwargs["data"]) < 1024:
            log_data["data"] = kwargs["data"]
        if "json" in kwargs:
            log_data["json"] = kwargs["json"]
            
        self.logger.info(f"Making request: {method} {url}, request: {log_data}")

    def _log_response(self, method: str, url: str, response: httpx.Response) -> None:
        """Log response details in a structured way"""
        try:
            content_type = response.headers.get("content-type", "")
            log_data = {
                "method": method,
                "url": url,
                "status_code": response.status_code,
                "headers": dict(response.headers),
            }
            
            if "json" in content_type:
                try:
                    log_data["json"] = response.json()
                except Exception as e:
                    pass
            elif "text" in content_type:
                log_data["text"] = response.text
                
            self.logger.info(
                f"Received response: {response.status_code} for {method} {url}, response: {log_data}"
            )
        except Exception as e:
            self.logger.warning(f"Failed to log response details: {str(e)}")

    # ---------------------- Sync Methods ---------------------- #
    def request(self, method: str, url: str, raise_for_status: bool = True, **kwargs: Any) -> httpx.Response:
        self._log_request(method, url, **kwargs)
        
        self._init_sync_client()
        assert self.sync_client is not None
        
        try:
            response = self.sync_client.request(method, url, **kwargs)
            self._log_response(method, url, response)
            
            if raise_for_status:
                response.raise_for_status()

            return response
        except httpx.HTTPStatusError as e:
            self.logger.error(
                f"HTTP error {e.response.status_code} for {method} {url}: {e.response.text[:500] if e.response.text else None}",
                error=e
            )
            raise
        except Exception as e:
            self.logger.error(f"Request failed for {method} {url}: {str(e)}", error=e)
            raise

    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("POST", url, **kwargs)

    def put(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("PUT", url, **kwargs)

    def patch(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("PATCH", url, **kwargs)

    def delete(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("DELETE", url, **kwargs)

    def head(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("HEAD", url, **kwargs)

    def options(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("OPTIONS", url, **kwargs)

    # ---------------------- Sync Streaming Methods ---------------------- #
    def stream(self, method: str, url: str, raise_for_status: bool = True, **kwargs: Any) -> Iterable[bytes]:
        self.logger.info(f"Starting streaming {method} request to {url}")
        
        self._init_sync_client()
        assert self.sync_client is not None
        
        response = None
        try:
            response = self.sync_client.stream(method, url, **kwargs)
            self._log_response(method, url, response)
            
            if raise_for_status:
                response.raise_for_status()
            
            for chunk in response.iter_bytes():
                yield chunk
                
            self.logger.info(f"Stream completed for {method} {url}")
                           
        except httpx.HTTPStatusError as e:
            self.logger.error(f"HTTP error {e.response.status_code} in stream for {method} {url}", error=e)
            raise
        except Exception as e:
            self.logger.error(f"Stream failed for {method} {url}: {str(e)}", error=e)
            raise
        finally:
            if response:
                response.close()

    def get_stream(self, url: str, **kwargs: Any) -> Iterable[bytes]:
        return self.stream("GET", url, **kwargs)

    def post_stream(self, url: str, **kwargs: Any) -> Iterable[bytes]:
        return self.stream("POST", url, **kwargs)

    def put_stream(self, url: str, **kwargs: Any) -> Iterable[bytes]:
        return self.stream("PUT", url, **kwargs)

    def patch_stream(self, url: str, **kwargs: Any) -> Iterable[bytes]:
        return self.stream("PATCH", url, **kwargs)

    def delete_stream(self, url: str, **kwargs: Any) -> Iterable[bytes]:
        return self.stream("DELETE", url, **kwargs)

    # ---------------------- Async Methods ---------------------- #
    async def arequest(self, method: str, url: str, raise_for_status: bool = True, **kwargs: Any) -> httpx.Response:
        self._log_request(method, url, **kwargs)
        
        await self._init_async_client()
        assert self.async_client is not None
        
        task = asyncio.create_task(self.async_client.request(method, url, **kwargs))
        async with self._lock:
            self._tasks.add(task)
        
        try:
            response = await task
            self._log_response(method, url, response)
            
            if raise_for_status:
                response.raise_for_status()
                
            return response
        except httpx.HTTPStatusError as e:
            self.logger.error(
                f"HTTP error {e.response.status_code} for async {method} {url}: {e.response.text[:500] if e.response.text else None}",
                error=e
            )
            raise
        except Exception as e:
            self.logger.error(f"Async request failed for {method} {url}: {str(e)}", error=e)
            raise
        finally:
            async with self._lock:
                self._tasks.discard(task)

    async def aget(self, url: str, **kwargs: Any) -> httpx.Response:
        return await self.arequest("GET", url, **kwargs)

    async def apost(self, url: str, **kwargs: Any) -> httpx.Response:
        return await self.arequest("POST", url, **kwargs)

    async def aput(self, url: str, **kwargs: Any) -> httpx.Response:
        return await self.arequest("PUT", url, **kwargs)

    async def apatch(self, url: str, **kwargs: Any) -> httpx.Response:
        return await self.arequest("PATCH", url, **kwargs)

    async def adelete(self, url: str, **kwargs: Any) -> httpx.Response:
        return await self.arequest("DELETE", url, **kwargs)

    async def ahead(self, url: str, **kwargs: Any) -> httpx.Response:
        return await self.arequest("HEAD", url, **kwargs)

    async def aoptions(self, url: str, **kwargs: Any) -> httpx.Response:
        return await self.arequest("OPTIONS", url, **kwargs)

    # ---------------------- Async Streaming Methods ---------------------- #
    async def astream(self, method: str, url: str, raise_for_status: bool = True, **kwargs: Any) -> AsyncIterable[bytes]:
        self.logger.info(f"Starting async streaming {method} request to {url}")
        
        await self._init_async_client()
        assert self.async_client is not None
        
        try:
            async with self.async_client.stream(method, url, **kwargs) as response:
                async with self._lock:
                    self._active_streaming_responses.add(response)
                
                try:
                    self._log_response(method, url, response)
                    
                    if raise_for_status:
                        response.raise_for_status()
                    
                    async for chunk in response.aiter_bytes():
                        yield chunk
                        
                    self.logger.info(f"Async stream completed for {method} {url}")
                                   
                except httpx.HTTPStatusError as e:
                    self.logger.error(f"HTTP error {e.response.status_code} in async stream for {method} {url}", error=e)
                    raise
                except Exception as e:
                    self.logger.error(f"Async stream failed for {method} {url}: {str(e)}", error=e)
                    raise
                finally:
                    async with self._lock:
                        self._active_streaming_responses.discard(response)
        except Exception as e:
            self.logger.error(f"Failed to establish async stream for {method} {url}: {str(e)}", error=e)
            raise

    async def aget_stream(self, url: str, **kwargs: Any) -> AsyncIterable[bytes]:
        async for chunk in self.astream("GET", url, **kwargs):
            yield chunk

    async def apost_stream(self, url: str, **kwargs: Any) -> AsyncIterable[bytes]:
        async for chunk in self.astream("POST", url, **kwargs):
            yield chunk

    async def aput_stream(self, url: str, **kwargs: Any) -> AsyncIterable[bytes]:
        async for chunk in self.astream("PUT", url, **kwargs):
            yield chunk

    async def apatch_stream(self, url: str, **kwargs: Any) -> AsyncIterable[bytes]:
        async for chunk in self.astream("PATCH", url, **kwargs):
            yield chunk

    async def adelete_stream(self, url: str, **kwargs: Any) -> AsyncIterable[bytes]:
        async for chunk in self.astream("DELETE", url, **kwargs):
            yield chunk

    # ---------------------- Cancel Async Requests ---------------------- #
    async def cancel_all_requests(self) -> None:
        """Cancel all pending async tasks and streaming responses"""
        self.logger.warning("Cancelling all active async requests")
        
        async with self._lock:
            # Cancel tasks
            for task in self._tasks:
                if not task.cancelled():
                    task.cancel()
            self._tasks.clear()
            
            # Close active streaming responses
            for response in self._active_streaming_responses:
                try:
                    await response.aclose()
                except Exception as e:
                    self.logger.warning(f"Error closing streaming response: {str(e)}")
            self._active_streaming_responses.clear()

    # ---------------------- Cleanup ---------------------- #
    def close(self) -> None:
        self.logger.info("Closing sync HTTP client")
        if self.sync_client:
            try:
                self.sync_client.close()
                self.sync_client = None
            except Exception as e:
                self.logger.error(f"Error closing sync HTTP client: {str(e)}", error=e)

    async def aclose(self) -> None:
        self.logger.info("Closing async HTTP client")
        try:
            await self.cancel_all_requests()
            if self.async_client:
                await self.async_client.aclose()
                self.async_client = None
        except Exception as e:
            self.logger.error(f"Error closing async HTTP client: {str(e)}", error=e)
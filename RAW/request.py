import httpx
import asyncio
from typing import Any, Dict, Optional, Iterable, AsyncIterable, Set, Union
from types import TracebackType
from .logger import Logger

class HTTPClient:
    HTTPStatusError = httpx.HTTPStatusError
    Timeout = httpx.Timeout

    def __init__(self,base_url: str = "",headers: Optional[Dict[str, str]] = None,timeout: Union[float, Timeout] = 10.0, logger: Optional[Logger] = None, **client_kwargs: Any):
        self.base_url = base_url.rstrip("/")
        self.headers = headers or {}
        self.timeout = timeout
        self.client_kwargs = client_kwargs
        self.logger:Logger = logger if logger else Logger(name="request_logger")
        self.sync_client: Optional[httpx.Client] = None
        self.async_client: Optional[httpx.AsyncClient] = None
        self._tasks: Set[asyncio.Task] = set()
        self._active_streaming_responses: Set[httpx.Response] = set()
        self._lock = asyncio.Lock()

    def __enter__(self) -> "HTTPClient":
        self._init_sync_client()
        return self

    def __exit__(self,exc_type: Optional[type[BaseException]],exc_val: Optional[BaseException],exc_tb: Optional[TracebackType]) -> None:
        self.logger.info_sync("Closed HTTP Client")
        self.close()

    async def __aenter__(self) -> "HTTPClient":
        await self._init_async_client()
        return self

    async def __aexit__(self,exc_type: Optional[type[BaseException]],exc_val: Optional[BaseException],exc_tb: Optional[TracebackType]) -> None:
        await self.logger.info("Closed Sync HTTP Client")
        await self.aclose()

    def _init_sync_client(self) -> None:
        if self.sync_client is None:
            self.sync_client = httpx.Client(
                base_url=self.base_url,
                headers=self.headers,
                timeout=self.timeout,
                **self.client_kwargs
            )
            self.logger.info_sync("Initialized HTTP Client")

    async def _init_async_client(self) -> None:
        if self.async_client is None:
            self.async_client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self.headers,
                timeout=self.timeout,
                **self.client_kwargs
            )
            await self.logger.info("Initialized Sync HTTP Client")

    # ---------------------- Sync Methods ---------------------- #
    def request(self, method: str, url: str, raise_for_status: bool = True,**kwargs: Any) -> httpx.Response:
        self._init_sync_client()
        assert self.sync_client is not None
        
        response = self.sync_client.request(method, url, **kwargs)
        if raise_for_status:
            response.raise_for_status()
        return response

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
    def stream(self,method: str,url: str,raise_for_status: bool = True,**kwargs: Any) -> Iterable[bytes]:
        self._init_sync_client()
        assert self.sync_client is not None
        
        response = self.sync_client.stream(method, url, **kwargs)
        try:
            if raise_for_status:
                response.raise_for_status()
            for chunk in response.iter_bytes():
                yield chunk
        finally:
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
    async def arequest(self,method: str,url: str,raise_for_status: bool = True,**kwargs: Any) -> httpx.Response:
        await self._init_async_client()
        assert self.async_client is not None
        
        task = asyncio.create_task(self.async_client.request(method, url, **kwargs))
        async with self._lock:
            self._tasks.add(task)
        
        try:
            response = await task
            if raise_for_status:
                response.raise_for_status()
            return response
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
    async def astream(self,method: str,url: str,raise_for_status: bool = True,**kwargs: Any) -> AsyncIterable[bytes]:
        await self._init_async_client()
        assert self.async_client is not None
        
        # FIX: Use stream() method instead of request(stream=True)
        async with self.async_client.stream(method, url, **kwargs) as response:
            async with self._lock:
                self._active_streaming_responses.add(response)
            
            try:
                if raise_for_status:
                    response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    yield chunk
            finally:
                async with self._lock:
                    self._active_streaming_responses.discard(response)

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
        async with self._lock:
            # Cancel tasks
            for task in self._tasks:
                task.cancel()
            self._tasks.clear()
            
            # Close active streaming responses
            for response in self._active_streaming_responses:
                await response.aclose()
            self._active_streaming_responses.clear()

    # ---------------------- Cleanup ---------------------- #
    def close(self) -> None:
        if self.sync_client:
            self.sync_client.close()
            self.sync_client = None

    async def aclose(self) -> None:
        await self.cancel_all_requests()
        if self.async_client:
            await self.async_client.aclose()
            self.async_client = None
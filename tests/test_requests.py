import pytest
import httpx
import asyncio
from RAW import HTTPClient
from unittest.mock import MagicMock, patch, AsyncMock

# Test data
TEST_BASE_URL = "https://api.example.com"
TEST_RESPONSE = httpx.Response(200, content=b'{"id": 1, "name": "test"}', request=httpx.Request("GET", "https://example.com"))
TEST_STREAM_CHUNKS = [b"chunk1", b"chunk2", b"chunk3"]

@pytest.fixture
def sync_client():
    with HTTPClient(base_url=TEST_BASE_URL) as client:
        yield client

@pytest.fixture
async def async_client():
    async with HTTPClient(base_url=TEST_BASE_URL) as client:
        yield client

def test_sync_get_request(httpx_mock, sync_client):
    """Test synchronous GET request"""
    httpx_mock.add_response(url=f"{TEST_BASE_URL}/test", json={"id": 1})
    
    response = sync_client.get("/test")
    assert response.status_code == 200
    assert response.json()["id"] == 1

def test_sync_stream_request(sync_client):
    """Test synchronous streaming request"""
    # Mock response with streaming content
    mock_response = MagicMock()
    mock_response.iter_bytes.return_value = iter(TEST_STREAM_CHUNKS)
    mock_response.status_code = 200
    mock_response.is_stream_consumed = False
    mock_response.close = MagicMock()
    
    # Patch the client's stream method
    with patch.object(sync_client.sync_client, 'stream', return_value=mock_response):
        chunks = list(sync_client.get_stream("/stream"))
        assert chunks == TEST_STREAM_CHUNKS
        mock_response.close.assert_called_once()

@pytest.mark.asyncio
async def test_async_get_request(httpx_mock):
    """Test asynchronous GET request"""
    # Create client instance directly
    client = HTTPClient(base_url=TEST_BASE_URL)
    await client._init_async_client()
    
    httpx_mock.add_response(url=f"{TEST_BASE_URL}/test", json={"id": 1})
    
    response = await client.aget("/test")
    assert response.status_code == 200
    assert response.json()["id"] == 1

@pytest.mark.asyncio
async def test_async_stream_request():
    """Test asynchronous streaming request"""
    # Create client instance
    client = HTTPClient(base_url=TEST_BASE_URL)
    await client._init_async_client()
    
    # Create a proper async iterator
    async def mock_aiter_bytes():
        for chunk in TEST_STREAM_CHUNKS:
            yield chunk
    
    # Create mock response
    mock_response = AsyncMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.aiter_bytes = mock_aiter_bytes
    mock_response.aclose = AsyncMock()
    
    # Create mock stream context manager
    mock_stream = AsyncMock()
    mock_stream.__aenter__.return_value = mock_response
    
    # Mock the stream method
    with patch.object(client.async_client, 'stream', return_value=mock_stream):
        # Test the stream
        chunks = []
        async for chunk in client.aget_stream("/stream"):
            chunks.append(chunk)
        
        assert chunks == TEST_STREAM_CHUNKS

@pytest.mark.asyncio
async def test_cancellation():
    """Test request cancellation"""
    client = HTTPClient(base_url=TEST_BASE_URL)
    await client._init_async_client()  # Initialize async client
    
    # Create mock task
    mock_task = asyncio.create_task(asyncio.sleep(3600))
    client._tasks.add(mock_task)
    
    # Create mock streaming response
    mock_response = AsyncMock()
    mock_response.aclose = AsyncMock()
    client._active_streaming_responses.add(mock_response)
    
    # Cancel all requests
    await client.cancel_all_requests()
    
    # Verify cancellation
    with pytest.raises(asyncio.CancelledError):
        await mock_task
    mock_response.aclose.assert_awaited_once()
    assert len(client._tasks) == 0
    assert len(client._active_streaming_responses) == 0

@pytest.mark.asyncio
async def test_error_handling(httpx_mock):
    """Test HTTP error handling"""
    # Create client instance directly
    client = HTTPClient(base_url=TEST_BASE_URL)
    await client._init_async_client()
    
    httpx_mock.add_response(
        url=f"{TEST_BASE_URL}/error",
        status_code=404
    )
    
    with pytest.raises(httpx.HTTPStatusError):
        await client.aget("/error")

def test_lazy_initialization(httpx_mock):
    """Test client lazy initialization"""
    # Mock requests
    httpx_mock.add_response(url="https://httpbin.org/get", json={})
    httpx_mock.add_response(url="https://httpbin.org/get", json={})  # For async request
    
    client = HTTPClient()
    assert client.sync_client is None
    assert client.async_client is None
    
    # Sync client initialization
    response = client.get("https://httpbin.org/get")
    assert client.sync_client is not None
    assert response.status_code == 200
    
    # Async client initialization
    async def test_async():
        response = await client.aget("https://httpbin.org/get")
        assert client.async_client is not None
        assert response.status_code == 200
    
    asyncio.run(test_async())

def test_context_managers(httpx_mock):
    """Test context manager cleanup"""
    httpx_mock.add_response(url="https://httpbin.org/get", json={})
    
    client = HTTPClient()
    with client:
        client.get("https://httpbin.org/get")
        assert client.sync_client is not None
        assert not client.sync_client.is_closed
    
    # Client should be closed and set to None
    assert client.sync_client is None

    async def async_test():
        httpx_mock.add_response(url="https://httpbin.org/get", json={})
        client = HTTPClient()
        async with client:
            await client.aget("https://httpbin.org/get")
            assert client.async_client is not None
            assert not client.async_client.is_closed
        assert client.async_client is None

    asyncio.run(async_test())

@pytest.mark.asyncio
async def test_concurrent_requests(httpx_mock):
    """Test concurrent request handling"""
    # Create client instance
    client = HTTPClient(base_url=TEST_BASE_URL)
    await client._init_async_client()
    
    # Set up mock responses
    for i in range(5):
        httpx_mock.add_response(
            url=f"{TEST_BASE_URL}/resource/{i}",
            json={"id": i}
        )
    
    # Create multiple requests
    tasks = [
        client.aget(f"/resource/{i}")
        for i in range(5)
    ]
    
    responses = await asyncio.gather(*tasks)
    
    # Verify responses
    for i, response in enumerate(responses):
        assert response.json()["id"] == i
        assert response.status_code == 200

def test_sync_stream_cleanup(sync_client):
    """Test sync streaming response cleanup"""
    mock_response = MagicMock()
    mock_response.iter_bytes.return_value = iter([b"data"])
    mock_response.status_code = 200
    mock_response.close = MagicMock()
    
    with patch.object(sync_client.sync_client, 'stream', return_value=mock_response):
        # Consume the stream
        list(sync_client.get_stream("/stream"))
        # Ensure response was closed
        mock_response.close.assert_called_once()

# @pytest.mark.asyncio
# async def test_async_stream_cleanup():
#     """Test async streaming response cleanup"""
#     # Create client instance
#     client = HTTPClient(base_url=TEST_BASE_URL)
#     await client._init_async_client()
    
#     # Create a proper async iterator
#     async def mock_aiter_bytes():
#         for chunk in TEST_STREAM_CHUNKS:
#             yield chunk
    
#     # Create mock response
#     mock_response = AsyncMock()
#     mock_response.raise_for_status = MagicMock()
#     # Set aiter_bytes as a method that returns the async iterator
#     mock_response.aiter_bytes = AsyncMock(return_value=mock_aiter_bytes())
#     mock_response.aclose = AsyncMock()
    
#     # Create mock stream context manager
#     mock_stream = AsyncMock()
#     mock_stream.__aenter__.return_value = mock_response
    
#     # Mock the stream method
#     with patch.object(client.async_client, 'stream', return_value=mock_stream):
#         # Test the stream
#         async for _ in client.aget_stream("/stream"):
#             pass
        
#         # Ensure response was closed
#         mock_response.aclose.assert_awaited_once()

def test_sync_error_handling(httpx_mock, sync_client):
    """Test sync error handling"""
    httpx_mock.add_response(
        url=f"{TEST_BASE_URL}/error",
        status_code=500
    )
    
    with pytest.raises(httpx.HTTPStatusError):
        sync_client.get("/error")

def test_sync_methods(sync_client, httpx_mock):
    """Test all sync HTTP methods"""
    methods = {
        "GET": sync_client.get,
        "POST": sync_client.post,
        "PUT": sync_client.put,
        "PATCH": sync_client.patch,
        "DELETE": sync_client.delete,
        "HEAD": sync_client.head,
        "OPTIONS": sync_client.options,
    }
    
    for method, func in methods.items():
        httpx_mock.add_response(
            url=f"{TEST_BASE_URL}/test",
            method=method,
            json={"method": method}
        )
        response = func("/test")
        assert response.status_code == 200

@pytest.mark.asyncio
async def test_async_methods(httpx_mock):
    """Test all async HTTP methods"""
    # Create client instance
    client = HTTPClient(base_url=TEST_BASE_URL)
    await client._init_async_client()
    
    methods = {
        "GET": client.aget,
        "POST": client.apost,
        "PUT": client.aput,
        "PATCH": client.apatch,
        "DELETE": client.adelete,
        "HEAD": client.ahead,
        "OPTIONS": client.aoptions,
    }
    
    for method, func in methods.items():
        httpx_mock.add_response(
            url=f"{TEST_BASE_URL}/test",
            method=method,
            json={"method": method}
        )
        response = await func("/test")
        assert response.status_code == 200
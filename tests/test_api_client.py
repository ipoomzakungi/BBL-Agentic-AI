import asyncio

import httpx
from openai import BadRequestError

from src.agents import _api_management_headers, _build_model_client
from src.config import DEFAULT_KNOWLEDGE_BASE, Settings


def test_client_matches_api_management_contract() -> None:
    captured_request: httpx.Request | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_request
        captured_request = request
        return httpx.Response(
            status_code=400,
            request=request,
            json={
                "error": {
                    "message": "Intentional test response",
                    "type": "invalid_request_error",
                    "code": "test_only",
                }
            },
        )

    async def send_request() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            settings = Settings(
                azure_endpoint="https://example.azure-api.net/llm/",
                azure_api_key="test-secret",
                azure_deployment="gpt-5-mini",
                knowledge_base_path=DEFAULT_KNOWLEDGE_BASE,
            )
            client = _build_model_client(settings)
            # Replace only the HTTP transport so no real network call is made.
            client._client = http_client

            try:
                await client.responses.create(
                    model=settings.azure_deployment,
                    input="Connection contract test",
                    extra_headers=_api_management_headers(settings),
                )
            except BadRequestError:
                pass

    asyncio.run(send_request())

    assert captured_request is not None
    assert str(captured_request.url) == (
        "https://example.azure-api.net/llm/responses"
    )
    assert captured_request.headers["api-key"] == "test-secret"
    assert "authorization" not in captured_request.headers

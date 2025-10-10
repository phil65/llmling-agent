from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import anyenv
import pytest

from llmling_agent_toolsets.openapi import OpenAPITools


if TYPE_CHECKING:
    from jsonschema_path.typing import Schema


BASE_URL = "https://api.example.com"
PETSTORE_SPEC: Schema = {
    "openapi": "3.0.0",
    "info": {"title": "Pet Store API", "version": "1.0.0"},
    "paths": {
        "/pet/{petId}": {
            "get": {
                "operationId": "get_pet",
                "summary": "Get pet by ID",
                "description": "Get pet by ID",
                "parameters": [
                    {
                        "name": "petId",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "integer"},
                        "description": "ID of pet to find",
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Pet found",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["id", "name"],
                                    "properties": {
                                        "id": {"type": "integer"},
                                        "name": {"type": "string"},
                                    },
                                }
                            }
                        },
                    }
                },
            }
        }
    },
    "servers": [{"url": BASE_URL, "description": "Pet store API server"}],
}


# Create mock httpx response
class MockResponse:
    """Mock httpx response."""

    def __init__(self):
        self.status_code = 200
        self._text = anyenv.dump_json(PETSTORE_SPEC)

    @property
    def text(self):
        return self._text

    def raise_for_status(self):
        pass


@pytest.fixture
def mock_openapi_spec(tmp_path):
    """Set up OpenAPI spec mocking and local file."""
    from openapi_spec_validator import validate

    validate(PETSTORE_SPEC)
    local_spec = tmp_path / "openapi.json"  # Create local spec file
    local_spec.write_text(anyenv.dump_json(PETSTORE_SPEC))
    url = f"{BASE_URL}/openapi.json"
    return {"local_path": str(local_spec), "remote_url": url}


async def test_openapi_toolset_local(mock_openapi_spec):
    """Test OpenAPI toolset with local file."""
    from openapi_spec_validator import validate

    local_path = mock_openapi_spec["local_path"]
    toolset = OpenAPITools(spec=local_path, base_url=BASE_URL)
    spec = await toolset._load_spec()  # Load and validate spec
    validate(spec)
    tools = await toolset.get_tools()
    assert len(tools) == 1, f"Expected 1 tool, got {len(tools)}: {tools}"


async def test_openapi_toolset_remote(mock_openapi_spec, caplog, monkeypatch):
    """Test OpenAPI toolset with remote spec."""
    from openapi_spec_validator import validate

    caplog.set_level("DEBUG")
    url = mock_openapi_spec["remote_url"]
    mock_response = MockResponse()
    mock_client = MagicMock()
    mock_get = AsyncMock(return_value=mock_response)
    mock_client.get = mock_get

    # Create sync client factory (no need for async here)
    def mock_client_factory(*args, **kwargs):
        return mock_client

    monkeypatch.setattr("httpx.AsyncClient", mock_client_factory)
    toolset = OpenAPITools(spec=url, base_url=BASE_URL)
    spec = await toolset._load_spec()
    validate(spec)
    mock_get.assert_called_once_with(url)  # Verify mocks were called correctly
    tools = await toolset.get_tools()
    assert len(tools) == 1, f"Expected 1 tool, got {len(tools)}: {tools}"


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])

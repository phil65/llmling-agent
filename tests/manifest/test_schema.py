from __future__ import annotations

import pytest

from agentpool import AgentsManifest


def test_schema_generation():
    """Test that JSON schema can be generated from config models."""
    try:
        _schema = AgentsManifest.model_json_schema()
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"Failed to generate schema: {exc}")


if __name__ == "__main__":
    pytest.main([__file__])

"""OpenAPI reference resolver with HTTP-aware resolution."""

from __future__ import annotations

from typing import Any, Self
from urllib.parse import urljoin

import httpx
import yamling

from llmling_agent.log import get_logger


logger = get_logger(__name__)


class OpenAPIResolver:
    """Resolves $ref references in OpenAPI specs, including external HTTP refs."""

    def __init__(self, base_url: str, timeout: float = 30.0):
        """Initialize resolver.

        Args:
            base_url: Base URL of the OpenAPI spec (used for relative refs).
            timeout: HTTP request timeout in seconds.
        """
        self.base_url = base_url
        self.timeout = timeout
        self._cache: dict[str, dict[str, Any]] = {}
        self._client: httpx.Client | None = None
        self._resolving: set[str] = set()  # Track refs being resolved to detect cycles

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout, follow_redirects=True)
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def resolve(self, spec: dict[str, Any]) -> dict[str, Any]:
        """Fully resolve all $ref in the spec.

        Args:
            spec: The OpenAPI spec dictionary.

        Returns:
            A new dictionary with all refs resolved.
        """
        self._resolving.clear()
        return self._resolve_value(spec, self.base_url)  # type: ignore[no-any-return]

    def _resolve_value(self, value: Any, current_url: str) -> Any:
        """Recursively resolve refs in a value."""
        match value:
            case dict() if "$ref" in value:
                return self._resolve_ref(value["$ref"], current_url)
            case dict():
                return {k: self._resolve_value(v, current_url) for k, v in value.items()}
            case list():
                return [self._resolve_value(item, current_url) for item in value]
            case _:
                return value

    def _resolve_ref(self, ref: str, current_url: str) -> dict[str, Any]:
        """Resolve a single $ref.

        Args:
            ref: The reference string (e.g., '../parameters/foo.yml#/codes').
            current_url: URL of the document containing this ref.

        Returns:
            The resolved schema/object.
        """
        # Split ref into URL part and JSON pointer
        url_part, pointer = ref.split("#", 1) if "#" in ref else (ref, "")
        # Handle internal refs (same document)
        if not url_part:
            # Internal ref like #/components/schemas/Foo
            # Keep as-is, will be resolved in second pass
            return {"$ref": ref}
        # Build absolute URL with pointer for cycle detection
        abs_url = urljoin(current_url, url_part)
        full_ref = f"{abs_url}#{pointer}" if pointer else abs_url
        if full_ref in self._resolving:  # Detect circular references
            logger.debug("Circular reference detected: %s", full_ref)
            return {"$ref": ref}  # Keep original ref to break cycle

        self._resolving.add(full_ref)
        try:
            doc = self._fetch_document(abs_url)  # Fetch and cache the external document
            result = _navigate_pointer(doc, pointer) if pointer else doc  # nav to the pointer
            # Recursively resolve any refs in the fetched content
            return self._resolve_value(result, abs_url)  # type: ignore[no-any-return]
        finally:
            self._resolving.discard(full_ref)

    def _fetch_document(self, url: str) -> dict[str, Any]:
        """Fetch and cache an external document."""
        if url in self._cache:
            return self._cache[url]

        logger.debug("Fetching external ref: %s", url)
        try:
            response = self.client.get(url)
            response.raise_for_status()
            content = response.text
            doc = yamling.load_yaml(content, verify_type=dict)
            self._cache[url] = doc
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to fetch %s: %s", url, e)
            # Return empty dict to avoid breaking the whole spec
            return {}
        else:
            return doc


def _navigate_pointer(doc: dict[str, Any], pointer: str) -> Any:
    """Navigate a JSON pointer within a document.

    Args:
        doc: The document to navigate.
        pointer: JSON pointer like '/codes' or '/components/schemas/Foo'.

    Returns:
        The value at the pointer location.
    """
    if not pointer or pointer == "/":
        return doc

    # Remove leading slash and split
    parts = pointer.lstrip("/").split("/")
    current: Any = doc

    for part in parts:
        # Handle JSON pointer escaping
        cleaned = part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict):
            current = current.get(cleaned, {})
        elif isinstance(current, list):
            try:
                current = current[int(cleaned)]
            except (ValueError, IndexError):
                return {}
        else:
            return {}

    return current


def resolve_openapi_refs(
    spec: dict[str, Any],
    base_url: str,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Resolve all $ref in an OpenAPI spec.

    Args:
        spec: The OpenAPI spec dictionary.
        base_url: Base URL for resolving relative references.
        timeout: HTTP timeout for fetching external refs.

    Returns:
        Spec with all external refs resolved.
    """
    with OpenAPIResolver(base_url, timeout) as resolver:
        resolved = resolver.resolve(spec)

    # Second pass: resolve internal refs now that externals are inlined
    return _resolve_internal_refs(resolved)


def _resolve_internal_refs(spec: dict[str, Any]) -> dict[str, Any]:
    """Resolve internal #/components/... refs."""

    def resolve_value(value: Any) -> Any:
        match value:
            case dict() if "$ref" in value:
                ref = value["$ref"]
                if isinstance(ref, str) and ref.startswith("#/"):
                    return _navigate_internal_ref(spec, ref)
                return value
            case dict():
                return {k: resolve_value(v) for k, v in value.items()}
            case list():
                return [resolve_value(item) for item in value]
            case _:
                return value

    return resolve_value(spec)  # type: ignore[no-any-return]


def _navigate_internal_ref(spec: dict[str, Any], ref: str) -> Any:
    """Navigate an internal ref like #/components/schemas/Foo."""
    if not ref.startswith("#/"):
        return {"$ref": ref}

    parts = ref[2:].split("/")
    current: Any = spec

    for part in parts:
        cleaned = part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict):
            current = current.get(cleaned)
            if current is None:
                return {"$ref": ref}  # Keep unresolved
        else:
            return {"$ref": ref}

    return current

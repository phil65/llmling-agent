"""Base class for component registries."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from typing import TYPE_CHECKING, Any, TypeVar

from psygnal.containers import EventedDict


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator, Sequence

    from psygnal.containers import DictEvents


TKey = TypeVar("TKey", str, int)


class LLMLingError(Exception):
    """Base exception for all llmling errors."""


class BaseRegistry[TKey, TItem](MutableMapping[TKey, TItem], ABC):
    """Base class for registries providing item storage and change notifications.

    This registry implements a dictionary-like interface backed by an EventedDict,
    providing automatic event emission for all mutations (additions, removals,
    modifications).

    Features:
    - Dictionary-like access (registry[key] = item)
    - Event emission for all changes
    - Item validation
    - Type safety
    - Customizable error handling

    Available events (accessed via .events):
        - adding(key, value): Before an item is added
        - added(key, value): After an item is added
        - removing(key, value): Before an item is removed
        - removed(key, value): After an item is removed
        - changing(key, value): Before an item is modified
        - changed(key, value): After an item is modified

    To implement, override:
    - _validate_item: Custom validation/transformation of items
    - _error_class: Custom error type for exceptions
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._items = EventedDict[TKey, TItem]()
        self._initialized = False
        self._configs: dict[TKey, Any] = {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._items})"

    @property
    def is_empty(self) -> bool:
        """Check if registry has any items."""
        return not bool(self._items)

    def has_item(self, key: TKey) -> bool:
        """Check if an item is registered."""
        return key in self._items

    @property
    def events(self) -> DictEvents:
        """Access to all dictionary events."""
        return self._items.events

    def register(self, key: TKey, item: TItem | Any, replace: bool = False) -> None:
        """Register an item."""
        if key in self._items and not replace:
            msg = f"Item already registered: {key}"
            raise self._error_class(msg)

        validated_item = self._validate_item(item)
        self._items[key] = validated_item

    def get(self, key: TKey) -> TItem:  # type: ignore
        """Get an item by key."""
        return self[key]

    def list_items(self) -> Sequence[TKey]:
        """List all registered item keys."""
        return list(self._items.keys())

    def reset(self) -> None:
        """Reset registry to initial state."""
        self._items.clear()
        self._configs.clear()
        self._initialized = False

    async def startup(self) -> None:
        """Initialize all registered items."""
        if self._initialized:
            return

        try:
            for item in self._items.values():
                await self._initialize_item(item)
            self._initialized = True
        except Exception as exc:
            await self.shutdown()
            msg = f"Registry startup failed: {exc}"
            raise self._error_class(msg) from exc

    async def shutdown(self) -> None:
        """Cleanup all registered items."""
        if not self._initialized:
            return

        errors: list[tuple[TKey, Exception]] = []

        for key, item in self._items.items():
            try:
                await self._cleanup_item(item)
            except Exception as exc:  # noqa: BLE001
                errors.append((key, exc))

        self._initialized = False

        if errors:
            error_msgs = [f"{key}: {exc}" for key, exc in errors]
            msg = f"Errors during shutdown: {', '.join(error_msgs)}"
            raise self._error_class(msg)

    @property
    def _error_class(self) -> type[LLMLingError]:
        """Error class to use for this registry."""
        return LLMLingError

    @abstractmethod
    def _validate_item(self, item: Any) -> TItem:
        """Validate and possibly transform item before registration."""

    async def _initialize_item(self, item: TItem) -> None:
        """Initialize an item during startup."""
        if hasattr(item, "startup") and callable(item.startup):  # pyright: ignore
            await item.startup()  # pyright: ignore  # ty: ignore

    async def _cleanup_item(self, item: TItem) -> None:
        """Clean up an item during shutdown."""
        if hasattr(item, "shutdown") and callable(item.shutdown):  # pyright: ignore
            await item.shutdown()  # pyright: ignore  # ty: ignore

    # Implementing MutableMapping methods
    def __getitem__(self, key: TKey) -> TItem:
        try:
            return self._items[key]
        except KeyError as exc:
            msg = f"Item not found: {key}"
            raise self._error_class(msg) from exc

    def __setitem__(self, key: TKey, value: Any) -> None:
        """Support dict-style assignment."""
        self.register(key, value)

    def __contains__(self, key: object) -> bool:
        """Support 'in' operator without raising exceptions."""
        return key in self._items

    def __delitem__(self, key: TKey) -> None:
        if key in self._items:
            del self._items[key]
        else:
            msg = f"Item not found: {key}"
            raise self._error_class(msg)

    def __iter__(self) -> Iterator[TKey]:
        return iter(self._items)

    async def __aiter__(self) -> AsyncIterator[tuple[TKey, TItem]]:
        """Async iterate over items, ensuring they're initialized."""
        if not self._initialized:
            await self.startup()
        for key, item in self._items.items():
            yield key, item

    def __len__(self) -> int:
        return len(self._items)

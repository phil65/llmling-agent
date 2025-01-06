from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, TypeVar

from fieldz import fields, get_adapter
from pydantic import BaseModel


T = TypeVar("T")


class PromptConvertible(Protocol):
    """Protocol for instances that can be converted to prompts."""

    def __prompt__(self) -> str: ...


class PromptTypeConvertible(Protocol):
    """Protocol for types that can be converted to prompts."""

    @classmethod
    def __prompt_type__(cls) -> str: ...


class FieldFormattable(Protocol):
    """Protocol for types that can be formatted through their fields."""

    __annotations__: dict[str, Any]


AnyPromptType = (
    str
    | PromptConvertible
    | PromptTypeConvertible
    | FieldFormattable
    | BaseModel
    | dict[str, Any]
    | list[Any]
    | tuple[Any, ...]
    | Callable[..., str]
)


def to_prompt(obj: AnyPromptType) -> str:  # noqa: PLR0911
    """Convert any supported type to a prompt string.

    Args:
        obj: Object to convert

    Examples:
        >>> to_prompt("Hello")
        'Hello'

        >>> class Greeter:
        ...     def __prompt__(self) -> str:
        ...         return "Hello!"
        ...     @classmethod
        ...     def __prompt_type__(cls) -> str:
        ...         return "Greeter class that says hello"
        >>> to_prompt(Greeter())  # Instance prompt
        'Hello!'
        >>> to_prompt(Greeter)    # Type prompt
        'Greeter class that says hello'
    """
    match obj:
        case str():
            return obj

        case type() if hasattr(obj, "__prompt_type__"):
            return obj.__prompt_type__()

        case _ if hasattr(obj, "__prompt__"):
            return obj.__prompt__()  # pyright: ignore[reportAttributeAccessIssue]

        case _ if can_format_fields(obj):
            return format_instance_for_llm(obj)

        case dict():
            formatted = [f"{k}: {to_prompt(v)}" for k, v in obj.items()]
            return "\n".join(formatted)

        case list() | tuple():
            return "\n".join(to_prompt(item) for item in obj)

        case _ if callable(obj):
            return obj()

        case _:
            return str(obj)


def format_instance_for_llm(obj: Any) -> str:
    """Format object instance showing structure and current values."""
    try:
        obj_fields = fields(obj)
    except TypeError:
        return f"Unable to inspect fields of {type(obj)}"

    lines = [f"{type(obj).__name__}:\n{type(obj).__doc__}\n"]

    for field in obj_fields:
        if field.name.startswith("_"):
            continue
        value = getattr(obj, field.name)
        if field.description:
            lines.append(f"- {field.name} = {value!r} ({field.description})")
        else:
            type_name = field.type if field.type else "any"
            lines.append(f"- {field.name} = {value!r} ({type_name})")

    return "\n".join(lines)


def can_format_fields(obj: Any) -> bool:
    """Check if object can be inspected by fieldz."""
    try:
        get_adapter(obj)
    except TypeError:
        return False
    else:
        return True


if __name__ == "__main__":
    from dataclasses import dataclass

    from pydantic import Field

    class Greeter:
        def __prompt__(self) -> str:
            return "Hello from instance!"

        @classmethod
        def __prompt_type__(cls) -> str:
            return "The Greeter class says hello"

    @dataclass
    class User:
        """A user in the system."""

        name: str
        age: int

    class Config(BaseModel):
        """System configuration."""

        host: str = Field(description="Server hostname")
        port: int = Field(default=8080, description="Server port")

    def get_greeting() -> str:
        return "Hello from callable!"

    # Test different types
    print(to_prompt("Simple string"))
    print("\n" + "=" * 50 + "\n")
    print(to_prompt(Greeter()))  # Instance
    print(to_prompt(Greeter))  # Class
    print("\n" + "=" * 50 + "\n")
    print(to_prompt(User("Alice", 30)))
    print("\n" + "=" * 50 + "\n")
    print(to_prompt(Config(host="localhost")))
    print("\n" + "=" * 50 + "\n")
    print(to_prompt(get_greeting))
    print("\n" + "=" * 50 + "\n")
    print(to_prompt([1, 2, 3]))
    print("\n" + "=" * 50 + "\n")
    print(to_prompt({"a": 1, "b": 2}))

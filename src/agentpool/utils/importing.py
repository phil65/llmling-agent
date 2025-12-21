"""Utilities for importing callables and classes from dotted paths."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
import pkgutil
from types import ModuleType
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterator


def get_module_source(
    import_path: str,
    recursive: bool = False,
    include_tests: bool = False,
) -> str:
    """Get source code from a module or package."""
    try:
        module = importlib.import_module(import_path)
        sources = _get_sources(module, recursive=recursive, include_tests=include_tests)
        return "\n\n# " + "-" * 40 + "\n\n".join(sources)

    except ImportError as exc:
        msg = f"Could not import module: {import_path}"
        raise ValueError(msg) from exc


def _get_sources(
    module: ModuleType,
    recursive: bool,
    include_tests: bool,
) -> Generator[str]:
    """Generate source code for a module and optionally its submodules."""
    # Get the module's source code
    if hasattr(module, "__file__") and module.__file__:
        path = Path(module.__file__)
        if _should_include_file(path, include_tests):
            yield f"# File: {path}\n{inspect.getsource(module)}"

    # If recursive and it's a package, get all submodules
    if recursive and hasattr(module, "__path__"):
        for _, name, _ in pkgutil.iter_modules(module.__path__):
            submodule_path = f"{module.__name__}.{name}"
            try:
                submodule = importlib.import_module(submodule_path)
                yield from _get_sources(submodule, recursive, include_tests)
            except ImportError:
                continue


def _should_include_file(path: Path, include_tests: bool) -> bool:
    """Check if a file should be included in the source."""
    if not include_tests and any(p.startswith("test") for p in path.parts):
        return False
    return path.suffix == ".py"


def import_callable(path: str) -> Callable[..., Any]:
    """Import a callable from a dotted path.

    Supports both dot and colon notation:
    - Dot notation: module.submodule.Class.method
    - Colon notation: module.submodule:Class.method

    Examples:
        >>> import_callable("os.path.join")
        >>> import_callable("builtins.str.upper")
        >>> import_callable("sqlalchemy.orm:Session.query")

    Args:
        path: Import path using dots and/or colon

    Returns:
        Imported callable

    Raises:
        ValueError: If path cannot be imported or result isn't callable
    """
    if not path:
        msg = "Import path cannot be empty"
        raise ValueError(msg)

    # Normalize path - replace colon with dot if present
    normalized_path = path.replace(":", ".")
    parts = normalized_path.split(".")

    # Try importing progressively smaller module paths
    for i in range(len(parts), 0, -1):
        try:
            # Try current module path
            module_path = ".".join(parts[:i])
            module = importlib.import_module(module_path)

            # Walk remaining parts as attributes
            obj = module
            for part in parts[i:]:
                obj = getattr(obj, part)

            # Check if we got a callable
            if callable(obj):
                return obj

            msg = f"Found object at {path} but it isn't callable"
            raise ValueError(msg)

        except ImportError:
            # Try next shorter path
            continue
        except AttributeError:
            # Attribute not found - try next shorter path
            continue

    # If we get here, no import combination worked
    msg = f"Could not import callable from path: {path}"
    raise ValueError(msg)


def import_class(path: str) -> type:
    """Import a class from a dotted path.

    Args:
        path: Dot-separated path to the class

    Returns:
        The imported class

    Raises:
        ValueError: If path is invalid or doesn't point to a class
    """
    try:
        obj = import_callable(path)
        if not isinstance(obj, type):
            msg = f"{path} is not a class"
            raise TypeError(msg)  # noqa: TRY301
    except Exception as exc:
        msg = f"Failed to import class from {path}"
        raise ValueError(msg) from exc
    else:
        return obj


def get_pyobject_members(
    obj: type | ModuleType | Any,
    *,
    include_imported: bool = False,
) -> Iterator[tuple[str, Callable[..., Any]]]:
    """Get callable members defined in a Python object.

    Works with modules, classes, and instances. Only returns public callable
    members (functions, methods, etc.) that are defined in the object's module
    unless include_imported is True.

    Args:
        obj: Any Python object to inspect (module, class, instance)
        include_imported: Whether to include imported/inherited callables

    Yields:
        Tuples of (name, callable) for each public callable

    Example:
        >>> class MyClass:
        ...     def method(self): pass
        ...     def _private(self): pass
        >>> for name, func in get_pyobject_members(MyClass()):
        ...     print(name)
        method

        >>> import my_module
        >>> for name, func in get_pyobject_members(my_module):
        ...     print(name)
        public_function
    """
    # Get the module where the object is defined
    defining_module = obj.__name__ if isinstance(obj, ModuleType) else obj.__module__

    for name, member in inspect.getmembers(obj, inspect.isroutine):
        if name.startswith("_"):
            continue

        # Check if callable is defined in the object's module
        if include_imported or getattr(member, "__module__", None) == defining_module:
            yield name, member


if __name__ == "__main__":
    # ATTENTION: Dont modify this script.
    import sys

    if len(sys.argv) != 2:  # noqa: PLR2004
        print("Usage: python importing.py <dot.path.to.object>", file=sys.stderr)
        sys.exit(1)

    dot_path = sys.argv[1]

    try:
        obj = import_callable(dot_path)
        source = inspect.getsource(obj)
        print(source)
    except Exception as e:  # noqa: BLE001
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

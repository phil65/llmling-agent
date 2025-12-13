"""Manages stored config mappings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict

import anyenv
import platformdirs
from upathtools import to_upath

from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from upathtools import JoinablePathLike


logger = get_logger(__name__)


@dataclass(frozen=True)
class ConfigFile:
    """Represents an active configuration."""

    name: str
    path: str

    def __str__(self) -> str:
        """Format for display."""
        return f"{self.name} ({self.path})"


class ConfigMapping(TypedDict):
    """Type for config storage format."""

    configs: dict[str, str]  # name -> uri mapping
    active: str | None


class ConfigStore:
    """Manages stored config mappings."""

    def __init__(self, filename: str | None = None) -> None:
        """Initialize store with default paths."""
        llmling_dir = platformdirs.user_config_dir("llmling")
        self.config_dir = to_upath(llmling_dir)
        name = filename or "configs.json"
        self.config_file = self.config_dir / name
        self._ensure_config_dir()

    def _ensure_config_dir(self) -> None:
        """Create config directory if needed."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        if not self.config_file.exists():
            mapping = ConfigMapping(configs={}, active=None)
            self.save_mapping(mapping)

    def load_mapping(self) -> ConfigMapping:
        """Load config mapping from disk."""
        if not self.config_file.exists():
            return ConfigMapping(configs={}, active=None)
        try:
            text = self.config_file.read_text("utf-8")
            data = anyenv.load_json(text, return_type=dict)
            active = data.get("active")
            configs = data.get("configs", {})
            return ConfigMapping(configs=configs, active=active)
        except Exception:
            logger.exception("Failed to load config mapping")
            return ConfigMapping(configs={}, active=None)

    def save_mapping(self, mapping: ConfigMapping) -> None:
        """Save config mapping to disk."""
        try:
            self.config_file.write_text(anyenv.dump_json(mapping, indent=True))
        except Exception:
            logger.exception("Failed to save config mapping")

    def add_config(self, name: str, path: JoinablePathLike) -> None:
        """Add a new named config.

        Args:
            name: Name to register the config under
            path: Path to the config file

        Raises:
            ValueError: If name is invalid
            FileNotFoundError: If config file doesn't exist
            PermissionError: If config file can't be read
            IsADirectoryError: If path points to a directory
        """
        # Basic validation

        if not name.isidentifier():
            msg = f"Invalid config name: {name} (must be a valid Python identifier)"
            raise ValueError(msg)

        path_obj = to_upath(path).resolve()
        if not path_obj.exists():
            msg = f"Config file not found: {path}"
            raise FileNotFoundError(msg)

        if not path_obj.is_file():
            msg = f"Path is not a file: {path}"
            raise IsADirectoryError(msg)

        try:
            # Try to read file to verify access
            path_obj.read_bytes()
        except PermissionError as exc:
            msg = f"Cannot read config file: {path}"
            raise PermissionError(msg) from exc
        # All good, save the config
        mapping = self.load_mapping()
        mapping["configs"][name] = str(path_obj)
        logger.debug("Adding config %r -> %s", name, path_obj)
        self.save_mapping(mapping)

    def remove_config(self, name: str) -> None:
        """Remove a named config."""
        mapping = self.load_mapping()
        if name == mapping["active"]:
            mapping["active"] = None
        mapping["configs"].pop(name, None)
        self.save_mapping(mapping)

    def set_active(self, name: str) -> None:
        """Set the active config."""
        mapping = self.load_mapping()
        if name not in mapping["configs"]:
            msg = f"Config {name} not found"
            raise KeyError(msg)
        mapping["active"] = name
        self.save_mapping(mapping)

    def get_active(self) -> ConfigFile | None:
        """Get active config if one is set.

        Returns:
            ConfigFile if an active config is set, None otherwise
        """
        mapping = self.load_mapping()
        if not mapping["active"]:
            return None
        name = mapping["active"]
        return ConfigFile(name=name, path=mapping["configs"][name])

    def list_configs(self) -> list[tuple[str, str]]:
        """List all configs with their paths."""
        mapping = self.load_mapping()
        return list(mapping["configs"].items())

    def get_config(self, name: str) -> str:
        """Get path for a named config."""
        mapping = self.load_mapping()
        if name not in mapping["configs"]:
            msg = f"Config {name} not found"
            raise KeyError(msg)
        return mapping["configs"][name]


config_store = ConfigStore()

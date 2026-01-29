"""Claude Code usage limits helper.

Fetches usage information from Anthropic's OAuth API using stored credentials.
Works on both Linux and macOS.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
import sys

import anyenv
import httpx
from pydantic import BaseModel


class UsageLimit(BaseModel):
    """A single usage limit with utilization percentage and reset time."""

    utilization: float
    """Utilization percentage (0-100)."""

    resets_at: datetime | None = None
    """When this limit resets, or None if not set."""

    def format(self, name: str) -> str:
        reset_str = ""
        if self.resets_at:
            reset_str = f" (resets {self.resets_at.strftime('%Y-%m-%d %H:%M UTC')})"
        return f"{name}: {self.utilization:.0f}%{reset_str}"


class ExtraUsage(BaseModel):
    """Extra usage information for paid plans."""

    is_enabled: bool = False
    monthly_limit: float | None = None
    used_credits: float | None = None
    utilization: float | None = None


class ClaudeCodeUsage(BaseModel):
    """Claude Code usage limits."""

    five_hour: UsageLimit | None = None
    """5-hour rolling usage limit."""

    seven_day: UsageLimit | None = None
    """7-day rolling usage limit."""

    seven_day_opus: UsageLimit | None = None
    """7-day Opus-specific limit."""

    seven_day_sonnet: UsageLimit | None = None
    """7-day Sonnet-specific limit."""

    seven_day_oauth_apps: UsageLimit | None = None
    """7-day OAuth apps limit."""

    extra_usage: ExtraUsage | None = None
    """Extra usage info for paid plans."""

    def format_table(self) -> str:
        """Format usage as a readable table."""
        lines = ["Claude Code Usage Limits", "=" * 50]
        for name, limit in [
            ("5-hour", self.five_hour),
            ("7-day", self.seven_day),
            ("7-day Opus", self.seven_day_opus),
            ("7-day Sonnet", self.seven_day_sonnet),
            ("7-day OAuth Apps", self.seven_day_oauth_apps),
        ]:
            if limit and (formatted := limit.format(name)):
                lines.append(formatted)

        if self.extra_usage and self.extra_usage.is_enabled:
            lines.append("")
            lines.append("Extra Usage (paid):")
            if self.extra_usage.utilization is not None:
                lines.append(f"  Utilization: {self.extra_usage.utilization:.0f}%")
            if self.extra_usage.used_credits is not None:
                lines.append(f"  Used credits: {self.extra_usage.used_credits}")
            if self.extra_usage.monthly_limit is not None:
                lines.append(f"  Monthly limit: {self.extra_usage.monthly_limit}")

        return "\n".join(lines)


def _get_credentials_path() -> Path | None:
    """Get the path to Claude Code credentials file.

    Returns:
        Path to credentials file, or None if not found.
    """
    # Linux: ~/.claude/.credentials.json
    linux_path = Path.home() / ".claude" / ".credentials.json"
    if linux_path.exists():
        return linux_path
    # macOS: Also check ~/.claude first (newer versions)
    if linux_path.exists():
        return linux_path
    return None


def _get_access_token_from_file(path: Path) -> str | None:
    """Read access token from credentials file."""
    try:
        data = anyenv.load_json(path.read_text(), return_type=dict)
        val = data.get("claudeAiOauth", {}).get("accessToken")
    except (anyenv.JsonLoadError, OSError):
        return None
    else:
        assert isinstance(val, str)
        return val


async def _get_access_token_from_keychain() -> str | None:
    """Read access token from macOS Keychain."""
    if sys.platform != "darwin":
        return None

    try:
        proc = await asyncio.create_subprocess_exec(
            "security",
            "find-generic-password",
            "-s",
            "Claude Code-credentials",
            "-w",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode != 0:
            return None
        data = anyenv.load_json(stdout.decode().strip(), return_type=dict)
        val = data.get("claudeAiOauth", {}).get("accessToken")
    except (anyenv.JsonLoadError, FileNotFoundError, OSError):
        return None
    else:
        assert isinstance(val, str)
        return val


async def get_access_token() -> str | None:
    """Get Claude Code OAuth access token.

    Checks both file-based storage (Linux/newer macOS) and Keychain (macOS).

    Returns:
        Access token string, or None if not found.
    """
    # Try file-based first (works on Linux and newer macOS)
    if (path := _get_credentials_path()) and (token := _get_access_token_from_file(path)):
        return token
    return await _get_access_token_from_keychain()  # Fall back to macOS Keychain


async def get_usage(token: str | None = None) -> ClaudeCodeUsage:
    """Fetch Claude Code usage limits asynchronously.

    Args:
        token: OAuth access token. If not provided, will attempt to read from
               stored credentials.

    Returns:
        ClaudeCodeUsage with current limits.

    Raises:
        ValueError: If no token provided and credentials not found.
        httpx.HTTPStatusError: If API request fails.
    """
    if token is None:
        token = await get_access_token()
        if token is None:
            msg = "No Claude Code credentials found. Please authenticate with Claude Code first."
            raise ValueError(msg)

    async with httpx.AsyncClient() as client:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "claude-code/2.0.32",
            "Authorization": f"Bearer {token}",
            "anthropic-beta": "oauth-2025-04-20",
        }
        response = await client.get("https://api.anthropic.com/api/oauth/usage", headers=headers)
        response.raise_for_status()
        return ClaudeCodeUsage.model_validate(response.json())


if __name__ == "__main__":

    async def main() -> None:
        usage = await get_usage()
        print(usage.format_table())

    asyncio.run(main())

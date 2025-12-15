"""Project tasks."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Literal

from duty import duty  # pyright: ignore[reportMissingImports]


# Read configuration from copier-answers.yml
_answers_file = Path(".copier-answers.yml")
if _answers_file.exists():
    content = _answers_file.read_text()
    match = re.search(r"^python_package_import_name:\s*(.+)$", content, re.MULTILINE)
    if match:
        PACKAGE_NAME = match.group(1).strip()
    else:
        msg = "python_package_import_name not found in copier-answers.yml"
        raise ValueError(msg)
else:
    msg = "copier-answers.yml not found in project root"
    raise FileNotFoundError(msg)


@duty(capture=False)
def build(ctx, *args: str):
    """Build documentation."""
    args_str = " " + " ".join(args) if args else ""
    ctx.run(f"uv run zensical build{args_str}")
    ctx.run("uv run python scripts/reorder_nav.py")


@duty(capture=False)
def serve(ctx, *args: str):
    """Serve documentation."""
    args_str = " " + " ".join(args) if args else ""
    ctx.run(f"uv run zensical serve{args_str}")


@duty(capture=False)
def test(ctx, *args: str):
    """Run tests."""
    args_str = " " + " ".join(args) if args else ""
    args_str = " -n auto" + args_str
    ctx.run(f"uv run pytest{args_str}")


@duty(capture=False)
def clean(ctx):
    """Clean all files from the Git directory except checked-in files."""
    ctx.run("git clean -dfX")


@duty(capture=False)
def update(ctx):
    """Update all environment packages using pip directly."""
    ctx.run("uv lock --upgrade")
    ctx.run("uv sync --all-extras")


@duty(capture=False)
def lint(ctx):
    """Lint the code and fix issues if possible."""
    ctx.run("uv run ruff check --fix --unsafe-fixes .")
    ctx.run("uv run ruff format .")
    ctx.run("uv run mypy src/ --fixed-format-cache")
    ctx.run(
        "uv run check-jsonschema --schemafile schema/config-schema.json "
        "src/llmling_agent/config_resources/*.yml "
        "docs/examples/**/config.yml"
    )


@duty(capture=False)
def lint_check(ctx):
    """Lint the code."""
    ctx.run("uv run ruff check .")
    ctx.run("uv run ruff format --check .")
    ctx.run("uv run mypy src/llmling_agent/ --fixed-format-cache")
    ctx.run(
        "uv run check-jsonschema --schemafile schema/config-schema.json "
        "src/llmling_agent/config_resources/*.yml "
        "docs/examples/**/config.yml"
    )


@duty(capture=False)
def version(
    ctx,
    *bump_type: Literal["major", "minor", "patch", "stable", "alpha", "beta", "rc"],
):
    """Release a new version with git operations. (major|minor|patch|stable|alpha|beta|rc)."""
    # Check for uncommitted changes
    result = ctx.run("git status --porcelain", capture=True)
    if result.strip():
        msg = "Cannot release with uncommitted changes. Please commit or stash first."
        raise RuntimeError(msg)

    # Read current version
    old_version = ctx.run("uv version --short", capture=True).strip()
    print(f"Current version: {old_version}")
    bump_str = " ".join(f"--bump {i}" for i in bump_type)
    ctx.run(f"uv version {bump_str}")
    new_version = ctx.run("uv version --short", capture=True).strip()
    print(f"New version: {new_version}")
    ctx.run("uv lock")

    # Update extension.toml
    ext_toml = Path("distribution/zed/extension.toml")
    if ext_toml.exists():
        content = ext_toml.read_text()
        content = content.replace(old_version, new_version)
        ext_toml.write_text(content)
        ctx.run("git add distribution/zed/extension.toml")

    ctx.run("git add pyproject.toml uv.lock")
    ctx.run(f'git commit -m "chore: bump version {old_version} -> {new_version}"')

    # Create and push tag
    tag = f"v{new_version}"
    ctx.run(f"git tag {tag}")
    print(f"Created tag: {tag}")


@duty(capture=False)
def ui(ctx, *args: str):
    """Launch Textual UI."""
    args_str = " " + " ".join(args) if args else ""
    cmd_parts = [
        "uv run --project ../toad/",
        "textual run --dev ../toad/src/toad/__main__.py acp",
        f'"uv run --python 3.13 llmling-agent serve-acp --model-provider openai{args_str}"',
    ]
    ctx.run(" ".join(cmd_parts))


@duty(capture=False)
def schema_html(ctx):
    """Create HTML documentation for JSON schema."""
    ctx.run(
        "generate-schema-doc "
        "--config template_name=js "
        "--config expand_buttons=true "
        "schema/config-schema.json "
        "docs/schema/index.html"
    )

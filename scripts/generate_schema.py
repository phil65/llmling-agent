"""Generate JSON schema for config models.

Can be used:
1. As a standalone script: python tools/generate_schema.py
2. As a pre-commit hook
3. From CI
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

from llmling.core.log import get_logger

from llmling_agent import AgentsManifest


logger = get_logger(__name__)


def generate_schema(
    output_path: str | Path | None = None,
    check_only: bool = False,
    force: bool = True,
) -> tuple[bool, dict[str, Any]]:
    """Generate JSON schema for config models.

    Args:
        output_path: Where to write the schema. If None, uses default location
        check_only: Just check if schema would change, don't write
        force: Force-overwrite

    Returns:
        Tuple of (changed: bool, schema: dict)
    """
    # Get default path if none provided
    if output_path is None:
        root = Path(__file__).parent.parent
        output_path = root / "schema" / "config-schema.json"
    else:
        output_path = Path(output_path)
    logger.info("Generating schema to: %s", output_path)

    # Generate new schema
    schema = AgentsManifest.model_json_schema()
    logger.info("Generated schema")

    # Check if different from existing
    changed = True
    if output_path.exists():
        try:
            logger.info("Writing schema to %s", output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open() as f:
                current = json.load(f)
            changed = current != schema
            logger.info("Schema %s from current", "differs" if changed else "unchanged")

        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read current schema: %s", exc)

    # Write if needed
    if (changed or force) and not check_only:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(schema, f, indent=2)
        logger.info("Schema written to %s", output_path)

    return changed, schema


def main() -> int:
    """Run schema generation."""
    parser = argparse.ArgumentParser(description="Generate config schema")
    parser.add_argument(
        "--output",
        "-o",
        help="Output path (default: schema/config-schema.json)",
    )
    parser.add_argument(
        "--check",
        "-c",
        action="store_true",
        help="Check if schema would change without writing",
    )
    args = parser.parse_args()

    try:
        changed, _ = generate_schema(args.output, args.check)
        if args.check and changed:
            logger.warning("Schema would change")
            return 1
    except Exception:
        logger.exception("Schema generation failed")
        return 1
    else:
        return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())

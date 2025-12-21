"""Generate JSON schema for config models with different None-union representations.

Can be used:
1. As a standalone script: python tools/generate_schema.py
2. As a pre-commit hook
3. From CI
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any, Literal

from pydantic.json_schema import GenerateJsonSchema

from agentpool import AgentsManifest
from agentpool.log import configure_logging, get_logger


if TYPE_CHECKING:
    from pydantic.json_schema import JsonSchemaMode


logger = get_logger(__name__)


class SimpleNullableJsonSchemaGenerator(GenerateJsonSchema):
    """JSON Schema generator that uses simple type arrays for nullable fields."""

    def generate(self, schema: Any, mode: str = "validation") -> dict[str, Any]:
        """Generate schema with simplified nullable field representation."""
        json_schema = super().generate(schema, mode)

        # Convert anyOf nullable patterns to simple type arrays
        self._simplify_nullable_fields(json_schema)

        return json_schema

    def _simplify_nullable_fields(self, obj: dict[str, Any]) -> None:
        """Convert anyOf nullable patterns to type arrays."""
        if isinstance(obj, dict):
            # Check if this is a nullable anyOf pattern
            if "anyOf" in obj and len(obj["anyOf"]) == 2:  # noqa: PLR2004
                any_of = obj["anyOf"]
                types = []
                other_schemas = []

                for item in any_of:
                    if isinstance(item, dict) and "type" in item and len(item) == 1:
                        types.append(item["type"])
                    else:
                        other_schemas.append(item)

                # If we have exactly one null type and one other simple type
                if len(types) == 2 and "null" in types and len(other_schemas) == 0:  # noqa: PLR2004
                    non_null_type = next(t for t in types if t != "null")
                    # Replace anyOf with simple type array
                    del obj["anyOf"]
                    obj["type"] = [non_null_type, "null"]

            # Recurse into nested objects
            for value in obj.values():
                if isinstance(value, (dict, list)):
                    self._simplify_nullable_fields(value)
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, (dict, list)):
                    self._simplify_nullable_fields(item)


class OpenAPICompatibleJsonSchemaGenerator(GenerateJsonSchema):
    """JSON Schema generator optimized for OpenAPI/Swagger compatibility."""

    def generate(self, schema: Any, mode: JsonSchemaMode = "validation") -> dict[str, Any]:
        """Generate schema with OpenAPI-friendly nullable representation."""
        json_schema = super().generate(schema, mode)

        # Convert nullable patterns to OpenAPI style
        self._convert_to_openapi_nullable(json_schema)

        return json_schema

    def _convert_to_openapi_nullable(self, obj: dict[str, Any]) -> None:
        """Convert anyOf nullable patterns to OpenAPI nullable format."""
        if isinstance(obj, dict):
            # Check if this is a nullable anyOf pattern
            if "anyOf" in obj:
                any_of = obj["anyOf"]
                if len(any_of) == 2:  # noqa: PLR2004
                    null_item = None
                    type_item = None

                    for item in any_of:
                        if isinstance(item, dict):
                            if item.get("type") == "null":
                                null_item = item
                            elif "type" in item and len(item) == 1:
                                type_item = item

                    # If we found a simple null + type pattern
                    if null_item and type_item:
                        # Replace anyOf with type + nullable
                        del obj["anyOf"]
                        obj.update(type_item)
                        obj["nullable"] = True

            # Recurse into nested objects
            for value in obj.values():
                if isinstance(value, (dict, list)):
                    self._convert_to_openapi_nullable(value)
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, (dict, list)):
                    self._convert_to_openapi_nullable(item)


def generate_schema_variant(
    variant: Literal["default", "simple_nullable", "openapi"] = "default",
) -> dict[str, Any]:
    """Generate schema with specific None-union representation.

    Args:
        variant: Type of None-union representation
            - "default": Standard Pydantic anyOf pattern
            - "simple_nullable": Use type arrays ["string", "null"]
            - "openapi": Use type + nullable property

    Returns:
        Generated JSON schema
    """
    generator_map = {
        "default": None,
        "simple_nullable": SimpleNullableJsonSchemaGenerator,
        "openapi": OpenAPICompatibleJsonSchemaGenerator,
    }

    generator = generator_map.get(variant)

    if generator:
        return AgentsManifest.model_json_schema(schema_generator=generator)

    return AgentsManifest.model_json_schema()


def generate_schema(
    output_path: str | Path | None = None,
    check_only: bool = False,
    force: bool = True,
    variant: Literal["default", "simple_nullable", "openapi"] = "simple_nullable",
) -> tuple[bool, dict[str, Any]]:
    """Generate JSON schema for config models.

    Args:
        output_path: Where to write the schema. If None, uses default location
        check_only: Just check if schema would change, don't write
        force: Force-overwrite
        variant: Which None-union representation to use

    Returns:
        Tuple of (changed: bool, schema: dict)
    """
    # Get default path if none provided
    if output_path is None:
        root = Path(__file__).parent.parent
        output_path = root / "schema" / "config-schema.json"
    else:
        output_path = Path(output_path)

    logger.info("Generating schema", output_path=output_path, variant=variant)

    # Generate new schema
    schema = generate_schema_variant(variant)
    logger.info("Generated schema", variant=variant)

    # Check if different from existing
    changed = True
    if output_path.exists():
        try:
            with output_path.open() as f:
                current = json.load(f)
            changed = current != schema
            logger.info("Schema status", status="differs" if changed else "unchanged")

        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read current schema", error=exc)

    # Write if needed
    if (changed or force) and not check_only:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(schema, f, indent=2)
        logger.info("Schema written", output_path=output_path)

    return changed, schema


def main() -> int:
    """Run schema generation."""
    configure_logging()
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
    parser.add_argument(
        "--variant",
        "-v",
        choices=["default", "simple_nullable", "openapi"],
        default="simple_nullable",
        help="None-union representation style (default: simple_nullable)",
    )
    args = parser.parse_args()

    try:
        changed, _ = generate_schema(args.output, args.check, variant=args.variant)
        if args.check and changed:
            logger.warning("Schema would change")
            return 1
    except Exception:
        logger.exception("Schema generation failed")
        return 1
    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())

"""Helper functions and constants for documentation commands."""

from __future__ import annotations


# Script constants for client-side execution
SCHEMA_EXTRACTION_SCRIPT = """
import sys
import json
import importlib

def import_callable(path):
    \"\"\"Simple import_callable implementation.\"\"\"
    parts = path.split(".")
    for i in range(len(parts), 0, -1):
        try:
            module_path = ".".join(parts[:i])
            module = importlib.import_module(module_path)
            obj = module
            for part in parts[i:]:
                obj = getattr(obj, part)
            return obj
        except (ImportError, AttributeError):
            continue
    raise ImportError(f"Could not import %s" % path)

try:
    model_class = import_callable("{input_path}")
    schema = model_class.model_json_schema()
    print(json.dumps(schema, indent=2))
except Exception as e:
    print(f"Error: %s" % e, file=sys.stderr)
    sys.exit(1)
"""


def generate_from_url(input_path: str, class_name: str | None = None) -> str:
    """Generate Pydantic model code from an OpenAPI specification URL."""
    from datamodel_code_generator import DataModelType, LiteralType, PythonVersion
    from datamodel_code_generator.model import get_data_model_types
    from datamodel_code_generator.parser.openapi import OpenAPIParser

    data_model_types = get_data_model_types(
        DataModelType.PydanticV2BaseModel,
        target_python_version=PythonVersion.PY_312,
    )
    parser = OpenAPIParser(
        source=input_path,
        data_model_type=data_model_types.data_model,
        data_model_root_type=data_model_types.root_model,
        data_model_field_type=data_model_types.field_model,
        data_type_manager_type=data_model_types.data_type_manager,
        dump_resolve_reference_action=data_model_types.dump_resolve_reference_action,
        class_name=class_name,
        use_union_operator=True,
        use_schema_description=True,
        enum_field_as_literal=LiteralType.All,
    )
    return str(parser.parse())


def generate_from_schema(
    schema_json: str,
    class_name: str | None = None,
    input_path: str | None = None,
):
    """Generate Pydantic model code from a JSON schema."""
    from datamodel_code_generator import DataModelType, LiteralType, PythonVersion
    from datamodel_code_generator.model import get_data_model_types
    from datamodel_code_generator.parser.jsonschema import JsonSchemaParser

    data_model_types = get_data_model_types(
        DataModelType.PydanticV2BaseModel,
        target_python_version=PythonVersion.PY_312,
    )
    parser = JsonSchemaParser(
        source=schema_json,
        data_model_type=data_model_types.data_model,
        data_model_root_type=data_model_types.root_model,
        data_model_field_type=data_model_types.field_model,
        data_type_manager_type=data_model_types.data_type_manager,
        dump_resolve_reference_action=data_model_types.dump_resolve_reference_action,
        class_name=class_name or (input_path or "DefaultClass").split(".")[-1],
        use_union_operator=True,
        use_schema_description=True,
        enum_field_as_literal=LiteralType.All,
    )
    return parser.parse()

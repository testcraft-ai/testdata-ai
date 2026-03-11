"""testdata-ai: AI-powered test data generator for QA engineers."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("testdata-ai")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

from testdata_ai.generator import DataGenerator, generate
from testdata_ai.async_generator import async_generate, GenerateSpec
from testdata_ai.result_types import GenerateResult, RelationshipResult
from testdata_ai.contexts import (
    ContextSchema,
    list_contexts,
    get_context_schema,
    register_context,
    load_contexts_from_file,
)

__all__ = [
    "DataGenerator",
    "generate",
    "async_generate",
    "GenerateSpec",
    "GenerateResult",
    "RelationshipResult",
    "ContextSchema",
    "list_contexts",
    "get_context_schema",
    "register_context",
    "load_contexts_from_file",
]

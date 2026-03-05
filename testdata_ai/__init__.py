"""testdata-ai: AI-powered test data generator for QA engineers."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("testdata-ai")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

from testdata_ai.generator import DataGenerator, generate, generate_from_model
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
    "generate_from_model",
    "ContextSchema",
    "list_contexts",
    "get_context_schema",
    "register_context",
    "load_contexts_from_file",
]

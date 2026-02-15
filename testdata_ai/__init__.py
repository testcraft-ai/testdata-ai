"""testdata-ai: AI-powered test data generator for QA engineers."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("testdata-ai")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

from testdata_ai.generator import TestDataGenerator, generate
from testdata_ai.contexts import list_contexts, get_context_schema

__all__ = [
    "TestDataGenerator",
    "generate",
    "list_contexts",
    "get_context_schema",
]

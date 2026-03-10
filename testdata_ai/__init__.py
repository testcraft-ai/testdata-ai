"""testdata-ai: AI-powered test data generator for QA engineers."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("testdata-ai")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

from testdata_ai.generator import DataGenerator, generate, generate_from_model, generate_with_relationships, generate_as_dataframe
from testdata_ai.async_generator import generate_parallel, async_generate, GenerateSpec

try:
    import pandas as _pandas  # noqa: F401
    del _pandas
except ImportError:
    _PANDAS_EXPORTS = []
else:
    from testdata_ai.pandas_bridge import records_to_dataframe, relationships_to_dataframes
    to_dataframe = records_to_dataframe
    _PANDAS_EXPORTS = ["to_dataframe", "records_to_dataframe", "relationships_to_dataframes"]

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
    "generate_with_relationships",
    "generate_as_dataframe",
    "generate_parallel",
    "async_generate",
    "GenerateSpec",
    "ContextSchema",
    "list_contexts",
    "get_context_schema",
    "register_context",
    "load_contexts_from_file",
] + _PANDAS_EXPORTS

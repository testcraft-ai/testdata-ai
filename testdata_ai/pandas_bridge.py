"""Pandas DataFrame conversion for testdata-ai."""
from typing import Any, Dict, List

__all__ = ["records_to_dataframe", "relationships_to_dataframes"]

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]


def records_to_dataframe(
    records: List[Dict[str, Any]],
    flatten: bool = True,
):
    """Convert a list of record dicts to a pandas DataFrame.

    Args:
        records: Output of ``generate()`` / ``generate_from_model()`` etc.
        flatten: If ``True`` (default), uses ``pd.json_normalize()`` which
            expands nested dicts into dot-separated column names
            (e.g. ``address.city``).  If ``False``, uses ``pd.DataFrame(records)``
            which keeps nested objects as object-typed cells.

    Returns:
        ``pandas.DataFrame`` with one row per record.

    Raises:
        ImportError: If ``pandas`` is not installed.
    """
    _require_pandas()
    if flatten:
        return pd.json_normalize(records)
    return pd.DataFrame(records)


def relationships_to_dataframes(
    result: Dict[str, List[Dict[str, Any]]],
    flatten: bool = True,
) -> Dict[str, Any]:  # values are pd.DataFrame
    """Convert a ``generate_with_relationships()`` result to a dict of DataFrames.

    Args:
        result: Output of ``generate_with_relationships()`` — a mapping of
            entity name to list of records.
        flatten: Forwarded to ``records_to_dataframe()`` for each entity.

    Returns:
        Dict mapping entity name to ``pandas.DataFrame``.

    Raises:
        ImportError: If ``pandas`` is not installed.
    """
    _require_pandas()
    return {
        entity: records_to_dataframe(records, flatten=flatten)
        for entity, records in result.items()
    }


def _require_pandas() -> None:
    if pd is None:
        raise ImportError(
            "The 'pandas' package is required for DataFrame conversion. "
            "Install it with: pip install 'testdata-ai[pandas]'"
        )

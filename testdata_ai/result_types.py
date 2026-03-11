"""Rich result types for testdata-ai generated data."""

__all__ = ["GenerateResult", "RelationshipResult"]

import csv
import io
import json
from typing import Any, Dict, Iterator, List, Optional


class GenerateResult:
    """Result of a single-context generation call.

    Behaves like a list (iterable, indexable, has len) while exposing
    convenience conversion methods.

    Example::

        result = generate("ecommerce_customer", count=10)
        result[0]               # first record
        for r in result: ...    # iterate
        len(result)             # number of records
        result.to_dataframe()   # pandas DataFrame
        result.to_csv()         # CSV string
    """

    def __init__(self, data: List[Dict[str, Any]]) -> None:
        self._data: List[Dict[str, Any]] = list(data)

    # ------------------------------------------------------------------
    # List-like interface
    # ------------------------------------------------------------------

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index):
        return self._data[index]

    def __repr__(self) -> str:
        return f"GenerateResult({self._data!r})"

    def __eq__(self, other) -> bool:
        if isinstance(other, GenerateResult):
            return self._data == other._data
        if isinstance(other, list):
            return self._data == other
        return NotImplemented

    # ------------------------------------------------------------------
    # Conversion methods
    # ------------------------------------------------------------------

    def to_records(self) -> List[Dict[str, Any]]:
        """Return a plain list of record dicts."""
        return list(self._data)

    def to_dataframe(self, flatten: bool = True):
        """Convert to a pandas DataFrame.

        Args:
            flatten: If True, nested dicts are expanded into dot-separated
                column names via pd.json_normalize(). If False, nested objects
                are kept as object-typed cells.

        Returns:
            pandas.DataFrame with one row per record.

        Raises:
            ImportError: If pandas is not installed.
        """
        from testdata_ai.pandas_bridge import records_to_dataframe
        return records_to_dataframe(self._data, flatten=flatten)

    def to_csv(self, path: Optional[str] = None) -> Optional[str]:
        """Convert to CSV.

        Args:
            path: If None, returns CSV as a string. If given, writes to file
                and returns None.

        Returns:
            CSV string when path is None, otherwise None.
        """
        csv_text = _records_to_csv(self._data)
        if path is None:
            return csv_text
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write(csv_text)
        return None

    def to_json(self, path: Optional[str] = None) -> Optional[str]:
        """Convert to JSON.

        Args:
            path: If None, returns JSON string. If given, writes to file.

        Returns:
            JSON string when path is None, otherwise None.
        """
        text = json.dumps(self._data, indent=2, ensure_ascii=False)
        if path is None:
            return text
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return None

    def to_yaml(self, path: Optional[str] = None) -> Optional[str]:
        """Convert to YAML.

        Args:
            path: If None, returns YAML string. If given, writes to file.

        Returns:
            YAML string when path is None, otherwise None.

        Raises:
            ImportError: If PyYAML is not installed.
        """
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required for YAML output. "
                "Install it with: pip install pyyaml"
            )
        text = yaml.dump(self._data, allow_unicode=True, sort_keys=False)
        if path is None:
            return text
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return None

    def to_batches(self, batch_size: int = 50) -> Iterator[List[Dict[str, Any]]]:
        """Yield the records in chunks of batch_size.

        This operates on already-generated data — it does not make additional
        AI calls.

        Args:
            batch_size: Number of records per batch.

        Yields:
            Lists of record dicts.
        """
        for i in range(0, len(self._data), batch_size):
            yield self._data[i : i + batch_size]


class RelationshipResult(dict):
    """Result of a relationship or parallel generation call.

    Subclass of dict mapping entity/label name to a list of record dicts.

    Example::

        result = generate({"nodes": {...}})
        result["users"]          # list of user records
        result.to_dataframes()   # Dict[str, pd.DataFrame]
        result.to_json()         # JSON string
    """

    def to_dataframes(self, flatten: bool = True) -> Dict[str, Any]:
        """Convert each entity to a pandas DataFrame.

        Args:
            flatten: Forwarded to records_to_dataframe for each entity.

        Returns:
            Dict mapping entity name to pandas.DataFrame.

        Raises:
            ImportError: If pandas is not installed.
        """
        from testdata_ai.pandas_bridge import relationships_to_dataframes
        return relationships_to_dataframes(dict(self), flatten=flatten)

    def to_json(self, path: Optional[str] = None) -> Optional[str]:
        """Convert to JSON.

        Args:
            path: If None, returns JSON string. If given, writes to file.

        Returns:
            JSON string when path is None, otherwise None.
        """
        text = json.dumps(dict(self), indent=2, ensure_ascii=False)
        if path is None:
            return text
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return None

    def to_yaml(self, path: Optional[str] = None) -> Optional[str]:
        """Convert to YAML.

        Args:
            path: If None, returns YAML string. If given, writes to file.

        Returns:
            YAML string when path is None, otherwise None.

        Raises:
            ImportError: If PyYAML is not installed.
        """
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required for YAML output. "
                "Install it with: pip install pyyaml"
            )
        text = yaml.dump(dict(self), allow_unicode=True, sort_keys=False)
        if path is None:
            return text
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _flatten_dict(
    d: Dict[str, Any], parent_key: str = "", sep: str = "."
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            result.update(_flatten_dict(v, new_key, sep))
        elif isinstance(v, list):
            result[new_key] = json.dumps(v)
        else:
            result[new_key] = v
    return result


def _records_to_csv(records: List[Dict[str, Any]]) -> str:
    if not records:
        return ""
    flat = [_flatten_dict(r) for r in records]
    fieldnames = list(dict.fromkeys(k for row in flat for k in row))
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore", restval="")
    writer.writeheader()
    writer.writerows(flat)
    return buf.getvalue()

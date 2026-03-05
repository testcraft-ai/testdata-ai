"""Faker field provider integration for testdata-ai."""
import re
from typing import Any, Dict, List, Optional

try:
    from faker import Faker
except ImportError:  # pragma: no cover — tested via builtins mock
    Faker = None  # type: ignore[assignment,misc]

_METHOD_RE = re.compile(r"^faker:([a-z_][a-z0-9_]*)$")


def apply_faker_fields(
    records: List[Dict[str, Any]],
    field_providers: Dict[str, str],
    locale: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Overwrite specified record fields with Faker-generated values.

    AI-generated records are passed in; only the fields listed in
    ``field_providers`` are replaced.  All other fields keep their AI values,
    preserving semantic coherence.

    Args:
        records: List of record dicts from AI generation.
        field_providers: Mapping of field_name → ``"faker:method_name"``.
        locale: BCP 47 locale tag (e.g. ``"pl_PL"``); passed to ``Faker()``.
            If ``None``, Faker uses its default locale (``en_US``).

    Returns:
        New list of records with specified fields replaced by Faker values.

    Raises:
        ImportError: If the ``faker`` package is not installed.
        ValueError: If a provider spec is invalid or the Faker method does not exist.
    """
    if Faker is None:
        raise ImportError(
            "The 'faker' package is required for field_providers. "
            "Install it with: pip install 'testdata-ai[faker]'"
        )

    fake = Faker(locale) if locale else Faker()

    # Resolve methods upfront — fail fast before touching any records.
    method_map: Dict[str, Any] = {}
    for field, spec in field_providers.items():
        m = _METHOD_RE.match(spec)
        if not m:
            raise ValueError(
                f"Invalid provider spec {spec!r} for field {field!r}; "
                "expected 'faker:method_name'"
            )
        method_name = m.group(1)
        method = getattr(fake, method_name, None)
        if method is None:
            raise ValueError(
                f"Faker has no method {method_name!r} (field {field!r}). "
                "See https://faker.readthedocs.io/ for available providers."
            )
        method_map[field] = method

    return [
        {**record, **{field: method() for field, method in method_map.items()}}
        for record in records
    ]

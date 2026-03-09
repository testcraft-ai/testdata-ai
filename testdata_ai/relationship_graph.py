"""
Relationship graph support for generate_with_relationships().

Provides graph parsing, validation, topological sort (Kahn's BFS),
and FK injection post-processing.
"""

__all__ = ["RelationshipNodeSpec", "parse_graph", "topological_sort", "inject_fk"]

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RelationshipNodeSpec:
    """Specification for one entity node in a relationship graph.

    Attributes:
        name: Key name in the graph dict (e.g. "orders").
        context: Registered context identifier (e.g. "restaurant_order").
        count: Number of records to generate.
        parent: Name of the parent node in the same graph (None for root nodes).
        fk_field: Field name to inject into child records (required when parent is set).
        parent_pk: Field from parent records used as the FK value (required when parent is set).
        parent_sample_size: How many parent records to embed in the child prompt for
            semantic coherence. Capped at min(parent_sample_size, 5, len(parent_records)).
    """

    name: str
    context: str
    count: int
    parent: Optional[str] = None
    fk_field: Optional[str] = None
    parent_pk: Optional[str] = None
    parent_sample_size: int = 3
    batch_size: int = 10


def parse_graph(graph_dict: Dict[str, Any]) -> Dict[str, RelationshipNodeSpec]:
    """Parse and validate a raw relationship graph dict.

    Args:
        graph_dict: Mapping of entity name → spec dict. Each spec dict must have
            ``context`` (str) and ``count`` (int). Child nodes also require
            ``parent``, ``fk_field``, and ``parent_pk``.

    Returns:
        Ordered dict of entity name → RelationshipNodeSpec (insertion order preserved).

    Raises:
        ValueError: On missing required fields, invalid types, or unknown parent references.
    """
    if not isinstance(graph_dict, dict) or not graph_dict:
        raise ValueError("graph must be a non-empty dict")

    specs: Dict[str, RelationshipNodeSpec] = {}

    for name, raw in graph_dict.items():
        if not isinstance(raw, dict):
            raise ValueError(f"Graph node '{name}' must be a dict, got {type(raw).__name__}")

        context = raw.get("context")
        if not context or not isinstance(context, str):
            raise ValueError(f"Graph node '{name}': 'context' must be a non-empty string")

        count = raw.get("count")
        if count is None or not isinstance(count, int) or count < 1:
            raise ValueError(f"Graph node '{name}': 'count' must be a positive integer")

        parent = raw.get("parent")
        fk_field = raw.get("fk_field")
        parent_pk = raw.get("parent_pk")

        if parent is not None:
            if not isinstance(parent, str) or not parent:
                raise ValueError(f"Graph node '{name}': 'parent' must be a non-empty string")
            if not fk_field:
                raise ValueError(
                    f"Graph node '{name}': 'fk_field' is required when 'parent' is set"
                )
            if not parent_pk:
                raise ValueError(
                    f"Graph node '{name}': 'parent_pk' is required when 'parent' is set"
                )

        parent_sample_size = raw.get("parent_sample_size", 3)
        if not isinstance(parent_sample_size, int) or parent_sample_size < 1:
            raise ValueError(
                f"Graph node '{name}': 'parent_sample_size' must be a positive integer"
            )

        batch_size = raw.get("batch_size", 10)
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError(
                f"Graph node '{name}': 'batch_size' must be a positive integer"
            )

        specs[name] = RelationshipNodeSpec(
            name=name,
            context=context,
            count=count,
            parent=parent,
            fk_field=fk_field,
            parent_pk=parent_pk,
            parent_sample_size=parent_sample_size,
            batch_size=batch_size,
        )

    # Validate parent references point to existing nodes
    node_names = set(specs.keys())
    for name, spec in specs.items():
        if spec.parent is not None and spec.parent not in node_names:
            raise ValueError(
                f"Graph node '{name}': parent '{spec.parent}' is not defined in the graph"
            )

    return specs


def topological_sort(specs: Dict[str, RelationshipNodeSpec]) -> List[str]:
    """Return node names in dependency order (roots first) using Kahn's algorithm.

    Args:
        specs: Validated spec dict from :func:`parse_graph`.

    Returns:
        List of node names ordered so every parent appears before its children.

    Raises:
        ValueError: If the graph contains a cycle, naming the involved nodes.
    """
    in_degree: Dict[str, int] = {name: 0 for name in specs}
    children: Dict[str, List[str]] = {name: [] for name in specs}

    for name, spec in specs.items():
        if spec.parent is not None:
            in_degree[name] += 1
            children[spec.parent].append(name)

    queue: deque[str] = deque(n for n, deg in in_degree.items() if deg == 0)
    order: List[str] = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for child in children[node]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    if len(order) != len(specs):
        cycle_nodes = sorted(n for n in specs if n not in order)
        raise ValueError(
            f"Cycle detected in relationship graph involving nodes: {', '.join(cycle_nodes)}"
        )

    return order


def inject_fk(
    child_records: List[Dict[str, Any]],
    parent_records: List[Dict[str, Any]],
    fk_field: str,
    parent_pk: str,
) -> List[Dict[str, Any]]:
    """Overwrite fk_field in each child record with a value from parent[parent_pk].

    This is a safety-net step that enforces referential integrity regardless of
    whether the AI followed the FK instruction in the prompt.

    Args:
        child_records: Generated child records to modify.
        parent_records: Parent records to draw FK values from.
        fk_field: Field name to set in each child record.
        parent_pk: Field name whose values are sampled from parent records.

    Returns:
        New list of child records with fk_field injected. Original dicts are not mutated.

    Raises:
        ValueError: If parent_records is empty or parent_pk is missing from any parent record.
    """
    if not parent_records:
        raise ValueError("Cannot inject FK: parent_records is empty")

    pk_values: List[Any] = []
    for i, rec in enumerate(parent_records):
        if parent_pk not in rec:
            raise ValueError(
                f"Parent record {i} is missing field '{parent_pk}' needed for FK injection"
            )
        pk_values.append(rec[parent_pk])

    return [{**record, fk_field: random.choice(pk_values)} for record in child_records]

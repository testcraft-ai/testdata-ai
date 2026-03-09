"""Tests for relationship graph generation (generate_with_relationships)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from testdata_ai.contexts import ContextSchema, register_context
from testdata_ai.relationship_graph import (
    RelationshipNodeSpec,
    inject_fk,
    parse_graph,
    topological_sort,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_USER_SAMPLE = {"email": "alice@example.com", "name": "Alice", "age": 30}
_ORDER_SAMPLE = {"order_id": "ORD-001", "amount": 99.99, "status": "pending"}


def _ai_resp(records):
    """Build a JSON string that mimics an AI response with a 'data' key."""
    return json.dumps({"data": records})


# ---------------------------------------------------------------------------
# TestRelationshipNodeSpec
# ---------------------------------------------------------------------------


class TestRelationshipNodeSpec:
    def test_minimal_spec(self):
        spec = RelationshipNodeSpec(name="users", context="ecommerce_customer", count=5)
        assert spec.name == "users"
        assert spec.context == "ecommerce_customer"
        assert spec.count == 5
        assert spec.parent is None
        assert spec.fk_field is None
        assert spec.parent_pk is None

    def test_full_spec(self):
        spec = RelationshipNodeSpec(
            name="orders",
            context="restaurant_order",
            count=20,
            parent="users",
            fk_field="user_id",
            parent_pk="email",
            parent_sample_size=5,
        )
        assert spec.parent == "users"
        assert spec.fk_field == "user_id"
        assert spec.parent_pk == "email"
        assert spec.parent_sample_size == 5

    def test_default_parent_sample_size(self):
        spec = RelationshipNodeSpec(name="x", context="ecommerce_customer", count=1)
        assert spec.parent_sample_size == 3


# ---------------------------------------------------------------------------
# TestParseGraph
# ---------------------------------------------------------------------------


class TestParseGraph:
    def test_parses_root_node(self):
        specs = parse_graph({"users": {"context": "ecommerce_customer", "count": 5}})
        assert "users" in specs
        assert specs["users"].context == "ecommerce_customer"
        assert specs["users"].count == 5
        assert specs["users"].parent is None

    def test_parses_child_node(self):
        graph = {
            "users": {"context": "ecommerce_customer", "count": 3},
            "orders": {
                "context": "restaurant_order",
                "count": 10,
                "parent": "users",
                "fk_field": "user_id",
                "parent_pk": "email",
            },
        }
        specs = parse_graph(graph)
        assert specs["orders"].parent == "users"
        assert specs["orders"].fk_field == "user_id"
        assert specs["orders"].parent_pk == "email"

    def test_custom_parent_sample_size(self):
        graph = {
            "users": {"context": "ecommerce_customer", "count": 5},
            "orders": {
                "context": "restaurant_order",
                "count": 10,
                "parent": "users",
                "fk_field": "user_id",
                "parent_pk": "email",
                "parent_sample_size": 5,
            },
        }
        specs = parse_graph(graph)
        assert specs["orders"].parent_sample_size == 5

    def test_raises_on_empty_graph(self):
        with pytest.raises(ValueError, match="non-empty"):
            parse_graph({})

    def test_raises_on_non_dict_graph(self):
        with pytest.raises(ValueError):
            parse_graph("not a dict")

    def test_raises_on_node_missing_context(self):
        with pytest.raises(ValueError, match="context"):
            parse_graph({"x": {"count": 1}})

    def test_raises_on_node_missing_count(self):
        with pytest.raises(ValueError, match="count"):
            parse_graph({"x": {"context": "ecommerce_customer"}})

    def test_raises_on_non_positive_count(self):
        with pytest.raises(ValueError, match="count"):
            parse_graph({"x": {"context": "ecommerce_customer", "count": 0}})

    def test_raises_on_parent_without_fk_field(self):
        graph = {
            "users": {"context": "ecommerce_customer", "count": 5},
            "orders": {
                "context": "restaurant_order",
                "count": 10,
                "parent": "users",
                "parent_pk": "email",
                # fk_field missing
            },
        }
        with pytest.raises(ValueError, match="fk_field"):
            parse_graph(graph)

    def test_raises_on_parent_without_parent_pk(self):
        graph = {
            "users": {"context": "ecommerce_customer", "count": 5},
            "orders": {
                "context": "restaurant_order",
                "count": 10,
                "parent": "users",
                "fk_field": "user_id",
                # parent_pk missing
            },
        }
        with pytest.raises(ValueError, match="parent_pk"):
            parse_graph(graph)

    def test_raises_on_unknown_parent_reference(self):
        graph = {
            "orders": {
                "context": "restaurant_order",
                "count": 10,
                "parent": "users",  # "users" not in graph
                "fk_field": "user_id",
                "parent_pk": "email",
            }
        }
        with pytest.raises(ValueError, match="users"):
            parse_graph(graph)

    def test_raises_on_node_not_a_dict(self):
        with pytest.raises(ValueError, match="must be a dict"):
            parse_graph({"users": "not a dict"})


# ---------------------------------------------------------------------------
# TestTopologicalSort
# ---------------------------------------------------------------------------


class TestTopologicalSort:
    def test_root_only(self):
        specs = parse_graph({"users": {"context": "ecommerce_customer", "count": 5}})
        assert topological_sort(specs) == ["users"]

    def test_two_level_parent_before_child(self):
        graph = {
            "users": {"context": "ecommerce_customer", "count": 5},
            "orders": {
                "context": "restaurant_order",
                "count": 10,
                "parent": "users",
                "fk_field": "user_id",
                "parent_pk": "email",
            },
        }
        specs = parse_graph(graph)
        order = topological_sort(specs)
        assert order.index("users") < order.index("orders")

    def test_three_level_chain(self):
        graph = {
            "users": {"context": "ecommerce_customer", "count": 3},
            "orders": {
                "context": "restaurant_order",
                "count": 6,
                "parent": "users",
                "fk_field": "user_id",
                "parent_pk": "email",
            },
            "items": {
                "context": "logistics_shipment",
                "count": 12,
                "parent": "orders",
                "fk_field": "order_id",
                "parent_pk": "order_id",
            },
        }
        specs = parse_graph(graph)
        order = topological_sort(specs)
        assert order.index("users") < order.index("orders")
        assert order.index("orders") < order.index("items")

    def test_two_roots_both_before_shared_child(self):
        graph = {
            "users": {"context": "ecommerce_customer", "count": 3},
            "products": {"context": "ecommerce_customer", "count": 5},
            "orders": {
                "context": "restaurant_order",
                "count": 10,
                "parent": "users",
                "fk_field": "user_id",
                "parent_pk": "email",
            },
        }
        specs = parse_graph(graph)
        order = topological_sort(specs)
        assert order.index("users") < order.index("orders")
        assert "products" in order

    def test_detects_direct_cycle(self):
        # Manually construct cyclic specs (parse_graph would catch the unknown parent,
        # so we force specs with internal references that form a cycle.)
        specs = {
            "a": RelationshipNodeSpec("a", "ecommerce_customer", 1, parent="b",
                                      fk_field="f", parent_pk="p"),
            "b": RelationshipNodeSpec("b", "ecommerce_customer", 1, parent="a",
                                      fk_field="f", parent_pk="p"),
        }
        with pytest.raises(ValueError, match="Cycle detected"):
            topological_sort(specs)

    def test_cycle_error_names_involved_nodes(self):
        specs = {
            "alpha": RelationshipNodeSpec("alpha", "ecommerce_customer", 1, parent="beta",
                                          fk_field="f", parent_pk="p"),
            "beta": RelationshipNodeSpec("beta", "ecommerce_customer", 1, parent="alpha",
                                         fk_field="f", parent_pk="p"),
        }
        with pytest.raises(ValueError) as exc_info:
            topological_sort(specs)
        msg = str(exc_info.value)
        assert "alpha" in msg
        assert "beta" in msg


# ---------------------------------------------------------------------------
# TestInjectFk
# ---------------------------------------------------------------------------


class TestInjectFk:
    def test_overwrites_fk_field(self):
        parents = [{"email": "a@x.com"}, {"email": "b@x.com"}]
        children = [{"order_id": 1}, {"order_id": 2}]
        result = inject_fk(children, parents, fk_field="user_email", parent_pk="email")
        assert all("user_email" in r for r in result)
        assert all(r["user_email"] in {"a@x.com", "b@x.com"} for r in result)

    def test_all_child_records_get_valid_fk(self):
        parents = [{"id": i} for i in range(3)]
        children = [{"c": i} for i in range(10)]
        result = inject_fk(children, parents, fk_field="parent_id", parent_pk="id")
        assert len(result) == 10
        assert all(r["parent_id"] in {0, 1, 2} for r in result)

    def test_does_not_mutate_original_children(self):
        parents = [{"email": "x@x.com"}]
        children = [{"order_id": 1}]
        inject_fk(children, parents, fk_field="user_email", parent_pk="email")
        assert "user_email" not in children[0]

    def test_single_parent_all_children_get_same_fk(self):
        parents = [{"email": "only@x.com"}]
        children = [{"i": i} for i in range(5)]
        result = inject_fk(children, parents, fk_field="user_email", parent_pk="email")
        assert all(r["user_email"] == "only@x.com" for r in result)

    def test_empty_children_returns_empty(self):
        parents = [{"email": "x@x.com"}]
        result = inject_fk([], parents, fk_field="user_email", parent_pk="email")
        assert result == []

    def test_raises_on_empty_parent_records(self):
        with pytest.raises(ValueError, match="empty"):
            inject_fk([{"c": 1}], [], fk_field="user_email", parent_pk="email")

    def test_raises_when_parent_missing_pk_field(self):
        parents = [{"name": "Alice"}]  # no "email"
        with pytest.raises(ValueError, match="email"):
            inject_fk([{"c": 1}], parents, fk_field="user_email", parent_pk="email")


# ---------------------------------------------------------------------------
# TestGenerateWithRelationships
# ---------------------------------------------------------------------------


class TestGenerateWithRelationships:
    """Tests for DataGenerator.generate_with_relationships()."""

    def _make_gen(self, responses):
        """Return a DataGenerator whose provider returns responses in sequence."""
        with patch("testdata_ai.generator.get_provider_config") as mock_cfg, \
             patch("testdata_ai.generator.get_provider") as mock_get_prov:
            mock_cfg.return_value = MagicMock(
                provider="openai",
                api_key="sk-fake",
                model="test-model",
                temperature=0.7,
                max_tokens=4096,
            )
            mock_prov = MagicMock()
            mock_prov.generate.side_effect = list(responses)
            mock_get_prov.return_value = mock_prov
            from testdata_ai.generator import DataGenerator
            gen = DataGenerator()
        return gen

    # --- root-only graph ---

    def test_root_only_graph(self):
        users = [_USER_SAMPLE.copy()]
        gen = self._make_gen([_ai_resp(users)])
        result = gen.generate_with_relationships(
            {"users": {"context": "ecommerce_customer", "count": 1}},
            validate=False,
        )
        assert list(result.keys()) == ["users"]
        assert len(result["users"]) == 1

    # --- two-level graph ---

    def test_two_level_graph_result_keys(self):
        users = [_USER_SAMPLE.copy()]
        orders = [{**_ORDER_SAMPLE, "user_id": "placeholder"}]
        gen = self._make_gen([_ai_resp(users), _ai_resp(orders)])
        result = gen.generate_with_relationships(
            {
                "users": {"context": "ecommerce_customer", "count": 1},
                "orders": {
                    "context": "restaurant_order",
                    "count": 1,
                    "parent": "users",
                    "fk_field": "user_id",
                    "parent_pk": "email",
                },
            },
            validate=False,
        )
        assert set(result.keys()) == {"users", "orders"}

    def test_fk_injected_into_child_records(self):
        users = [_USER_SAMPLE.copy()]
        orders = [_ORDER_SAMPLE.copy()]  # no user_id — AI didn't comply
        gen = self._make_gen([_ai_resp(users), _ai_resp(orders)])
        result = gen.generate_with_relationships(
            {
                "users": {"context": "ecommerce_customer", "count": 1},
                "orders": {
                    "context": "restaurant_order",
                    "count": 1,
                    "parent": "users",
                    "fk_field": "user_id",
                    "parent_pk": "email",
                },
            },
            validate=False,
        )
        assert result["orders"][0]["user_id"] == _USER_SAMPLE["email"]

    def test_fk_values_are_subset_of_parent_pks(self):
        users = [
            {"email": "a@x.com", "name": "A", "age": 20},
            {"email": "b@x.com", "name": "B", "age": 30},
        ]
        orders = [_ORDER_SAMPLE.copy() for _ in range(5)]
        gen = self._make_gen([_ai_resp(users), _ai_resp(orders)])
        result = gen.generate_with_relationships(
            {
                "users": {"context": "ecommerce_customer", "count": 2},
                "orders": {
                    "context": "restaurant_order",
                    "count": 5,
                    "parent": "users",
                    "fk_field": "user_id",
                    "parent_pk": "email",
                },
            },
            validate=False,
        )
        parent_emails = {"a@x.com", "b@x.com"}
        assert all(r["user_id"] in parent_emails for r in result["orders"])

    def test_child_prompt_embeds_parent_data_and_fk_instruction(self):
        """Child prompt must embed parent records and explicitly instruct FK assignment."""
        users = [_USER_SAMPLE.copy()]
        orders = [_ORDER_SAMPLE.copy()]
        prompts_captured = []

        def _capture(prompt):
            prompts_captured.append(prompt)
            return _ai_resp(users) if len(prompts_captured) == 1 else _ai_resp(orders)

        with patch("testdata_ai.generator.get_provider_config") as mock_cfg, \
             patch("testdata_ai.generator.get_provider") as mock_get_prov:
            mock_cfg.return_value = MagicMock(
                provider="openai", api_key="sk-fake", model="test-model",
                temperature=0.7, max_tokens=4096,
            )
            mock_prov = MagicMock()
            mock_prov.generate.side_effect = _capture
            mock_get_prov.return_value = mock_prov
            from testdata_ai.generator import DataGenerator
            gen = DataGenerator()

        gen.generate_with_relationships(
            {
                "users": {"context": "ecommerce_customer", "count": 1},
                "orders": {
                    "context": "restaurant_order",
                    "count": 1,
                    "parent": "users",
                    "fk_field": "user_id",
                    "parent_pk": "email",
                },
            },
            validate=False,
        )
        assert len(prompts_captured) == 2
        child_prompt = prompts_captured[1]
        assert _USER_SAMPLE["email"] in child_prompt   # parent record embedded
        assert "PARENT RECORDS" in child_prompt        # section header present
        assert "user_id" in child_prompt               # FK field named
        assert "email" in child_prompt                 # parent_pk named

    def test_three_level_chain(self):
        """FK injection works across three levels (grandparent → parent → child)."""
        users = [{"email": "u@x.com", "name": "U", "age": 25}]
        orders = [{"order_id": "O1", "amount": 50.0, "status": "done"}]
        items = [{"item": "book", "qty": 1}]

        gen = self._make_gen([_ai_resp(users), _ai_resp(orders), _ai_resp(items)])
        result = gen.generate_with_relationships(
            {
                "users": {"context": "ecommerce_customer", "count": 1},
                "orders": {
                    "context": "restaurant_order",
                    "count": 1,
                    "parent": "users",
                    "fk_field": "user_email",
                    "parent_pk": "email",
                },
                "items": {
                    "context": "logistics_shipment",
                    "count": 1,
                    "parent": "orders",
                    "fk_field": "order_ref",
                    "parent_pk": "order_id",
                },
            },
            validate=False,
        )
        assert result["orders"][0]["user_email"] == "u@x.com"
        assert result["items"][0]["order_ref"] == "O1"

    def test_validate_true_raises_on_schema_mismatch(self, clean_contexts):
        """With validate=True, ValidationError raised when fields missing."""
        from testdata_ai.contexts import ValidationError

        register_context(
            "test_parent",
            ContextSchema(description="parent", sample={"pid": "P1"}, prompt_hints=[]),
            overwrite=True,
        )
        register_context(
            "test_child",
            ContextSchema(description="child", sample={"cid": "C1", "required_field": "x"},
                          prompt_hints=[]),
            overwrite=True,
        )
        # AI returns child with missing required_field
        gen = self._make_gen([
            _ai_resp([{"pid": "P1"}]),
            _ai_resp([{"cid": "C1"}]),  # missing required_field
        ])
        with pytest.raises(ValidationError):
            gen.generate_with_relationships(
                {
                    "parents": {"context": "test_parent", "count": 1},
                    "children": {
                        "context": "test_child",
                        "count": 1,
                        "parent": "parents",
                        "fk_field": "parent_id",
                        "parent_pk": "pid",
                    },
                },
                validate=True,
            )

    def test_validate_false_skips_validation(self, clean_contexts):
        register_context(
            "test_child_v",
            ContextSchema(description="child", sample={"cid": "C1", "req": "x"},
                          prompt_hints=[]),
            overwrite=True,
        )
        register_context(
            "test_parent_v",
            ContextSchema(description="parent", sample={"pid": "P1"}, prompt_hints=[]),
            overwrite=True,
        )
        gen = self._make_gen([
            _ai_resp([{"pid": "P1"}]),
            _ai_resp([{"cid": "C1"}]),  # missing "req" — should not raise
        ])
        result = gen.generate_with_relationships(
            {
                "parents": {"context": "test_parent_v", "count": 1},
                "children": {
                    "context": "test_child_v",
                    "count": 1,
                    "parent": "parents",
                    "fk_field": "parent_id",
                    "parent_pk": "pid",
                },
            },
            validate=False,
        )
        assert "children" in result

    def test_raises_on_unknown_context(self):
        gen = self._make_gen([])
        with pytest.raises(ValueError):
            gen.generate_with_relationships(
                {"x": {"context": "nonexistent_ctx_xyz", "count": 1}},
                validate=False,
            )

    def test_raises_on_cycle_in_graph(self):
        """Cycles in the graph are detected and raised before any AI calls."""
        gen = self._make_gen([])
        with pytest.raises(ValueError, match="Cycle"):
            gen.generate_with_relationships(
                {
                    "a": {"context": "ecommerce_customer", "count": 1,
                          "parent": "b", "fk_field": "f", "parent_pk": "p"},
                    "b": {"context": "ecommerce_customer", "count": 1,
                          "parent": "a", "fk_field": "f", "parent_pk": "p"},
                },
                validate=False,
            )

    def test_applies_faker_bridge_to_child_context(self, clean_contexts):
        """Faker bridge is called for child contexts that have field_providers."""
        register_context(
            "faker_parent",
            ContextSchema(description="parent", sample={"pid": "P1"}, prompt_hints=[]),
            overwrite=True,
        )
        register_context(
            "faker_child",
            ContextSchema(
                description="child",
                sample={"cid": "C1"},
                prompt_hints=[],
                field_providers={"cid": "faker:uuid4"},
            ),
            overwrite=True,
        )
        gen = self._make_gen([
            _ai_resp([{"pid": "P1"}]),
            _ai_resp([{"cid": "original"}]),
        ])
        with patch("testdata_ai.faker_bridge.Faker") as mock_faker_cls:
            fake = MagicMock()
            fake.uuid4.return_value = "mocked-uuid"
            mock_faker_cls.return_value = fake
            result = gen.generate_with_relationships(
                {
                    "parents": {"context": "faker_parent", "count": 1},
                    "children": {
                        "context": "faker_child",
                        "count": 1,
                        "parent": "parents",
                        "fk_field": "parent_id",
                        "parent_pk": "pid",
                    },
                },
                validate=False,
            )
        # Faker-overridden field should use the mocked uuid4
        assert result["children"][0]["cid"] == "mocked-uuid"

    def test_module_level_convenience_function(self):
        """Module-level generate_with_relationships() delegates to DataGenerator."""
        from testdata_ai.generator import generate_with_relationships

        users = [_USER_SAMPLE.copy()]
        with patch("testdata_ai.generator.DataGenerator") as mock_cls:
            mock_gen = MagicMock()
            mock_gen.generate_with_relationships.return_value = {"users": users}
            mock_cls.return_value = mock_gen

            result = generate_with_relationships(
                {"users": {"context": "ecommerce_customer", "count": 1}},
                validate=False,
                locale="pl",
            )

        mock_cls.assert_called_once_with(locale="pl")
        mock_gen.generate_with_relationships.assert_called_once_with(
            {"users": {"context": "ecommerce_customer", "count": 1}},
            validate=False,
        )
        assert result == {"users": users}

    def test_public_api_export(self):
        """generate_with_relationships is exported from the top-level package."""
        import testdata_ai
        assert hasattr(testdata_ai, "generate_with_relationships")
        assert "generate_with_relationships" in testdata_ai.__all__

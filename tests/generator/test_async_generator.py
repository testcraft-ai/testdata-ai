"""Tests for testdata_ai.async_generator — generate_parallel, async_generate."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from testdata_ai.async_generator import (
    FIELD_FAKER_MAP,
    GenerateSpec,
    _UniqueFieldManager,
    async_generate,
    generate_parallel,
)
from testdata_ai.result_types import GenerateResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ai_resp(records):
    """Return an AI-style JSON string with a 'data' key."""
    return json.dumps({"data": records})


def _customer(**overrides):
    base = {"name": "Alice", "email": "alice@example.com", "age": 30}
    base.update(overrides)
    return base


def _banking(**overrides):
    base = {"name": "Bob", "email": "bob@example.com", "balance": 5000}
    base.update(overrides)
    return base


def _patch_generate_one(side_effect):
    """Patch _generate_one with an AsyncMock using given side_effect."""
    return patch("testdata_ai.async_generator._generate_one", new=AsyncMock(side_effect=side_effect))


# ---------------------------------------------------------------------------
# TestGenerateSpec
# ---------------------------------------------------------------------------


class TestGenerateSpec:
    def test_defaults(self):
        spec = GenerateSpec(context="ecommerce_customer", count=5)
        assert spec.context == "ecommerce_customer"
        assert spec.count == 5
        assert spec.locale is None
        assert spec.validate is False
        assert spec.label is None

    def test_full_spec(self):
        spec = GenerateSpec(
            context="banking_user",
            count=10,
            locale="pl",
            validate=True,
            label="accounts",
        )
        assert spec.label == "accounts"
        assert spec.validate is True
        assert spec.locale == "pl"
        assert spec.count == 10


# ---------------------------------------------------------------------------
# TestFieldFakerMap
# ---------------------------------------------------------------------------


class TestFieldFakerMap:
    def test_email_maps_to_email(self):
        assert FIELD_FAKER_MAP["email"] == "email"

    def test_id_fields_map_to_uuid4(self):
        for field in ("id", "user_id", "customer_id", "order_id"):
            assert FIELD_FAKER_MAP[field] == "uuid4"

    def test_phone_fields_map_to_phone_number(self):
        assert FIELD_FAKER_MAP["phone"] == "phone_number"
        assert FIELD_FAKER_MAP["phone_number"] == "phone_number"

    def test_name_maps_to_name(self):
        assert FIELD_FAKER_MAP["name"] == "name"
        assert FIELD_FAKER_MAP["full_name"] == "name"

    def test_username_maps_to_user_name(self):
        assert FIELD_FAKER_MAP["username"] == "user_name"


# ---------------------------------------------------------------------------
# TestUniqueFieldManager
# ---------------------------------------------------------------------------


class TestUniqueFieldManager:
    def _make_mgr(self, fields=None):
        """Create a _UniqueFieldManager with a mocked Faker."""
        fields = fields or ["email"]
        mgr = object.__new__(_UniqueFieldManager)
        mgr._fields = fields
        mgr._faker = MagicMock()
        mgr._seen = {f: set() for f in fields}
        return mgr

    def test_raises_import_error_when_faker_not_installed(self):
        import builtins
        real_import = builtins.__import__

        def _block_faker(name, *args, **kwargs):
            if name == "faker":
                raise ImportError("No module named 'faker'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_block_faker):
            with pytest.raises(ImportError, match="pip install"):
                _UniqueFieldManager(["email"])

    def test_no_duplicates_no_change(self):
        mgr = self._make_mgr(["email"])
        results = {
            "buyers": [{"email": "a@a.com"}, {"email": "b@b.com"}],
            "sellers": [{"email": "c@c.com"}],
        }
        deduped = mgr.deduplicate(results)
        emails = [r["email"] for group in deduped.values() for r in group]
        assert len(emails) == len(set(emails))

    def test_duplicate_email_is_replaced(self):
        mgr = self._make_mgr(["email"])
        mgr._faker.email.return_value = "fresh@new.com"
        results = {
            "buyers": [{"email": "same@example.com"}],
            "sellers": [{"email": "same@example.com"}],
        }
        deduped = mgr.deduplicate(results)
        all_emails = [r["email"] for group in deduped.values() for r in group]
        assert len(all_emails) == len(set(all_emails))

    def test_duplicate_across_three_contexts(self):
        mgr = self._make_mgr(["email"])
        counter = iter(["x@x.com", "y@y.com"])
        mgr._faker.email.side_effect = lambda: next(counter)
        results = {
            "a": [{"email": "dup@dup.com"}],
            "b": [{"email": "dup@dup.com"}],
            "c": [{"email": "dup@dup.com"}],
        }
        deduped = mgr.deduplicate(results)
        all_emails = [r["email"] for group in deduped.values() for r in group]
        assert len(all_emails) == len(set(all_emails)) == 3

    def test_non_deduplicated_fields_unchanged(self):
        mgr = self._make_mgr(["email"])
        results = {
            "a": [{"email": "x@x.com", "name": "Alice"}],
            "b": [{"email": "y@y.com", "name": "Alice"}],  # name dup is fine
        }
        deduped = mgr.deduplicate(results)
        names = [r["name"] for group in deduped.values() for r in group]
        assert names.count("Alice") == 2

    def test_original_results_not_mutated(self):
        mgr = self._make_mgr(["email"])
        mgr._faker.email.return_value = "fresh@new.com"
        original_val = "dup@dup.com"
        results = {
            "a": [{"email": original_val}],
            "b": [{"email": original_val}],
        }
        mgr.deduplicate(results)
        assert results["a"][0]["email"] == original_val
        assert results["b"][0]["email"] == original_val

    def test_faker_value_uses_mapped_method(self):
        mgr = self._make_mgr(["email"])
        mgr._faker.email.return_value = "mapped@example.com"
        val = mgr._faker_value("email")
        mgr._faker.email.assert_called_once()
        assert val == "mapped@example.com"

    def test_faker_value_unknown_field_falls_back_to_uuid4(self):
        mgr = self._make_mgr(["unknown_xyz"])
        mgr._faker.uuid4.return_value = "some-uuid"
        val = mgr._faker_value("unknown_xyz")
        mgr._faker.uuid4.assert_called_once()
        assert val == "some-uuid"

    def test_faker_value_id_field_calls_uuid4(self):
        mgr = self._make_mgr(["id"])
        mgr._faker.uuid4.return_value = "uuid-123"
        val = mgr._faker_value("id")
        mgr._faker.uuid4.assert_called_once()

    def test_missing_field_in_record_skipped(self):
        mgr = self._make_mgr(["email"])
        results = {
            "a": [{"name": "Alice"}],  # no email key
            "b": [{"email": "b@b.com"}],
        }
        deduped = mgr.deduplicate(results)
        assert deduped["a"][0] == {"name": "Alice"}

    def test_none_value_in_record_skipped(self):
        mgr = self._make_mgr(["email"])
        results = {"a": [{"email": None}, {"email": "ok@ok.com"}]}
        # Should not raise
        deduped = mgr.deduplicate(results)
        assert deduped["a"][0]["email"] is None

    def test_max_retries_exhaustion_logs_warning(self, caplog):
        import logging

        mgr = self._make_mgr(["email"])
        # Faker always returns the same duplicate value → retries exhausted.
        mgr._faker.email.return_value = "dup@dup.com"
        results = {"a": [{"email": "dup@dup.com"}, {"email": "dup@dup.com"}]}
        with caplog.at_level(logging.WARNING, logger="testdata_ai.async_generator"):
            deduped = mgr.deduplicate(results)
        assert "Could not find unique value" in caplog.text
        # Value left as-is after exhaustion.
        assert deduped["a"][1]["email"] == "dup@dup.com"


# ---------------------------------------------------------------------------
# TestGenerateParallel
# ---------------------------------------------------------------------------


class TestGenerateParallel:
    async def test_empty_specs_raises_value_error(self):
        with pytest.raises(ValueError, match="non-empty"):
            await generate_parallel([])

    async def test_basic_two_specs_both_execute(self):
        customer = _customer()
        banking = _banking()
        with _patch_generate_one([[customer], [banking]]):
            results = await generate_parallel([
                GenerateSpec("ecommerce_customer", count=1),
                GenerateSpec("banking_user", count=1),
            ])
        assert results["ecommerce_customer"] == [customer]
        assert results["banking_user"] == [banking]

    async def test_results_keyed_by_context_name_when_no_label(self):
        with _patch_generate_one([[_customer()]]):
            results = await generate_parallel([GenerateSpec("ecommerce_customer", 1)])
        assert "ecommerce_customer" in results

    async def test_label_overrides_context_as_result_key(self):
        with _patch_generate_one([[_customer()]]):
            results = await generate_parallel([
                GenerateSpec("ecommerce_customer", 1, label="buyers"),
            ])
        assert "buyers" in results
        assert "ecommerce_customer" not in results

    async def test_same_context_no_label_auto_merge(self):
        """Two specs with same context and no label → merged into one list."""
        r1, r2 = _customer(email="a@a.com"), _customer(email="b@b.com")
        with _patch_generate_one([[r1], [r2]]):
            results = await generate_parallel([
                GenerateSpec("ecommerce_customer", 1),
                GenerateSpec("ecommerce_customer", 1),
            ])
        assert len(results) == 1
        assert "ecommerce_customer" in results
        assert len(results["ecommerce_customer"]) == 2

    async def test_same_context_with_label_separate_keys(self):
        r1, r2 = _customer(email="a@a.com"), _customer(email="b@b.com")
        with _patch_generate_one([[r1], [r2]]):
            results = await generate_parallel([
                GenerateSpec("ecommerce_customer", 1, label="buyers"),
                GenerateSpec("ecommerce_customer", 1, label="sellers"),
            ])
        assert set(results.keys()) == {"buyers", "sellers"}

    async def test_global_unique_fields_none_skips_dedup(self):
        with _patch_generate_one([[_customer()]]):
            with patch("testdata_ai.async_generator._UniqueFieldManager") as mock_mgr:
                await generate_parallel(
                    [GenerateSpec("ecommerce_customer", 1)],
                    global_unique_fields=None,
                )
        mock_mgr.assert_not_called()

    async def test_global_unique_fields_triggers_dedup(self):
        customer = _customer()
        with _patch_generate_one([[customer]]):
            with patch("testdata_ai.async_generator._UniqueFieldManager") as mock_cls:
                mock_instance = MagicMock()
                mock_instance.deduplicate.return_value = {"ecommerce_customer": [customer]}
                mock_cls.return_value = mock_instance
                await generate_parallel(
                    [GenerateSpec("ecommerce_customer", 1)],
                    global_unique_fields=["email"],
                )
        mock_cls.assert_called_once()
        mock_instance.deduplicate.assert_called_once()

    async def test_cross_context_key_collision_logs_warning(self, caplog):
        import logging

        r1, r2 = _customer(), _banking()
        with _patch_generate_one([[r1], [r2]]):
            with caplog.at_level(logging.WARNING, logger="testdata_ai.async_generator"):
                results = await generate_parallel([
                    GenerateSpec("ecommerce_customer", 1, label="banking_user"),
                    GenerateSpec("banking_user", 1),
                ])
        assert "banking_user" in caplog.text
        assert "merge" in caplog.text.lower() or "separate" in caplog.text.lower()
        # Both results merged under the shared key.
        assert len(results["banking_user"]) == 2

    async def test_same_context_no_label_merge_does_not_warn(self, caplog):
        import logging

        r1, r2 = _customer(email="a@a.com"), _customer(email="b@b.com")
        with _patch_generate_one([[r1], [r2]]):
            with caplog.at_level(logging.WARNING, logger="testdata_ai.async_generator"):
                await generate_parallel([
                    GenerateSpec("ecommerce_customer", 1),
                    GenerateSpec("ecommerce_customer", 1),
                ])
        # Intentional same-context merge must not trigger the warning.
        assert "shared by specs with different contexts" not in caplog.text

    async def test_cross_call_dedup_replaces_duplicate_email(self):
        """Two contexts returning same email; dedup must produce unique emails."""
        r1 = _customer(email="dup@dup.com")
        r2 = _banking(email="dup@dup.com")
        with _patch_generate_one([[r1], [r2]]):
            results = await generate_parallel(
                [
                    GenerateSpec("ecommerce_customer", 1),
                    GenerateSpec("banking_user", 1),
                ],
                global_unique_fields=["email"],
            )
        all_emails = [r["email"] for records in results.values() for r in records]
        assert len(all_emails) == len(set(all_emails)), "Duplicate emails not replaced"

    async def test_error_in_task_propagates(self):
        async def _raise(*args, **kwargs):
            raise RuntimeError("AI provider failed")

        with patch("testdata_ai.async_generator._generate_one", new=AsyncMock(side_effect=_raise)):
            with pytest.raises(RuntimeError, match="AI provider failed"):
                await generate_parallel([GenerateSpec("ecommerce_customer", 1)])

    async def test_batch_ids_are_unique_per_spec(self):
        """Each call to _generate_one receives a different batch_id."""
        captured_ids = []

        async def _capture(spec, provider, batch_id, **kw):
            captured_ids.append(batch_id)
            return [_customer()]

        with patch("testdata_ai.async_generator._generate_one", new=AsyncMock(side_effect=_capture)):
            await generate_parallel([
                GenerateSpec("ecommerce_customer", 1),
                GenerateSpec("ecommerce_customer", 1),
            ])
        assert len(captured_ids) == 2
        assert captured_ids[0] != captured_ids[1]

    async def test_provider_forwarded_to_generate_one(self):
        captured = []

        async def _capture(spec, provider, batch_id, **kw):
            captured.append(provider)
            return [_customer()]

        with patch("testdata_ai.async_generator._generate_one", new=AsyncMock(side_effect=_capture)):
            await generate_parallel(
                [GenerateSpec("ecommerce_customer", 1)],
                provider="anthropic",
            )
        assert captured[0] == "anthropic"

    async def test_locale_forwarded_per_spec(self):
        captured_specs = []

        async def _capture(spec, provider, batch_id, **kw):
            captured_specs.append(spec)
            return [_customer()]

        with patch("testdata_ai.async_generator._generate_one", new=AsyncMock(side_effect=_capture)):
            await generate_parallel([GenerateSpec("ecommerce_customer", 1, locale="pl")])
        assert captured_specs[0].locale == "pl"


# ---------------------------------------------------------------------------
# TestAsyncGenerate
# ---------------------------------------------------------------------------


class TestAsyncGenerate:
    async def test_returns_generate_result(self):
        with _patch_generate_one([[_customer()], [_customer()], [_customer()]]):
            records = await async_generate("ecommerce_customer", count=3, parallelism=3)
        assert isinstance(records, GenerateResult)

    async def test_count_split_equal(self):
        """count=3000, parallelism=3 → 3 specs of 1000."""
        captured_specs = []

        async def _capture(spec, *args, **kw):
            captured_specs.append(spec)
            return [{}] * spec.count

        with patch("testdata_ai.async_generator._generate_one", new=AsyncMock(side_effect=_capture)):
            records = await async_generate("ecommerce_customer", count=3000, parallelism=3)
        assert len(captured_specs) == 3
        assert all(s.count == 1000 for s in captured_specs)
        assert len(records) == 3000

    async def test_count_split_uneven_last_batch_smaller(self):
        """count=10, parallelism=3 → batches [4, 4, 2]."""
        captured_counts = []

        async def _capture(spec, *args, **kw):
            captured_counts.append(spec.count)
            return [{}] * spec.count

        with patch("testdata_ai.async_generator._generate_one", new=AsyncMock(side_effect=_capture)):
            records = await async_generate("ecommerce_customer", count=10, parallelism=3)
        assert sum(captured_counts) == 10
        assert sorted(captured_counts, reverse=True) == captured_counts  # desc order
        assert len(records) == 10

    async def test_explicit_batch_size_creates_more_batches(self):
        """count=9, parallelism=3, batch_size=2 → 5 batches [2,2,2,2,1]."""
        captured_counts = []

        async def _capture(spec, *args, **kw):
            captured_counts.append(spec.count)
            return [{}] * spec.count

        with patch("testdata_ai.async_generator._generate_one", new=AsyncMock(side_effect=_capture)):
            records = await async_generate(
                "ecommerce_customer", count=9, parallelism=3, batch_size=2
            )
        assert sum(captured_counts) == 9
        assert len(captured_counts) == 5  # ceil(9/2)
        assert max(captured_counts) == 2
        assert min(captured_counts) == 1

    async def test_semaphore_limits_concurrency(self):
        """With parallelism=2, at most 2 tasks run concurrently."""
        active = [0]
        max_active = [0]

        async def _slow(spec, *args, **kw):
            active[0] += 1
            max_active[0] = max(max_active[0], active[0])
            await asyncio.sleep(0.01)
            active[0] -= 1
            return [{}] * spec.count

        with patch("testdata_ai.async_generator._generate_one", new=AsyncMock(side_effect=_slow)):
            await async_generate("ecommerce_customer", count=6, parallelism=2, batch_size=1)
        assert max_active[0] <= 2

    async def test_parallelism_one_works(self):
        with _patch_generate_one([[_customer()]]):
            records = await async_generate("ecommerce_customer", count=1, parallelism=1)
        assert len(records) == 1

    async def test_global_unique_fields_forwarded(self):
        customer = _customer()
        with _patch_generate_one([[customer]]):
            with patch("testdata_ai.async_generator._UniqueFieldManager") as mock_cls:
                mock_instance = MagicMock()
                mock_instance.deduplicate.return_value = {"_": [customer]}
                mock_cls.return_value = mock_instance
                await async_generate(
                    "ecommerce_customer", count=1, global_unique_fields=["email"]
                )
        mock_cls.assert_called_once()
        mock_instance.deduplicate.assert_called_once()

    async def test_locale_forwarded_to_specs(self):
        captured_specs = []

        async def _capture(spec, *args, **kw):
            captured_specs.append(spec)
            return [_customer()]

        with patch("testdata_ai.async_generator._generate_one", new=AsyncMock(side_effect=_capture)):
            await async_generate("ecommerce_customer", count=1, locale="pl")
        assert captured_specs[0].locale == "pl"

    async def test_locale_kwarg_in_generator_kwargs_does_not_crash(self):
        """locale= passed via **generator_kwargs must not cause TypeError."""
        captured = {}

        async def _capture(spec, provider, batch_id, **kw):
            captured["spec"] = spec
            captured["kw"] = kw
            return [_customer()]

        with patch("testdata_ai.async_generator._generate_one", new=AsyncMock(side_effect=_capture)):
            # locale= is extracted by _generate_one; should not crash here.
            await async_generate("ecommerce_customer", count=1, locale="ja")
        assert captured["spec"].locale == "ja"
        assert "locale" not in captured["kw"]

    async def test_invalid_count_raises(self):
        with pytest.raises(ValueError, match="count must be"):
            await async_generate("ecommerce_customer", count=0)

    async def test_invalid_parallelism_raises(self):
        with pytest.raises(ValueError, match="parallelism must be"):
            await async_generate("ecommerce_customer", count=5, parallelism=0)


# ---------------------------------------------------------------------------
# TestGenerateOneIntegration
# ---------------------------------------------------------------------------


class TestGenerateOneIntegration:
    """Tests for _generate_one that mock at the DataGenerator/provider level."""

    def _setup_mocks(self, records, locale=None):
        """Return a context manager tuple for patching _generate_one dependencies."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = _ai_resp(records)

        mock_gen = MagicMock()
        mock_gen.provider = mock_provider

        mock_schema = MagicMock()
        mock_schema.field_providers = None
        mock_schema.unique_fields = None

        return mock_gen, mock_provider, mock_schema

    async def test_batch_id_passed_to_get_prompt(self):
        from testdata_ai.async_generator import _generate_one

        records = [_customer()]
        mock_gen, mock_provider, mock_schema = self._setup_mocks(records)

        with patch("testdata_ai.generator.DataGenerator", return_value=mock_gen), \
             patch("testdata_ai.prompts.get_prompt", return_value="prompt text") as mock_gp, \
             patch("testdata_ai.generator._strip_markdown_fences", return_value=_ai_resp(records)), \
             patch("testdata_ai.generator._parse_ai_response", return_value=records), \
             patch("testdata_ai.contexts.get_context_schema", return_value=mock_schema):
            spec = GenerateSpec("ecommerce_customer", 1)
            await _generate_one(spec, None, "abc12345")

        mock_gp.assert_called_once_with(
            "ecommerce_customer", 1, locale=None, batch_id="abc12345"
        )

    async def test_provider_generate_called(self):
        from testdata_ai.async_generator import _generate_one

        records = [_customer()]
        mock_gen, mock_provider, mock_schema = self._setup_mocks(records)

        with patch("testdata_ai.generator.DataGenerator", return_value=mock_gen), \
             patch("testdata_ai.prompts.get_prompt", return_value="prompt text"), \
             patch("testdata_ai.generator._strip_markdown_fences", return_value=_ai_resp(records)), \
             patch("testdata_ai.generator._parse_ai_response", return_value=records), \
             patch("testdata_ai.contexts.get_context_schema", return_value=mock_schema):
            spec = GenerateSpec("ecommerce_customer", 1)
            result = await _generate_one(spec, None, "bid123")

        mock_provider.generate.assert_called_once_with("prompt text")
        assert result == records

    async def test_validation_called_when_spec_validate_true(self):
        from testdata_ai.async_generator import _generate_one

        records = [_customer()]
        mock_gen, mock_provider, mock_schema = self._setup_mocks(records)

        with patch("testdata_ai.generator.DataGenerator", return_value=mock_gen), \
             patch("testdata_ai.prompts.get_prompt", return_value="p"), \
             patch("testdata_ai.generator._strip_markdown_fences", return_value=_ai_resp(records)), \
             patch("testdata_ai.generator._parse_ai_response", return_value=records), \
             patch("testdata_ai.contexts.get_context_schema", return_value=mock_schema), \
             patch("testdata_ai.contexts.validate_generated_data", return_value=[]) as mock_val:
            spec = GenerateSpec("ecommerce_customer", 1, validate=True)
            await _generate_one(spec, None, "bid")

        mock_val.assert_called_once_with("ecommerce_customer", records)

    async def test_validation_skipped_when_spec_validate_false(self):
        from testdata_ai.async_generator import _generate_one

        records = [_customer()]
        mock_gen, mock_provider, mock_schema = self._setup_mocks(records)

        with patch("testdata_ai.generator.DataGenerator", return_value=mock_gen), \
             patch("testdata_ai.prompts.get_prompt", return_value="p"), \
             patch("testdata_ai.generator._strip_markdown_fences", return_value=_ai_resp(records)), \
             patch("testdata_ai.generator._parse_ai_response", return_value=records), \
             patch("testdata_ai.contexts.get_context_schema", return_value=mock_schema), \
             patch("testdata_ai.contexts.validate_generated_data") as mock_val:
            spec = GenerateSpec("ecommerce_customer", 1, validate=False)
            await _generate_one(spec, None, "bid")

        mock_val.assert_not_called()

    async def test_faker_bridge_applied_when_field_providers_set(self):
        from testdata_ai.async_generator import _generate_one

        records = [_customer()]
        mock_gen, mock_provider, mock_schema = self._setup_mocks(records)
        mock_schema.field_providers = {"email": "faker:email"}
        mock_schema.unique_fields = None

        with patch("testdata_ai.generator.DataGenerator", return_value=mock_gen), \
             patch("testdata_ai.prompts.get_prompt", return_value="p"), \
             patch("testdata_ai.generator._strip_markdown_fences", return_value=_ai_resp(records)), \
             patch("testdata_ai.generator._parse_ai_response", return_value=records), \
             patch("testdata_ai.contexts.get_context_schema", return_value=mock_schema), \
             patch("testdata_ai.faker_bridge.apply_faker_fields", return_value=records) as mock_ff:
            spec = GenerateSpec("ecommerce_customer", 1)
            await _generate_one(spec, None, "bid")

        mock_ff.assert_called_once()

    async def test_validation_error_raised_when_records_invalid(self):
        from testdata_ai.async_generator import _generate_one
        from testdata_ai.contexts import ValidationError

        records = [_customer()]
        mock_gen, mock_provider, mock_schema = self._setup_mocks(records)
        invalid = [{"record_index": 0, "missing_fields": ["email"]}]

        with patch("testdata_ai.generator.DataGenerator", return_value=mock_gen), \
             patch("testdata_ai.prompts.get_prompt", return_value="p"), \
             patch("testdata_ai.generator._strip_markdown_fences", return_value=_ai_resp(records)), \
             patch("testdata_ai.generator._parse_ai_response", return_value=records), \
             patch("testdata_ai.contexts.get_context_schema", return_value=mock_schema), \
             patch("testdata_ai.contexts.validate_generated_data", return_value=invalid):
            spec = GenerateSpec("ecommerce_customer", 1, validate=True)
            with pytest.raises(ValidationError):
                await _generate_one(spec, None, "bid")

    async def test_faker_bridge_skipped_when_no_field_providers(self):
        from testdata_ai.async_generator import _generate_one

        records = [_customer()]
        mock_gen, mock_provider, mock_schema = self._setup_mocks(records)
        mock_schema.field_providers = None

        with patch("testdata_ai.generator.DataGenerator", return_value=mock_gen), \
             patch("testdata_ai.prompts.get_prompt", return_value="p"), \
             patch("testdata_ai.generator._strip_markdown_fences", return_value=_ai_resp(records)), \
             patch("testdata_ai.generator._parse_ai_response", return_value=records), \
             patch("testdata_ai.contexts.get_context_schema", return_value=mock_schema), \
             patch("testdata_ai.faker_bridge.apply_faker_fields") as mock_ff:
            spec = GenerateSpec("ecommerce_customer", 1)
            await _generate_one(spec, None, "bid")

        mock_ff.assert_not_called()


# ---------------------------------------------------------------------------
# TestBatchIdInjection — verify prompts.py batch_id support
# ---------------------------------------------------------------------------


class TestBatchIdInjection:
    def test_batch_id_appears_in_prompt_when_set(self):
        from testdata_ai.prompts import get_prompt

        prompt = get_prompt("ecommerce_customer", 1, batch_id="testbatch")
        assert "testbatch" in prompt

    def test_no_batch_id_prompt_unchanged(self):
        """Prompt without batch_id is identical to old behaviour (backward compat)."""
        from testdata_ai.prompts import get_prompt

        prompt_old = get_prompt("ecommerce_customer", 1)
        prompt_new = get_prompt("ecommerce_customer", 1, batch_id=None)
        assert prompt_old == prompt_new

    def test_batch_id_not_in_prompt_when_none(self):
        from testdata_ai.prompts import get_prompt

        prompt = get_prompt("ecommerce_customer", 1, batch_id=None)
        assert "batch identifier" not in prompt

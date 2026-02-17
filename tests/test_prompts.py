"""Tests for testdata_ai.prompts — prompt building from schemas."""

import json

import pytest

from testdata_ai.contexts import CONTEXTS
from testdata_ai.prompts import get_prompt


class TestGetPrompt:

    def test_includes_count(self):
        prompt = get_prompt("ecommerce_customer", 5)
        assert "exactly 5" in prompt

    def test_includes_description(self):
        prompt = get_prompt("banking_user", 3)
        assert "banking customer profiles" in prompt

    def test_includes_json_data_key_instruction(self):
        prompt = get_prompt("saas_trial", 1)
        assert '"data"' in prompt

    def test_includes_all_prompt_hints(self):
        schema = CONTEXTS["ecommerce_customer"]
        prompt = get_prompt("ecommerce_customer", 2)
        for hint in schema.prompt_hints:
            assert hint in prompt

    def test_includes_sample_json(self):
        schema = CONTEXTS["banking_user"]
        prompt = get_prompt("banking_user", 1)
        sample_json = json.dumps(schema.sample, indent=2)
        assert sample_json in prompt

    def test_raises_for_unknown_context(self):
        with pytest.raises(ValueError, match="Unknown context"):
            get_prompt("nonexistent_context", 1)

    @pytest.mark.parametrize("context_name", list(CONTEXTS.keys()))
    def test_every_context_produces_nonempty_prompt(self, context_name):
        prompt = get_prompt(context_name, 1)
        assert prompt

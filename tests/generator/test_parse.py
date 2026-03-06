"""Tests for testdata_ai.generator — _strip_markdown_fences helper."""

from testdata_ai.generator import _strip_markdown_fences


class TestStripMarkdownFences:

    def test_plain_json_unchanged(self):
        text = '{"data": [1, 2, 3]}'
        assert _strip_markdown_fences(text) == text

    def test_strips_json_fence(self):
        text = '```json\n{"a": 1}\n```'
        assert _strip_markdown_fences(text) == '{"a": 1}'

    def test_strips_uppercase_json_fence(self):
        text = '```JSON\n{"a": 1}\n```'
        assert _strip_markdown_fences(text) == '{"a": 1}'

    def test_strips_bare_fence(self):
        text = '```\n[1, 2, 3]\n```'
        assert _strip_markdown_fences(text) == '[1, 2, 3]'

    def test_strips_fence_with_trailing_whitespace(self):
        text = '```json\n{"x": 1}\n```  \n'
        assert _strip_markdown_fences(text) == '{"x": 1}'

    def test_preserves_whitespace_inside_json(self):
        inner = '{\n  "a": 1,\n  "b": 2\n}'
        text = f"```json\n{inner}\n```"
        assert _strip_markdown_fences(text) == inner

    def test_strips_missing_closing_fence(self):
        text = '```json\n{"a": 1}'
        assert _strip_markdown_fences(text) == '{"a": 1}'

    def test_strips_missing_opening_fence(self):
        text = '[1, 2, 3]\n```'
        assert _strip_markdown_fences(text) == '[1, 2, 3]'

    def test_empty_string(self):
        assert _strip_markdown_fences("") == ""

    def test_whitespace_only(self):
        assert _strip_markdown_fences("   \n  ") == ""

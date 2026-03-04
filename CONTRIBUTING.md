# Contributing to testdata-ai

Thanks for your interest! Contributions of all kinds are welcome — bug reports, new built-in contexts, new AI providers, docs improvements, and code fixes.

## Quick start

```bash
git clone https://github.com/<you>/testdata-ai
cd testdata-ai
pip install -e ".[all,dev]"
pytest -x        # all tests should pass before you start
```

> **Python 3.9–3.12** is supported. Pick whichever you have locally.

## Ways to contribute

| Type | Where to start |
|---|---|
| Bug report | [Open a bug report](../../issues/new?template=bug_report.md) |
| Feature idea | [Open a feature request](../../issues/new?template=feature_request.md) |
| New built-in context | See [Add a context](#add-a-built-in-context) below |
| New AI provider | See [Add a provider](#add-an-ai-provider) below |
| Docs / README fix | Edit and open a PR — no issue needed for small fixes |
| Other code change | Open an issue first to discuss the approach |

## Development workflow

1. Fork the repo and create a branch: `git checkout -b feat/my-thing`
2. Make your changes (see conventions below)
3. Run the test suite: `pytest -x`
4. Push and open a Pull Request — fill in the PR template

CI runs on Python 3.9–3.12. All checks must pass.

## Code conventions

- **Type hints** on all public methods; `__all__` in every module
- **`logging.getLogger(__name__)`** — never `print()`
- **Specific `except`** clauses — never bare `except:`
- **Dataclasses** for config / schema objects
- Errors: `ValueError` for bad input, `ValidationError(ValueError)` for missing fields, `_PluginConfigError(RuntimeError)` for plugin init failures

## Tests

- One test file per module in `tests/`
- **AI calls must always be mocked** — zero real API calls in tests
- Shared fixtures live in `tests/conftest.py`
- Coverage must not decrease; run `pytest --cov=testdata_ai --cov-report=term-missing`

## Add a built-in context

Edit `testdata_ai/contexts.py` and add an entry to the `CONTEXTS` dict:

```python
"my_context": ContextSchema(
    description="Short description shown in list-contexts.",
    category="my_category",
    sample={
        "field_one": "example value",
        "field_two": 42,
    },
    prompt_hints=[
        "Generate realistic values for field_one.",
        "field_two should be between 1 and 100.",
    ],
),
```

- `sample` keys define the required fields (single source of truth for validation)
- The pytest plugin auto-registers `my_context` (single dict) and `my_contexts` (list[10]) as session-scoped fixtures

Add tests in `tests/test_contexts.py`. The PR checklist will guide you through the rest.

## Add an AI provider

Four steps:

1. `config.py` → add `"myprovider": "default-model"` to `DEFAULT_MODELS`
2. `ai_providers.py` → implement `class MyProvider(AIProvider)` with `generate(prompt) -> str`
3. `ai_providers.py` → register in `get_provider()` factory
4. `pyproject.toml` → add optional dependency group and include in `[all]`

## Commit style

Use short imperative subject lines:

```
feat: add crypto_wallet context
fix: handle empty batch from Ollama
docs: clarify custom context YAML format
test: cover ValidationError on missing field
```

## Questions?

Open a [discussion](../../discussions) or an issue — happy to help.

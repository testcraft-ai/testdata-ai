# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.12.0] - 2026-03-11

### Added
- **Rich result types** — `GenerateResult` and `RelationshipResult` wrap list output with
  a list-like interface and conversion helpers: `.to_dataframe()`, `.to_csv()`, `.to_json()`,
  `.to_jsonl()`, `.to_dict()`; added `testdata_ai/result_types.py`
- **Retry logic** — `generator.py` retries up to 3 attempts to fill shortfall when the AI
  returns fewer records than requested; Faker is applied once on the full accumulated batch
- `GenerateResult` and `RelationshipResult` exported from top-level `testdata_ai` package
- New tests: `tests/generator/test_result_types.py`, `tests/generator/test_generate_dispatch.py`
- Updated examples and README to reflect new return types
- pytest: suppress `prompt_hints is empty` `UserWarning` in test runs

## [0.11.0] - 2026-03-10

### Added
- **pandas DataFrame output** — `generate_as_dataframe(context, count)` generates records and returns
  a `pandas.DataFrame` directly; `to_dataframe(records)` / `records_to_dataframe(records)` convert
  any `List[Dict]` output; `relationships_to_dataframes(result)` converts `generate_with_relationships()`
  output to a `Dict[str, DataFrame]`
- `testdata_ai/pandas_bridge.py`: `records_to_dataframe()`, `relationships_to_dataframes()`,
  `_require_pandas()` with a clear install-hint `ImportError`
- `DataGenerator.generate_as_dataframe()` method and module-level `generate_as_dataframe()` function
- `flatten=True` (default) uses `pd.json_normalize()` to expand nested dicts into dot-separated
  column names (e.g. `location.city`); `flatten=False` uses `pd.DataFrame()` to keep nested objects
  as object-typed cells
- `to_dataframe` alias exported from top-level `testdata_ai` package (available when pandas is installed)
- New optional dependency extra `[pandas]` (`pandas>=1.3.0`); pandas included in `[all]` and `[dev]`
- `tests/generator/test_pandas_bridge.py` — 23 unit tests (pandas calls mocked via `patch`)
- `examples/pandas_output.py` — 5 usage patterns (flat records, one-liner, DataGenerator method,
  nested fields with flatten control, multi-entity join)

## [0.10.0] - 2026-03-10

### Added
- **Gemini provider** — `GeminiProvider` uses `google-genai>=0.7.0`; install with `testdata-ai[gemini]`;
  configured via `GEMINI_API_KEY` / `GEMINI_MODEL` (default `gemini-2.0-flash`);
  JSON mode via `response_mime_type="application/json"`
- **Mistral provider** — `MistralProvider` uses `mistralai>=1.0.0`; install with `testdata-ai[mistral]`;
  configured via `MISTRAL_API_KEY` / `MISTRAL_MODEL` (default `mistral-small-latest`);
  JSON mode via `response_format={"type": "json_object"}`
- **Cohere provider** — `CohereProvider` uses `cohere>=5.0.0` (`ClientV2`); install with `testdata-ai[cohere]`;
  configured via `COHERE_API_KEY` / `COHERE_MODEL` (default `command-r`)
- New optional dependency extras: `[gemini]`, `[mistral]`, `[cohere]`; all included in `[all]`
- Default model entries for all three providers in `DEFAULT_MODELS`
- `tests/providers/test_gemini.py`, `tests/providers/test_mistral.py`, `tests/providers/test_cohere.py`
  — full unit test coverage (all AI calls mocked)

## [0.9.0] - 2026-03-10

### Added
- **Async / parallel generation** — `generate_parallel(specs)` and `async_generate(context, count)`
  run multiple AI provider calls concurrently via `asyncio` + `asyncio.to_thread`; blocking
  synchronous providers work unchanged; all three providers (OpenAI, Anthropic, Ollama) supported
- `testdata_ai/async_generator.py`: `GenerateSpec` dataclass, `generate_parallel()`,
  `async_generate()`, `_UniqueFieldManager` for cross-call deduplication
- **Two-layer uniqueness** for parallel generation:
  1. *Prompt injection* (statistical) — each task receives a unique 8-char `batch_id` injected into
     its prompt to reduce duplicate values across concurrent AI calls
  2. *Faker dedup* (guaranteed) — `global_unique_fields` parameter triggers a post-generation pass
     that replaces confirmed cross-call duplicates using Faker; requires `testdata-ai[faker]`
- `async_generate()` splits a single large count into parallel batches; `parallelism` caps concurrent
  AI calls via an `asyncio.Semaphore`; `batch_size` controls records per AI call
- `generate_parallel()` merges results from specs with the same context name (no `label`) and keeps
  them separate when `label` is set
- `GenerateSpec`, `generate_parallel`, `async_generate` exported from top-level `testdata_ai` package
- `examples/async_generation.py` — 6 usage patterns (multi-context, single-context merge, batching,
  unique fields, explicit labels, locale-aware parallel)
- `tests/generator/test_async_generator.py` — 49 unit tests (all AI calls mocked)
- `prompts.py`: `get_prompt()` and `_build_prompt()` accept an optional `batch_id` parameter
  (backward-compatible default `None`)
- `pytest-asyncio>=0.21` added to dev dependencies; `asyncio_mode = "auto"` in `[tool.pytest.ini_options]`

## [0.8.0] - 2026-03-09

### Added
- **Multi-entity datasets with referential integrity** — `generate_with_relationships(graph)` generates
  multiple related entity types (customers → orders → shipments) in topological order; child prompts
  include sample parent records for semantic coherence; FK fields are injected as a safety net after
  AI generation, guaranteeing valid foreign keys in every child record
- `testdata_ai/relationship_graph.py`: `RelationshipNodeSpec`, `parse_graph()`, `topological_sort()`
  (Kahn BFS), `inject_fk()`
- CLI command `generate-related --graph-file PATH` with `-o json|jsonl-per-entity` output formats
- `generate_with_relationships` exported from top-level `testdata_ai` package
- `examples/ecommerce_graph.yaml` and `examples/relationships.py` — usage demos

### Changed
- **Test suite reorganised into subdirectories** — `tests/` now contains seven focused subdirectories
  (`cli/`, `generator/`, `contexts/`, `plugin/`, `providers/`, `cache/`, `core/`) instead of one flat
  file per module; shared fixtures moved to root `tests/conftest.py`

## [0.7.0] - 2026-03-06

### Added
- **Unique field constraints** — `ContextSchema` accepts an optional `unique_fields` list; fields in
  this list (must be a subset of `field_providers` keys) are generated via Faker's uniqueness proxy,
  guaranteeing no duplicate values within a single batch
- `apply_faker_fields()` in `faker_bridge.py` gains a `unique_fields` parameter; uniqueness is
  enforced per `generate()` call using `fake.unique.<method>()`
- `DataGenerator.generate()`, `generate_from_model()`, and the module-level convenience functions
  accept a new `unique_fields` keyword argument
- **SQL output format** — CLI `-o sql` emits a `CREATE TABLE IF NOT EXISTS` DDL statement followed by
  `INSERT INTO` statements compatible with SQLite and most major databases; column types are inferred
  per field (`INTEGER`, `REAL`, `TEXT`); nested dicts are flattened with `_` separators; lists are
  serialized as JSON strings
- `--table TEXT` CLI option overrides the default table name (context name or `"records"`)
- `examples/unique_contexts.yaml` and `examples/unique_fields.py` — usage demos for unique fields

### Changed
- `ContextSchema` validates at construction time that `unique_fields ⊆ field_providers.keys()`

## [0.6.0] - 2026-03-05

### Added
- **Faker hybrid mode** — `ContextSchema` accepts an optional `field_providers` dict mapping field
  names to `"faker:method_name"` specs; specified fields are overwritten with Faker-generated values
  after AI generation, guaranteeing format correctness for critical fields (email, IBAN, phone, UUID…)
- `testdata_ai/faker_bridge.py`: `apply_faker_fields(records, field_providers, locale)` — locale-aware
  Faker integration; methods are resolved upfront (fail-fast) before any record is modified
- `DataGenerator.generate()` applies `field_providers` automatically when set on the context schema
- `DataGenerator.generate_from_model()` and module-level `generate_from_model()` accept a new
  `field_providers` keyword argument
- New optional dependency extra: `testdata-ai[faker]` (`faker>=18.0`); included in `[all]`
- `examples/faker_hybrid.py` — usage demo for the new API

## [0.5.0] - 2026-03-05

### Fixed
- CLI: context existence is now validated before provider initialisation — unknown context names
  no longer produce a misleading "API key not found" error

## [0.4.0] - 2026-03-05

### Added
- `schema_adapter.py`: `model_to_context_schema()` — converts Pydantic v1/v2 model classes or
  JSON Schema dicts to `ContextSchema` (resolves `$ref`, `anyOf`/`oneOf`, enums, format hints,
  numeric/string constraints)
- `DataGenerator.generate_from_model(model_or_schema, count, validate)` — generate test data
  directly from a Pydantic model or JSON Schema dict without writing a `ContextSchema` by hand
- Module-level `generate_from_model()` convenience function (exported from `testdata_ai`)
- CLI: `--schema-file PATH` option on `generate` — accepts JSON/YAML files with a JSON Schema
  definition; `--context` is now optional and mutually exclusive with `--schema-file`
- `examples/generate_from_model.py` — usage demo for the new API

### Changed
- `prompts.py`: extracted `_build_prompt()` internal helper; `get_prompt()` delegates to it
- `generator.py`: extracted `_parse_ai_response()` helper (shared by `generate` and `generate_from_model`)

## [0.3.0] - 2026-03-04

### Added
- Locale / language support — generate data values in any language via `--locale <tag>` (CLI),
  `DataGenerator(locale="pl")` (Python API), `generate(..., locale="ja")` (convenience functions),
  `@pytest.mark.testdata(context=..., locale="de")` (pytest plugin)
- `AI_LOCALE` environment variable sets a session-level default locale
- Locale-aware cache key in `CacheManager` — data generated with different locales is cached separately

### Changed
- README: marked PyPI publish as done, added CHANGELOG link

## [0.2.0] - 2026-03-04

### Changed
- Version bump for PyPI re-publish (0.1.0 release smoke-test)

## [0.1.0] - 2026-03-04

### Added
- 13 built-in data contexts: `ecommerce_customer`, `banking_user`, `saas_trial`, `healthcare_patient`,
  `education_student`, `b2b_lead`, `hr_employee`, `real_estate_listing`, `iot_device`,
  `social_media_profile`, `travel_booking`, `restaurant_order`, `logistics_shipment`
- Multi-provider AI backend: OpenAI, Anthropic, Ollama (via stdlib `urllib.request`, no extra deps)
- Custom context registration via `register_context()` and `load_contexts_from_file()` (YAML/JSON)
- pytest plugin auto-loaded via `pytest11` entry point — `testdata` fixture, named context fixtures,
  `--testdata-seed`, `--testdata-last-seed`, `--testdata-clear-cache` options
- File-based result cache with deterministic seed system (`CacheManager`, `FileLock`)
- Click CLI: `testdata-ai generate`, `list-contexts`, `show-context`
- `.env` / environment variable configuration per provider
- `py.typed` marker — fully typed public API

[Unreleased]: https://github.com/testcraft-ai/testdata-ai/compare/v0.10.0...HEAD
[0.10.0]: https://github.com/testcraft-ai/testdata-ai/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/testcraft-ai/testdata-ai/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/testcraft-ai/testdata-ai/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/testcraft-ai/testdata-ai/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/testcraft-ai/testdata-ai/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/testcraft-ai/testdata-ai/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/testcraft-ai/testdata-ai/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/testcraft-ai/testdata-ai/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/testcraft-ai/testdata-ai/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/testcraft-ai/testdata-ai/releases/tag/v0.1.0

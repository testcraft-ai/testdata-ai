# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/testcraft-ai/testdata-ai/compare/v0.7.0...HEAD
[0.7.0]: https://github.com/testcraft-ai/testdata-ai/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/testcraft-ai/testdata-ai/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/testcraft-ai/testdata-ai/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/testcraft-ai/testdata-ai/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/testcraft-ai/testdata-ai/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/testcraft-ai/testdata-ai/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/testcraft-ai/testdata-ai/releases/tag/v0.1.0

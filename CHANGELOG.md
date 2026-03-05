# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/testcraft-ai/testdata-ai/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/testcraft-ai/testdata-ai/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/testcraft-ai/testdata-ai/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/testcraft-ai/testdata-ai/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/testcraft-ai/testdata-ai/releases/tag/v0.1.0

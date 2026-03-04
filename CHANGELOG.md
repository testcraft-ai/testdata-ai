# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/testcraft-ai/testdata-ai/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/testcraft-ai/testdata-ai/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/testcraft-ai/testdata-ai/releases/tag/v0.1.0

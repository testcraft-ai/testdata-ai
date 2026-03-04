# testdata-ai

> Stop writing `test@test.com`. Generate realistic, context-aware test data with GPT-4, Claude, or a local Ollama model — in one command.

[![CI](https://github.com/testcraft-ai/testdata-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/testcraft-ai/testdata-ai/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/testcraft-ai/testdata-ai/branch/main/graph/badge.svg)](https://codecov.io/gh/testcraft-ai/testdata-ai)
[![PyPI](https://img.shields.io/pypi/v/testdata-ai)](https://pypi.org/project/testdata-ai/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://pypi.org/project/testdata-ai/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

<p align="center">
  <img src="demo/demo.gif" alt="testdata-ai CLI demo" width="720">
</p>

---

```bash
pip install "testdata-ai[openai]"
testdata-ai generate --context ecommerce_customer --count 10
```

```python
from testdata_ai import generate
users = generate("ecommerce_customer", count=50)  # list of 50 realistic dicts
```

**Why testdata-ai?**

- **13 built-in domains** — e-commerce, banking, healthcare, HR, IoT, travel, and more
- **3 AI providers** — OpenAI, Anthropic, or a local Ollama model (no API cost)
- **pytest plugin** — session-scoped fixtures with caching, named seeds, and xdist support, auto-loaded

| | Faker | testdata-ai |
|---|---|---|
| Realistic emails | `test123@example.com` | `aisha.patel.2024@gmail.com` |
| Cultural diversity | Limited | Names from many cultures |
| Behavioral coherence | None | Age, location, and habits match |
| Edge-case variety | Manual | AI generates it automatically |

---

## Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [CLI](#cli)
- [Python API](#python-api)
- [Custom Contexts](#custom-contexts)
- [Pytest Plugin](#pytest-plugin)
- [Available Contexts](#available-contexts)
- [Development Roadmap](#development-roadmap)

---

## Installation

```bash
pip install "testdata-ai[openai]"       # OpenAI only
pip install "testdata-ai[anthropic]"    # Anthropic only
pip install "testdata-ai[ollama]"       # Ollama only (no extra packages — uses stdlib)
pip install "testdata-ai[all]"          # All providers
```

### Development install (from source)

```bash
git clone https://github.com/testcraft-ai/testdata-ai.git
cd testdata-ai
python -m venv venv && source venv/bin/activate
pip install -e ".[all]"
```

---

## Configuration

Create a `.env` file in the project root:

```bash
# Provider selection
AI_PROVIDER=openai          # openai | anthropic | ollama

# OpenAI
OPENAI_API_KEY=sk-proj-...
OPENAI_MODEL=gpt-4o-mini    # default; gpt-4o for higher quality
OPENAI_MAX_TOKENS=4096
OPENAI_TEMPERATURE=0.7

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-haiku-4-5-20251001   # default
ANTHROPIC_MAX_TOKENS=4096
ANTHROPIC_TEMPERATURE=0.7

# Ollama (local, no API key required)
OLLAMA_BASE_URL=http://localhost:11434  # default
OLLAMA_MODEL=qwen2.5:14b               # default
OLLAMA_MAX_TOKENS=4096
OLLAMA_TEMPERATURE=0.7
```

All env vars are optional except `*_API_KEY` (Ollama requires no API key). Defaults: `gpt-4o-mini` / `claude-haiku-4-5-20251001` / `qwen2.5:14b`, temperature `0.7`, max_tokens `4096`.

---

## CLI

After installation, use the `testdata-ai` command (or `python -m testdata_ai`):

### `generate`

Generate test data records and output as JSON, JSONL, CSV, or YAML.

```bash
testdata-ai generate --context <name> [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--context TEXT` | (required) | Context name (see [Available Contexts](#available-contexts)) |
| `--count INTEGER` | `10` | Number of records to generate |
| `--batch-size INTEGER` | `10` | Records per AI call. For `count > batch-size`, records are output progressively |
| `-o, --output [json\|jsonl\|csv\|yaml]` | `json` | Output format. Write to file via shell redirection: `-o csv > data.csv` |
| `--provider TEXT` | from env | AI provider override (`openai` / `anthropic` / `ollama`) |
| `--model TEXT` | from env | Model name override |
| `--max-tokens INTEGER` | from env | Max tokens per AI call (auto-adjusted to `batch-size` by default) |
| `--temperature FLOAT` | from env | Sampling temperature `0.0–1.0` |
| `--no-validate` | off | Skip schema validation |
| `--context-file PATH` | — | YAML or JSON file with custom context definitions (repeatable) |
| `-q, --quiet` | off | Suppress status messages (data only to stdout) |

**Examples:**

```bash
# 10 e-commerce customers to stdout (JSON)
testdata-ai generate --context ecommerce_customer --count 10

# 50 SaaS trial users saved as CSV
testdata-ai generate --context saas_trial --count 50 -o csv > trials.csv

# 100 records in batches of 20 — JSONL lines appear after each batch
testdata-ai generate --context ecommerce_customer --count 100 --batch-size 20 -o jsonl

# Use Anthropic instead of the default provider
testdata-ai generate --context banking_user --count 5 --provider anthropic

# Use a local Ollama model
testdata-ai generate --context ecommerce_customer --count 10 --provider ollama

# Use a specific model with higher token budget
testdata-ai generate --context hr_employee --count 30 --model gpt-4o --max-tokens 8192

# Machine-readable output (no status messages, plain JSON)
testdata-ai generate --context iot_device --count 20 -q | jq '.[0]'

# Use as Python module (same interface)
python -m testdata_ai generate --context ecommerce_customer --count 5

# Load a custom context from a YAML file and generate data for it
testdata-ai generate --context game_character --context-file my_contexts.yaml --count 5

# Quiet: suppress all status messages including the "Loaded context(s)..." line
testdata-ai generate --context game_character --context-file my_contexts.yaml -q
```

**Batch generation / streaming:** Large counts are split into multiple AI calls of `--batch-size` records each. Progress is reported per batch in stderr. With `-o jsonl`, records are written to stdout as each batch completes — output starts immediately rather than waiting for all records. With `-o yaml`, each batch is appended as it arrives. With `-o json` or `-o csv`, all records are accumulated and written at the end.

**Token auto-adjustment:** When `--max-tokens` is not set, the CLI estimates the required token budget **per batch** and automatically increases it if needed, printing a yellow notice to stderr.

**CSV output:** Nested dicts are flattened with dot notation (e.g., `location.city`); lists are serialized as JSON strings.

**JSONL output:** One JSON object per line — records appear progressively as batches complete.

**YAML output:** Records are appended batch-by-batch as generation progresses.

---

### `list-contexts`

List all available contexts.

```bash
testdata-ai list-contexts [--category CATEGORY] [--context-file PATH]...
```

```bash
# List all contexts
testdata-ai list-contexts

# Filter by category
testdata-ai list-contexts --category finance
testdata-ai list-contexts --category healthcare

# Include custom contexts from a file
testdata-ai list-contexts --context-file my_contexts.yaml
```

---

### `show-context`

Show full details of a context: fields, sample record, and prompt hints.

```bash
testdata-ai show-context <context> [--context-file PATH]...
```

```bash
testdata-ai show-context ecommerce_customer
testdata-ai show-context logistics_shipment

# Show a custom context defined in a file
testdata-ai show-context game_character --context-file my_contexts.yaml
```

---

### `list-models` _(Ollama only)_

List models available in the running Ollama instance.

```bash
testdata-ai list-models [--provider ollama]
```

```bash
# Requires AI_PROVIDER=ollama in .env, or pass --provider explicitly
testdata-ai list-models
testdata-ai list-models --provider ollama
```

If no models are found, the command prints a hint to run `ollama pull <model>`.

---

## Python API

### `DataGenerator`

```python
from testdata_ai import DataGenerator

# Default provider from .env
gen = DataGenerator()

# Explicit provider
gen = DataGenerator(provider="anthropic")

# Local Ollama model (no API key needed)
gen = DataGenerator(provider="ollama")
gen = DataGenerator(provider="ollama", model="mistral:latest")

# Full control
gen = DataGenerator(
    provider="openai",
    model="gpt-4o",
    temperature=0.9,
    max_tokens=8192,
)

# Pass API key directly (provider required when using api_key)
gen = DataGenerator(provider="openai", api_key="sk-proj-...")

# Generate records
customers = gen.generate("ecommerce_customer", count=10)
patients  = gen.generate("healthcare_patient", count=5)

# Large counts — automatically split into batches of 20 AI calls each
many = gen.generate("banking_user", count=100, batch_size=20)

# Skip schema validation
records = gen.generate("banking_user", count=20, validate=False)
```

`DataGenerator.generate()` returns `List[Dict[str, Any]]` — a list of plain Python dicts. For `count > batch_size`, it automatically splits the work into multiple AI calls and combines the results.

**Raises:**
- `ValueError` — unknown context, invalid JSON from AI, or bad arguments
- `testdata_ai.contexts.ValidationError` — one or more records missing required fields (when `validate=True`)

---

### `generate()` convenience function

For one-off use without instantiating the class:

```python
from testdata_ai import generate

customers = generate("ecommerce_customer", count=20)

# Large counts split automatically into 20-record batches
many = generate("ecommerce_customer", count=100, batch_size=20)
```

Configuration (provider, model, etc.) is read from environment variables. For explicit control use `DataGenerator` directly.

---

### `generate_batched()` — streaming / incremental output

When you want to process or display records as they arrive rather than waiting for the full result:

```python
from testdata_ai import generate_batched

# Process records in batches of 10 as each batch completes
for batch in generate_batched("ecommerce_customer", count=50, batch_size=10):
    print(f"Got {len(batch)} records")
    save_to_db(batch)       # commit each batch immediately
    send_to_pipeline(batch) # or stream to a downstream system

# Or use DataGenerator directly for repeated use
gen = DataGenerator(provider="anthropic")
for batch in gen.generate_batched("banking_user", count=100, batch_size=20):
    process(batch)
```

`generate_batched()` / `DataGenerator.generate_batched()` yield `List[Dict[str, Any]]` — one batch per iteration.

---

### `list_contexts()` / `get_context_schema()`

```python
from testdata_ai import list_contexts, get_context_schema

# All context names
names = list_contexts()

# Filter by category
finance_contexts = list_contexts(category="finance")

# Inspect a schema
schema = get_context_schema("ecommerce_customer")
print(schema.fields)       # ['name', 'email', 'age', ...]
print(schema.description)  # 'e-commerce customer profiles'
print(schema.category)     # 'ecommerce'
print(schema.sample)       # full sample dict
print(schema.prompt_hints) # list of generation hints
```

---

### Sample output

```json
{
  "name": "Aisha Patel",
  "email": "aisha.patel.2024@gmail.com",
  "age": 28,
  "location": {
    "city": "Mumbai",
    "country": "India",
    "timezone": "Asia/Kolkata"
  },
  "shopping_behavior": {
    "frequency": "weekly",
    "avg_order_value": "$45-80",
    "preferred_categories": ["electronics", "books"],
    "device": "mobile",
    "payment_method": "upi"
  },
  "joined_date": "2023-04-15",
  "loyalty_tier": "silver"
}
```

---

## Custom Contexts

The 13 built-in contexts cover common domains, but you can define your own for any data shape your project needs.

### File-based (YAML or JSON)

Create a YAML file where each top-level key is a context name:

```yaml
# my_contexts.yaml
game_character:
  description: "RPG game character profiles"
  category: "gaming"
  sample:
    character_id: "CHAR-0042"
    name: "Theron Blackwood"
    class: "Ranger"
    level: 15
    gold: 340
  prompt_hints:
    - "Fantasy names from diverse real-world cultures"
    - "Classes: Warrior, Mage, Ranger, Rogue, Cleric, Paladin, Druid, Bard"
    - "Level range 1-20; gold 10-5000 depending on level"
```

Load it with `--context-file` on any CLI command:

```bash
testdata-ai generate --context game_character --context-file my_contexts.yaml --count 5
testdata-ai list-contexts --context-file my_contexts.yaml
testdata-ai show-context game_character --context-file my_contexts.yaml
```

The flag is **repeatable** — pass multiple files to load several context collections at once.

JSON files are also supported (same structure, `.json` extension).

### Programmatic (`register_context`)

Register contexts at runtime from Python — useful in `conftest.py` or application setup:

```python
from testdata_ai import register_context, ContextSchema

# Using ContextSchema
register_context("game_npc", ContextSchema(
    description="RPG non-player character profiles",
    category="gaming",
    sample={
        "npc_id": "NPC-0011",
        "name": "Mira Dawnwhisper",
        "role": "innkeeper",
        "disposition": "friendly",
        "gold": 80,
    },
    prompt_hints=[
        "Fantasy names from diverse real-world cultures",
        "Roles: innkeeper, blacksmith, guard, merchant, quest-giver",
        "Gold: 10-500 depending on role",
    ],
))

# Using a plain dict (no import of ContextSchema needed)
register_context("game_item", {
    "description": "RPG inventory items",
    "category": "gaming",
    "sample": {"item_id": "ITM-099", "name": "Elven Cloak", "rarity": "rare", "value_gold": 250},
    "prompt_hints": ["Rarities: common, uncommon, rare, epic, legendary"],
})
```

Both approaches register the context globally for the current process — `DataGenerator` and the pytest plugin pick it up immediately.

### Loading from Python

```python
from testdata_ai import load_contexts_from_file

names = load_contexts_from_file("my_contexts.yaml")  # returns ['game_character']
```

### Schema rules

| Field | Required | Notes |
|---|---|---|
| `description` | yes | Non-empty string |
| `sample` | yes | Non-empty dict; keys become the required field names |
| `prompt_hints` | yes | List of strings (empty list is allowed but reduces output quality) |
| `category` | no | Defaults to `"custom"` |

**Name rules:** context names must start with a letter or underscore and contain only letters, digits, and underscores (`snake_case` recommended).

**Warnings:** `register_context` and `load_contexts_from_file` emit a `UserWarning` when `prompt_hints` is empty or when the sample contains nested dicts/lists (nested types are not validated at runtime).

**Overwriting:** pass `overwrite=True` to replace an existing context (including built-ins). A warning is emitted when a built-in is shadowed.

**Atomicity:** if a file contains multiple contexts and one fails validation, none of them are registered.

---

## Pytest Plugin

The plugin ships with the package and is **auto-loaded via the `pytest11` entry point** — no import or conftest setup needed.

### Marker fixture: `testdata`

Function-scoped. Use with `@pytest.mark.testdata` to generate any context at any count. `count` defaults to `1` if omitted.

```python
import pytest

@pytest.mark.testdata(context="ecommerce_customer", count=5)
def test_checkout_flow(testdata):
    assert len(testdata) == 5
    assert all("email" in row for row in testdata)

@pytest.mark.testdata(context="banking_user", count=1)
def test_single_bank_user(testdata):
    user = testdata[0]
    assert 300 <= user["credit_score"] <= 850
```

### Auto-generated context fixtures

For every context, the plugin auto-generates two **session-scoped** fixtures:

| Fixture name | Returns | Example |
|---|---|---|
| `<context>` | Single dict (1 record) | `ecommerce_customer` |
| `<context>s` | List of 10 dicts | `ecommerce_customers` |

```python
def test_single(ecommerce_customer):
    assert "email" in ecommerce_customer

def test_list(ecommerce_customers):
    assert len(ecommerce_customers) == 10

def test_patient(healthcare_patient):
    assert "blood_type" in healthcare_patient

def test_employees(hr_employees):
    assert all("salary" in e for e in hr_employees)
```

### Caching and seeds

The plugin caches AI responses to avoid redundant API calls within and across test runs. Cache files live in `.testdata_ai_cache/`. Add `.testdata_ai_cache/` and `.testdata_ai.log` to your `.gitignore`.

**Seed = a named cache snapshot.** Use `--testdata-seed` to name and reuse a cache:

```bash
# First run: generate data and save under "smoke-seed"
pytest --testdata-seed smoke-seed

# Subsequent runs: reuse the cached data (no AI calls)
pytest --testdata-seed smoke-seed

# Reuse the most recently used named seed
pytest --testdata-last-seed
```

Without `--testdata-seed`, a temporary seed is created per run and **deleted automatically** when the session ends.

### Seed and cache management

These options perform an admin action and exit without running tests:

```bash
# List all available seeds
pytest --testdata-list-seeds

# Show what's cached in the current (or a specific) seed
pytest --testdata-show-cache
pytest --testdata-show-cache smoke-seed

# Delete a specific seed
pytest --testdata-delete-seed smoke-seed

# Delete the last used seed
pytest --testdata-delete-last

# Clear all seeds and reset the last-seeds queue
pytest --testdata-clear-cache
```

### pytest-xdist support

When running with `pytest-xdist`, each worker will make its own AI calls unless you specify a shared named seed:

```bash
# Recommended: share one cache across all workers
pytest -n 4 --testdata-seed my-seed
```

Without `--testdata-seed`, a warning is printed per worker.

### Manual fixture pattern

If you prefer explicit control in `conftest.py`:

```python
# conftest.py
import pytest
from testdata_ai import DataGenerator

@pytest.fixture(scope="session")
def test_customers():
    gen = DataGenerator()
    return gen.generate("ecommerce_customer", count=10)

# test_checkout.py
def test_checkout_flow(test_customers):
    customer = test_customers[0]
    assert customer["email"]
    assert customer["age"] >= 18
```

### Logging

The plugin writes structured logs to `.testdata_ai.log` (rotating, max 5 MB × 3 backups) and to stderr. Log entries include seed name and xdist worker ID.

---

## Available Contexts

| Context | Category | Key Fields |
|---|---|---|
| `ecommerce_customer` | `ecommerce` | name, email, age, location, shopping_behavior, joined_date, loyalty_tier |
| `banking_user` | `finance` | name, email, age, account_type, balance, monthly_income, credit_score, branch, account_opened |
| `saas_trial` | `saas` | name, email, company, role, plan, signup_date, trial_expires, usage_stats |
| `healthcare_patient` | `healthcare` | patient_id, name, date_of_birth, gender, blood_type, primary_diagnosis, medications, allergies, insurance_provider, last_visit, attending_physician |
| `education_student` | `education` | student_id, name, email, age, major, minor, year, gpa, enrollment_status, courses, advisor |
| `b2b_lead` | `b2b` | lead_id, contact_name, email, phone, company, industry, company_size, job_title, lead_source, lead_score, deal_value, stage, notes |
| `hr_employee` | `hr` | employee_id, name, email, department, job_title, hire_date, salary, employment_type, manager, location, performance_rating |
| `real_estate_listing` | `real_estate` | listing_id, address, property_type, bedrooms, bathrooms, sqft, year_built, list_price, status, days_on_market, agent, features |
| `iot_device` | `iot` | device_id, device_type, manufacturer, firmware_version, location, status, battery_level, last_reading, alert_threshold, installed_date |
| `social_media_profile` | `social_media` | username, display_name, bio, followers, following, posts, verified, joined, category, engagement_rate, top_hashtags |
| `travel_booking` | `travel` | booking_id, passenger_name, email, trip_type, origin, destination, departure_date, return_date, cabin_class, total_price, currency, travelers, status, add_ons |
| `restaurant_order` | `food` | order_id, customer_name, restaurant, cuisine, items, subtotal, delivery_fee, tip, total, payment_method, order_type, status, ordered_at |
| `logistics_shipment` | `logistics` | tracking_number, carrier, origin, destination, ship_date, estimated_delivery, actual_delivery, weight_kg, dimensions_cm, contents, status, last_checkpoint |

Run `testdata-ai list-contexts` to see all contexts, or `testdata-ai show-context <name>` for full field details and a sample record.

---

## Development Roadmap

**Done:**
- [x] OpenAI + Anthropic + Ollama provider-agnostic architecture
- [x] 13 built-in contexts across 13 categories
- [x] Schema validation with missing-field reporting
- [x] CLI (`generate`, `list-contexts`, `show-context`, `list-models`) with JSON, JSONL, CSV, and YAML output
- [x] Auto token estimation and adjustment
- [x] Spinner with elapsed time (animated on TTY, static on non-TTY)
- [x] `python -m testdata_ai` support
- [x] Pytest plugin: marker fixture, auto-context fixtures, seed/cache system
- [x] Seed cache management CLI options (list, show, delete, clear)
- [x] TEMP seed auto-cleanup after session
- [x] pytest-xdist support with shared named seeds
- [x] Rotating log file (`.testdata_ai.log`)
- [x] Batch generation / streaming — `generate_batched()`, `--batch-size`, progressive JSONL/YAML output
- [x] Custom contexts — `register_context()`, `load_contexts_from_file()`, `--context-file` CLI option
- [ ] PyPI package (`pip install testdata-ai`)

**Next:**

---

## Contributing

Contributions welcome:

- Found a bug? Open an issue
- Have an idea? Start a discussion
- Want to code? Fork and PR

---

## License

MIT License — see [LICENSE](LICENSE)

---

**Built by [TestCraft AI](https://github.com/testcraft-ai)**

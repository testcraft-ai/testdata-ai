# testdata-ai

AI-powered test data generator for QA engineers.

Generate realistic, context-aware test data using GPT-4o or Claude — because `test@test.com` and `John Doe` aren't cutting it anymore.

![PyPI](https://img.shields.io/pypi/v/testdata-ai)
![Status](https://img.shields.io/badge/status-alpha-orange)
![Python](https://img.shields.io/badge/python-3.9+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [CLI](#cli)
- [Python API](#python-api)
- [Pytest Plugin](#pytest-plugin)
- [Available Contexts](#available-contexts)
- [Why testdata-ai?](#why-testdata-ai)
- [Development Roadmap](#development-roadmap)

---

## Installation

```bash
pip install "testdata-ai[openai]"       # OpenAI only
pip install "testdata-ai[anthropic]"    # Anthropic only
pip install "testdata-ai[all]"          # Both providers
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
AI_PROVIDER=openai          # or 'anthropic'

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
```

All env vars are optional except `*_API_KEY`. Defaults: `gpt-4o-mini` / `claude-haiku-4-5-20251001`, temperature `0.7`, max_tokens `4096`.

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
| `-o, --output [json\|jsonl\|csv\|yaml]` | `json` | Output format. Write to file via shell redirection: `-o csv > data.csv` |
| `--provider TEXT` | from env | AI provider override (`openai` / `anthropic`) |
| `--model TEXT` | from env | Model name override |
| `--max-tokens INTEGER` | from env | Max tokens for AI response |
| `--temperature FLOAT` | from env | Sampling temperature `0.0–1.0` |
| `--no-validate` | off | Skip schema validation |
| `-q, --quiet` | off | Suppress status messages (data only to stdout) |

**Examples:**

```bash
# 10 e-commerce customers to stdout (JSON)
testdata-ai generate --context ecommerce_customer --count 10

# 50 SaaS trial users saved as CSV
testdata-ai generate --context saas_trial --count 50 -o csv > trials.csv

# Use Anthropic instead of the default provider
testdata-ai generate --context banking_user --count 5 --provider anthropic

# Use a specific model with higher token budget
testdata-ai generate --context hr_employee --count 30 --model gpt-4o --max-tokens 8192

# Machine-readable output (no status messages, plain JSON)
testdata-ai generate --context iot_device --count 20 -q | jq '.[0]'

# Use as Python module (same interface)
python -m testdata_ai generate --context ecommerce_customer --count 5
```

**Token auto-adjustment:** When `--max-tokens` is not set, the CLI estimates the required token budget and automatically increases it if needed, printing a yellow notice to stderr.

**CSV output:** Nested dicts are flattened with dot notation (e.g., `location.city`); lists are serialized as JSON strings.

**JSONL output:** One JSON object per line — useful for streaming pipelines and tools like `jq`.

**YAML output:** Records as a YAML list — requires `pyyaml` (included in core dependencies).

---

### `list-contexts`

List all available contexts.

```bash
testdata-ai list-contexts [--category CATEGORY]
```

```bash
# List all contexts
testdata-ai list-contexts

# Filter by category
testdata-ai list-contexts --category finance
testdata-ai list-contexts --category healthcare
```

---

### `show-context`

Show full details of a context: fields, sample record, and prompt hints.

```bash
testdata-ai show-context <context>
```

```bash
testdata-ai show-context ecommerce_customer
testdata-ai show-context logistics_shipment
```

---

## Python API

### `DataGenerator`

```python
from testdata_ai import DataGenerator

# Default provider from .env
gen = DataGenerator()

# Explicit provider
gen = DataGenerator(provider="anthropic")

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

# Skip schema validation
records = gen.generate("banking_user", count=20, validate=False)
```

`DataGenerator.generate()` returns `List[Dict[str, Any]]` — a list of plain Python dicts.

> **Note:** Generating more than 50 records at once may exceed model token limits. For large datasets consider splitting into multiple calls (e.g. `count=50` × N).

**Raises:**
- `ValueError` — unknown context, invalid JSON from AI, or bad arguments
- `testdata_ai.contexts.ValidationError` — one or more records missing required fields (when `validate=True`)

---

### `generate()` convenience function

For one-off use without instantiating the class:

```python
from testdata_ai import generate

customers = generate("ecommerce_customer", count=20)
```

Configuration (provider, model, etc.) is read from environment variables. For explicit control use `DataGenerator` directly.

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

## Why testdata-ai?

**Traditional approach (Faker):**

```python
user = {
    "name": "John Doe",             # generic
    "email": "test123@example.com", # obviously fake
    "age": 42                       # random, no context
}
```

**testdata-ai:**

```python
gen = DataGenerator()
users = gen.generate("ecommerce_customer", count=50)
# 50 unique, realistic, context-aware profiles in seconds
```

| | Faker | testdata-ai |
|---|---|---|
| Realistic emails | `test123@example.com` | `aisha.patel.2024@gmail.com` |
| Cultural diversity | Limited | Names from many cultures |
| Behavioral data | None | Shopping patterns, preferences |
| Context awareness | No | Age matches behavior, location matches payment |
| Edge cases | Manual | AI generates variety automatically |

---

## Development Roadmap

**Done:**
- [x] OpenAI + Anthropic provider-agnostic architecture
- [x] 13 built-in contexts across 13 categories
- [x] Schema validation with missing-field reporting
- [x] CLI (`generate`, `list-contexts`, `show-context`) with JSON, JSONL, CSV, and YAML output
- [x] Auto token estimation and adjustment
- [x] Spinner with elapsed time
- [x] `python -m testdata_ai` support
- [x] Pytest plugin: marker fixture, auto-context fixtures, seed/cache system
- [x] Seed cache management CLI options (list, show, delete, clear)
- [x] TEMP seed auto-cleanup after session
- [x] pytest-xdist support with shared named seeds
- [x] Rotating log file (`.testdata_ai.log`)
- [ ] PyPI package (`pip install testdata-ai`)

**Next:**
- [ ] Custom context definitions
- [ ] Streaming / partial output for large counts

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

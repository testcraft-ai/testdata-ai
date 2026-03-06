# testdata-ai

Most test data looks like this:

```
email="test@test.com"
name="John Doe"
age=30
```

This causes unrealistic tests and hides edge cases.

testdata-ai generates realistic, culturally diverse, and behaviorally coherent data using modern LLMs. 

[![CI](https://github.com/testcraft-ai/testdata-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/testcraft-ai/testdata-ai/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/testcraft-ai/testdata-ai/branch/main/graph/badge.svg)](https://codecov.io/gh/testcraft-ai/testdata-ai)
[![PyPI](https://img.shields.io/pypi/v/testdata-ai?cacheSeconds=300)](https://pypi.org/project/testdata-ai/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://pypi.org/project/testdata-ai/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

<p align="center">
  <img src="demo/demo.gif" alt="testdata-ai CLI demo" width="720">
</p>


## Quick start


```bash
pip install "testdata-ai[openai]"
testdata-ai generate --context ecommerce_customer --count 10
```

```python
from testdata_ai import generate, generate_from_model
from pydantic import BaseModel

# Built-in context
users = generate("ecommerce_customer", count=50)

# Your own Pydantic model — no ContextSchema needed
class Order(BaseModel):
    customer_name: str
    total: float
    status: str

orders = generate_from_model(Order, count=10)
```

**Why testdata-ai?**

- **13 built-in domains** — e-commerce, banking, healthcare, HR, IoT, travel, and more
- **3 AI providers** — OpenAI, Anthropic, or a local Ollama model (no API cost)
- **pytest plugin** — session-scoped fixtures with caching, named seeds, and xdist support, auto-loaded
- **Pydantic / JSON Schema support** — generate data directly from your existing models
- **Faker hybrid mode** — mark fields as `faker:email` / `faker:iban` to get format-guaranteed values while AI handles the semantic context
- **Unique field constraints** — add `unique_fields=["email", "user_id"]` to any context and those fields will never duplicate within a batch

| | Faker | testdata-ai |
|---|---|---|
| Realistic emails | `test123@example.com` | `aisha.patel.2024@gmail.com` |
| Cultural diversity | Limited | Names from many cultures |
| Behavioral coherence | None | Age, location, and habits match |
| Edge-case variety | Manual | AI generates it automatically |
| Use your own Pydantic model | Not possible | `generate_from_model(MyModel, count=10)` |
| Format-safe critical fields | ✅ Faker's domain | `field_providers={"email": "faker:email"}` |
| Unique values across records | Requires manual set tracking | `unique_fields=["email", "user_id"]` |

**Why not just use Faker?**

Faker is excellent for generating syntactically valid values
(emails, UUIDs, phone numbers), but it lacks semantic coherence.

Example Faker output:

```
name="John Smith"
email="random42@example.com"
country="Japan"
```

testdata-ai generates consistent records:

```
name="Yuki Tanaka"
email="yuki.tanaka@gmail.com"
country="Japan"
```

---

## Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [CLI](#cli)
- [Python API](#python-api)
  - [generate\_from\_model()](#generate_from_model--schema-from-pydantic--json-schema)
- [Custom Contexts](#custom-contexts)
- [Faker Hybrid Mode](#faker-hybrid-mode)
  - [Unique Field Constraints](#unique-field-constraints)
- [Pytest Plugin](#pytest-plugin)
- [Available Contexts](#available-contexts)
- [Development Roadmap](#development-roadmap)

---

## Installation

```bash
pip install "testdata-ai[openai]"       # OpenAI only
pip install "testdata-ai[anthropic]"    # Anthropic only
pip install "testdata-ai[ollama]"       # Ollama only (no extra packages — uses stdlib)
pip install "testdata-ai[faker]"        # Faker hybrid mode (format-safe fields)
pip install "testdata-ai[all]"          # All providers + Faker
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

```bash
# Locale (optional — applies to all providers)
AI_LOCALE=pl   # BCP 47 tag; overridden by --locale or locale= per call
```

All env vars are optional except `*_API_KEY` (Ollama requires no API key). Defaults: `gpt-4o-mini` / `claude-haiku-4-5-20251001` / `qwen2.5:14b`, temperature `0.7`, max_tokens `4096`.

---

## CLI

After installation, use the `testdata-ai` command (or `python -m testdata_ai`):

### `generate`

Generate test data records and output as JSON, JSONL, CSV, YAML, or SQL.

```bash
testdata-ai generate --context <name> [OPTIONS]
testdata-ai generate --schema-file <path> [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--context TEXT` | — | Context name (see [Available Contexts](#available-contexts)). Mutually exclusive with `--schema-file` |
| `--schema-file PATH` | — | JSON or YAML file containing a JSON Schema definition. Mutually exclusive with `--context` |
| `--count INTEGER` | `10` | Number of records to generate |
| `--batch-size INTEGER` | `10` | Records per AI call. For `count > batch-size`, records are output progressively |
| `-o, --output [json\|jsonl\|csv\|yaml\|sql]` | `json` | Output format. Write to file via shell redirection: `-o csv > data.csv` |
| `--table TEXT` | context name | Table name for SQL output |
| `--provider TEXT` | from env | AI provider override (`openai` / `anthropic` / `ollama`) |
| `--model TEXT` | from env | Model name override |
| `--max-tokens INTEGER` | from env | Max tokens per AI call (auto-adjusted to `batch-size` by default) |
| `--temperature FLOAT` | from env | Sampling temperature `0.0–1.0` |
| `--locale TEXT` | from env | Locale/language for generated values (e.g. `pl`, `ja`, `de`). Overrides `AI_LOCALE` env var |
| `--no-validate` | off | Skip schema validation |
| `--context-file PATH` | — | YAML or JSON file with custom context definitions (repeatable) |
| `-q, --quiet` | off | Suppress status messages (data only to stdout) |

**Examples:**

```bash
# 10 e-commerce customers to stdout (JSON)
testdata-ai generate --context ecommerce_customer --count 10

# 50 SaaS trial users saved as CSV
testdata-ai generate --context saas_trial --count 50 -o csv > trials.csv

# SQL INSERT statements for direct database seeding
testdata-ai generate --context ecommerce_customer --count 100 -o sql > seed.sql

# SQL with a custom table name
testdata-ai generate --context banking_user --count 20 -o sql --table bank_accounts > accounts.sql

# 100 records in batches of 20 — JSONL lines appear after each batch
testdata-ai generate --context ecommerce_customer --count 100 --batch-size 20 -o jsonl

# Use Anthropic instead of the default provider
testdata-ai generate --context banking_user --count 5 --provider anthropic

# Use a local Ollama model
testdata-ai generate --context ecommerce_customer --count 10 --provider ollama

# Generate data in Polish
testdata-ai generate --context ecommerce_customer --count 5 --locale pl

# Generate data in Japanese, save as CSV
testdata-ai generate --context banking_user --count 10 --locale ja -o csv > data.csv

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

# Generate from a JSON Schema file (no built-in context needed)
testdata-ai generate --schema-file product_schema.json --count 10
testdata-ai generate --schema-file order_schema.yaml --count 5 -o csv > orders.csv
testdata-ai generate --schema-file ticket_schema.json --count 20 --locale pl
```

**Batch generation / streaming:** Large counts are split into multiple AI calls of `--batch-size` records each. Progress is reported per batch in stderr. With `-o jsonl`, records are written to stdout as each batch completes — output starts immediately rather than waiting for all records. With `-o yaml`, each batch is appended as it arrives. With `-o json`, `-o csv`, or `-o sql`, all records are accumulated and written at the end.

**Token auto-adjustment:** When `--max-tokens` is not set, the CLI estimates the required token budget **per batch** and automatically increases it if needed, printing a yellow notice to stderr.

**CSV output:** Nested dicts are flattened with dot notation (e.g., `location.city`); lists are serialized as JSON strings.

**JSONL output:** One JSON object per line — records appear progressively as batches complete.

**YAML output:** Records are appended batch-by-batch as generation progresses.

**SQL output:** Emits a `CREATE TABLE IF NOT EXISTS` DDL statement followed by `INSERT INTO` statements — compatible with SQLite and most major databases. Column types are inferred per field (`INTEGER`, `REAL`, `TEXT`). Nested dicts are flattened with underscore separators (e.g., `address_city`); lists are serialized as JSON strings. The table name defaults to the context name and can be overridden with `--table`.

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

# Generate data in a specific locale
gen = DataGenerator(locale="pl")   # Polish names, addresses, etc.
gen = DataGenerator(locale="ja")   # Japanese

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

# Generate in a specific locale
polish_customers = generate("ecommerce_customer", count=20, locale="pl")

# Large counts split automatically into 20-record batches
many = generate("ecommerce_customer", count=100, batch_size=20)
```

Configuration (provider, model, etc.) is read from environment variables. For explicit control use `DataGenerator` directly.

---

### `generate_batched()` — streaming / incremental output

When you want to process or display records as they arrive rather than waiting for the full result:

```python
from testdata_ai.generator import generate_batched

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

### `generate_from_model()` — Schema from Pydantic / JSON Schema

If you already have Pydantic models, pass them directly — no need to write a `ContextSchema` by hand. The field names, types, descriptions, and constraints are extracted automatically and used to guide the AI.

```python
from pydantic import BaseModel, Field
from testdata_ai import generate_from_model

class Customer(BaseModel):
    name: str
    email: str = Field(description="Valid email address")
    age: int = Field(ge=18, le=99, description="Age in years")
    is_active: bool

data = generate_from_model(Customer, count=10)
# [{"name": "Aisha Patel", "email": "aisha@...", "age": 34, "is_active": True}, ...]
```

**Nested models** work too:

```python
class Address(BaseModel):
    street: str
    city: str
    country: str

class Order(BaseModel):
    order_id: str
    customer_name: str
    total: float
    shipping_address: Address

data = generate_from_model(Order, count=5)
```

**Raw JSON Schema dict** — no Pydantic needed:

```python
schema = {
    "title": "Product",
    "properties": {
        "sku":      {"type": "string"},
        "name":     {"type": "string", "description": "Display name"},
        "price":    {"type": "number", "minimum": 0},
        "category": {"enum": ["electronics", "clothing", "food"]},
        "in_stock": {"type": "boolean"},
    },
}
data = generate_from_model(schema, count=5)
```

**All the usual options apply:**

```python
# Locale
data = generate_from_model(Customer, count=5, locale="pl")

# Skip validation (useful for models with many optional fields)
data = generate_from_model(Customer, count=10, validate=False)

# Via DataGenerator (reuse across multiple models)
gen = DataGenerator(provider="anthropic")
customers = gen.generate_from_model(Customer, count=5)
orders    = gen.generate_from_model(Order, count=3)
```

**Inspect the derived schema without calling the AI:**

```python
from testdata_ai.schema_adapter import model_to_context_schema

cs = model_to_context_schema(Customer)
print(cs.description)    # "Auto-generated from Customer schema"
print(cs.fields)         # ['name', 'email', 'age', 'is_active']
print(cs.sample)         # {'name': 'example_name', 'email': 'user@example.com', ...}
print(cs.prompt_hints)   # ['email: Valid email address', 'age: Age in years', 'age: min=18, max=99']
```

**Supported schema features:** `$ref` / `$defs`, `anyOf` / `oneOf` (null-safe), `enum`, `const`, string `format` (`email`, `date`, `date-time`, `uri`), numeric `minimum` / `maximum`, `minLength` / `maxLength`, nested objects and arrays, Pydantic v1 (`.schema()`) and v2 (`.model_json_schema()`). No new dependencies — Pydantic is detected by duck-typing.

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
| `field_providers` | no | Dict mapping field name → `"faker:method_name"`. Requires `pip install testdata-ai[faker]` |
| `unique_fields` | no | List of field names (must be a subset of `field_providers` keys) that will be unique within a batch |

**Name rules:** context names must start with a letter or underscore and contain only letters, digits, and underscores (`snake_case` recommended).

**Warnings:** `register_context` and `load_contexts_from_file` emit a `UserWarning` when `prompt_hints` is empty or when the sample contains nested dicts/lists (nested types are not validated at runtime).

**Overwriting:** pass `overwrite=True` to replace an existing context (including built-ins). A warning is emitted when a built-in is shadowed.

**Atomicity:** if a file contains multiple contexts and one fails validation, none of them are registered.

---

## Faker Hybrid Mode

AI excels at semantic coherence — names, locations, and behaviors that feel like real people. Faker excels at format correctness — emails that pass regex checks, IBANs with valid checksums, UUIDs that are actually valid.

Faker hybrid mode combines both: AI generates the full record, then Faker overwrites specific fields with guaranteed-valid values.

```bash
pip install "testdata-ai[faker]"
```

Add `field_providers` to any `ContextSchema`:

```python
from testdata_ai import register_context, ContextSchema, DataGenerator

register_context("banking_pl", ContextSchema(
    description="Polish retail banking customer",
    sample={
        "name": "Jan Kowalski",
        "email": "jan.kowalski@bank.pl",
        "iban": "PL61109010140000071219812874",
        "phone": "+48 123 456 789",
        "balance": 4250.00,
    },
    prompt_hints=["Realistic Polish names", "Balance 500–50000 PLN"],
    field_providers={
        "email": "faker:email",
        "iban":  "faker:iban",
        "phone": "faker:phone_number",
    },
))

gen = DataGenerator(locale="pl_PL")
records = gen.generate("banking_pl", count=10)
# → AI generates name + balance (semantically coherent)
# → Faker generates email + iban + phone (format guaranteed)
```

Works with `generate_from_model` too:

```python
from testdata_ai import generate_from_model

data = generate_from_model(
    Customer,
    count=10,
    field_providers={"email": "faker:email", "phone": "faker:phone_number"},
)
```

**How it works:**
1. AI generates the complete record (all fields, semantically coherent)
2. Faker overwrites only the listed fields with format-guaranteed values
3. Schema validation runs on the final combined record

**Faker locale follows `DataGenerator.locale`** — `DataGenerator(locale="pl_PL")` gives Polish phone numbers and emails automatically.

**Common providers:**

| Spec | Example output |
|------|----------------|
| `faker:email` | `anna.kowalska@example.com` |
| `faker:phone_number` | `+48 123 456 789` |
| `faker:iban` | `PL61 1090 1014 0000 0712 1981 2874` |
| `faker:uuid4` | `550e8400-e29b-41d4-a716-446655440000` |
| `faker:url` | `https://example.com/path` |
| `faker:ipv4` | `192.168.1.42` |
| `faker:date` | `2024-03-15` |
| `faker:postcode` | `00-001` |
| `faker:company` | `Kowalski & Synowie Sp. z o.o.` |

Full list: [faker.readthedocs.io → Providers](https://faker.readthedocs.io/en/master/providers.html)

---

### Unique Field Constraints

`unique_fields` works with **any field backed by a Faker method** — emails, UUIDs, usernames, phone numbers, IBANs, IP addresses, and more. Add it to any `ContextSchema` to guarantee no duplicates within a generated batch:

```python
register_context("saas_user", ContextSchema(
    description="SaaS trial user",
    sample={
        "name": "Alice Chen",
        "email": "alice@startup.io",
        "company": "Acme Inc",
        "plan": "trial",
    },
    prompt_hints=["Diverse professional names", "Plans: trial / starter / pro / enterprise"],
    field_providers={
        "email": "faker:email",
    },
    unique_fields=["email"],   # no duplicate emails in the batch
))

gen = DataGenerator()
records = gen.generate("saas_user", count=100)
emails = [r["email"] for r in records]
assert len(emails) == len(set(emails))  # always passes
```

Multiple unique fields at once:

```python
register_context("order", ContextSchema(
    ...,
    field_providers={
        "order_id": "faker:uuid4",
        "customer_email": "faker:email",
    },
    unique_fields=["order_id", "customer_email"],
))
```

Works with `generate_from_model` too:

```python
records = generate_from_model(
    UserSchema,
    count=50,
    field_providers={"user_id": "faker:uuid4", "email": "faker:email"},
    unique_fields=["user_id", "email"],
)
```

And in YAML context files:

```yaml
employee_unique:
  description: "HR employee with unique email"
  sample:
    name: "Fatima Al-Rashid"
    email: "f.alrashid@corp.com"
    department: "Engineering"
    salary: 125000
  prompt_hints:
    - "Diverse names from different cultures"
    - "Salary 50k–250k depending on seniority"
  field_providers:
    email: "faker:email"
  unique_fields:
    - email
```

**Rules:**
- `unique_fields` must be a subset of `field_providers` keys — validated at schema construction time with a clear error
- Works with **any Faker method** that has sufficient cardinality: `faker:email`, `faker:uuid4`, `faker:user_name`, `faker:phone_number`, `faker:iban`, `faker:ipv4`, `faker:company`, etc.
- Avoid low-cardinality methods (e.g. `faker:boolean` has only 2 values) — Faker raises `UniquenessException` if it exhausts all possible distinct values
- Uniqueness is guaranteed **within a single `generate()` call** (one batch). Across multiple `generate_batched()` iterations, each batch is internally unique but values can repeat between batches

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

# Generate data in a specific locale
@pytest.mark.testdata(context="ecommerce_customer", count=3, locale="pl")
def test_polish_customers(testdata):
    assert len(testdata) == 3
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
- [x] CLI (`generate`, `list-contexts`, `show-context`, `list-models`) with JSON, JSONL, CSV, YAML, and SQL output
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
- [x] PyPI publish — `pip install testdata-ai` · `py.typed` marker for fully typed public API
- [x] Locale / language support — `--locale pl` / `DataGenerator(locale="ja")` / `AI_LOCALE` env var; pytest plugin marker support
- [x] Schema-from-model — `generate_from_model(MyPydanticModel)` / `generate_from_model(json_schema_dict)` / `--schema-file` CLI option
- [x] Faker hybrid mode — `field_providers={"email": "faker:email"}` in `ContextSchema`; optional `testdata-ai[faker]` extra; locale-aware
- [x] Unique field constraints — `unique_fields=["email", "user_id"]` in `ContextSchema`; uses Faker's uniqueness proxy; per-batch guarantee
- [x] SQL output format — `-o sql` with `CREATE TABLE IF NOT EXISTS` + `INSERT INTO`; type inference; `--table` override

**Next:**
- [ ] `/docs` folder — installation, quickstart, CLI reference, API reference, custom contexts, pytest integration
- [ ] Async API — `async def generate()` / `generate_batched()` for high-throughput pipelines
- [ ] pandas output — `DataGenerator.to_dataframe()` convenience method
- [ ] More providers — Google Gemini, Mistral, Cohere
- [ ] Relationship generation — `generate_with_relationships()` (e.g. customers + matching orders)

---

## Contributing

Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide. See [CHANGELOG.md](CHANGELOG.md) for version history.

- Found a bug? [Open a bug report](https://github.com/testcraft-ai/testdata-ai/issues/new?template=bug_report.md)
- Have an idea? [Open a feature request](https://github.com/testcraft-ai/testdata-ai/issues/new?template=feature_request.md)
- Want to code? Fork, branch, and [open a PR](.github/PULL_REQUEST_TEMPLATE.md)

---

## Related

- [qa-ai-prompts](https://github.com/testcraft-ai/qa-ai-prompts) — 100+ battle-tested AI prompts for QA engineers. Copy, paste, customize — get results in seconds.

---

## License

MIT License — see [LICENSE](LICENSE)

---

**Built by [TestCraft AI](https://github.com/testcraft-ai)**

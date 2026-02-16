# 🤖 testdata-ai

AI-powered test data generator for QA engineers.

Generate realistic, context-aware test data using GPT-4 or Claude - because `test@test.com` and `John Doe` aren't cutting it anymore.

![Status](https://img.shields.io/badge/status-alpha-orange)
![Python](https://img.shields.io/badge/python-3.9+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Quick Start

### Install

```bash
git clone https://github.com/testcraft-ai/testdata-ai.git
cd testdata-ai
python -m venv venv && source venv/bin/activate

pip install -e ".[openai]"      # OpenAI only
# pip install -e ".[anthropic]" # Anthropic only
# pip install -e ".[all]"       # Both providers

cp .env.example .env
# Edit .env and add your API key
```

### CLI

```bash
# Generate 10 e-commerce customers (JSON to stdout)
testdata-ai generate --context ecommerce_customer --count 10

# Save as JSON file
testdata-ai generate --context banking_user --count 20 -o users.json

# Export as CSV
testdata-ai generate --context saas_trial --count 50 -f csv -o trials.csv

# List available contexts
testdata-ai list-contexts

# Show context details (fields, sample, hints)
testdata-ai show-context ecommerce_customer
```

### Python API

```python
from testdata_ai import TestDataGenerator

gen = TestDataGenerator()

# Generate 5 realistic e-commerce customers
customers = gen.generate("ecommerce_customer", count=5)

# Also works for banking and SaaS contexts
bank_users = gen.generate("banking_user", count=10)
saas_trials = gen.generate("saas_trial", count=10)
```

**Sample output:**

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

Not `test@test.com` anymore.

---

## Features

- **Provider-agnostic** -- OpenAI (GPT-4o, GPT-4o-mini) or Anthropic (Claude)
- **13 built-in contexts** -- E-commerce, Banking, SaaS, Healthcare, Education, B2B, HR, Real Estate, IoT, Social Media, Travel, Restaurant, Logistics
- **CLI + Python API** -- Use from terminal or in your test code
- **Output formats** -- JSON and CSV
- **Context-aware data** -- Shopping behavior matches demographics, payment methods match location
- **Schema validation** -- Ensures generated data has all required fields
- **Smart token management** -- Auto-detects when more tokens are needed, prompts before generating

### CLI Options

```
testdata-ai generate [OPTIONS]

Options:
  --context TEXT              Context name (required)
  --count INTEGER             Number of records [default: 10]
  -o, --output PATH           Output file (default: stdout)
  -f, --format [json|csv]     Output format [default: json]
  --provider TEXT              AI provider override
  --model TEXT                 Model name override
  --max-tokens INTEGER         Max tokens for AI response
  --temperature FLOAT          Sampling temperature 0.0-1.0
  --no-validate               Skip schema validation
  -q, --quiet                  Suppress status messages (data only)
  --help                      Show help
```

Also available as a Python module:

```bash
python -m testdata_ai generate --context ecommerce_customer --count 5
```

---

## Why testdata-ai?

**Traditional approach (Faker):**
```python
user = {
    "name": "John Doe",              # Generic
    "email": "test123@example.com",   # Obviously fake
    "age": 42                         # Random, no context
}
```

**testdata-ai approach:**
```python
gen = TestDataGenerator()
users = gen.generate("ecommerce_customer", count=100)
# 100 unique, realistic, context-aware profiles in seconds
```

| | Faker | testdata-ai |
|---|---|---|
| Realistic emails | `test123@example.com` | `aisha.patel.2024@gmail.com` |
| Cultural diversity | Limited | Names from many cultures |
| Behavioral data | None | Shopping patterns, preferences |
| Context awareness | No | Age matches behavior, location matches payment |
| Edge cases | Manual | AI generates variety automatically |

---

## Available Contexts

| Context | Category | Description |
|---|---|---|
| `ecommerce_customer` | ecommerce | Customer profiles with shopping behavior, location, loyalty tier |
| `banking_user` | finance | Bank customers with account types, balances, credit scores |
| `saas_trial` | saas | Trial users with company info, roles, usage stats |
| `healthcare_patient` | healthcare | Patient records with diagnoses, medications, allergies |
| `education_student` | education | University student profiles with courses, GPA, enrollment |
| `b2b_lead` | b2b | Sales lead profiles with company info, deal stages, scores |
| `hr_employee` | hr | Employee records with departments, salary, performance |
| `real_estate_listing` | real_estate | Property listings with address, price, features |
| `iot_device` | iot | IoT device telemetry with sensor readings, status |
| `social_media_profile` | social_media | User profiles with followers, engagement, hashtags |
| `travel_booking` | travel | Booking records with routes, cabin class, pricing |
| `restaurant_order` | food | Food delivery orders with items, totals, status |
| `logistics_shipment` | logistics | Shipment tracking with routes, weight, checkpoints |

Run `testdata-ai list-contexts` to see all contexts, or `testdata-ai show-context <name>` for details.

---

## Configuration

Create a `.env` file (see [`.env.example`](.env.example)):

```bash
AI_PROVIDER=openai                    # or 'anthropic'
OPENAI_API_KEY=sk-proj-your-key-here
OPENAI_MODEL=gpt-4o-mini             # cost-effective default
OPENAI_MAX_TOKENS=4096               # increase for large counts
```

Or override per-command:

```bash
testdata-ai generate --context banking_user --count 5 --provider anthropic
testdata-ai generate --context saas_trial --count 30 --max-tokens 8192
```

---

## Use in Tests

```python
# conftest.py
import pytest
from testdata_ai import TestDataGenerator

@pytest.fixture(scope="session")
def test_customers():
    gen = TestDataGenerator()
    return gen.generate("ecommerce_customer", count=10)

# test_checkout.py
def test_checkout_flow(test_customers):
    customer = test_customers[0]
    assert customer["email"]
    assert customer["age"] >= 18
```

---

## Development Roadmap

**Done:**
- [x] OpenAI + Anthropic provider-agnostic architecture
- [x] Core generator with data-driven prompts
- [x] 13 built-in contexts across 13 categories
- [x] Schema validation (100% success rate)
- [x] CLI interface with Click (`generate`, `list-contexts`, `show-context`)
- [x] JSON + CSV output formats
- [x] Spinner with elapsed time
- [x] Smart token estimation
- [x] `python -m testdata_ai` support

**Next:**
- [ ] Pytest plugin
- [ ] PyPI package (`pip install testdata-ai`)
- [ ] Comprehensive documentation

---

## Contributing

Contributions welcome:

- Found a bug? Open an issue
- Have an idea? Start a discussion
- Want to code? Fork and PR

---

## License

MIT License -- see [LICENSE](LICENSE)

---

**Built by [TestCraft AI](https://github.com/testcraft-ai)**

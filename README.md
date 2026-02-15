# 🤖 testdata-ai

AI-powered test data generator for QA engineers.

Generate realistic test data using GPT-4/Claude API - because `test@test.com` and `John Doe` aren't cutting it anymore.

![Status](https://img.shields.io/badge/status-alpha-orange)
![Python](https://img.shields.io/badge/python-3.9+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## ✨ What Works NOW
```python
from testdata_ai import TestDataGenerator

# Initialize with OpenAI or Anthropic
gen = TestDataGenerator()

# Generate realistic customers
customers = gen.generate("ecommerce_customer", count=10)

# Output: 10 diverse, realistic profiles!
print(customers[0])
```

**Sample Output:**
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

**Not** `test@test.com` anymore! 🎉

---

## 🚀 Quick Start

### Installation
```bash
# Clone repo
git clone https://github.com/testcraft-ai/testdata-ai.git
cd testdata-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows

# Install package (with your preferred provider)
pip install -e ".[openai]"    # OpenAI only
pip install -e ".[anthropic]" # Anthropic only
pip install -e ".[all]"       # Both providers

# Setup API key
cp .env.example .env
# Edit .env and add your OpenAI or Anthropic API key
```

### Usage
```bash
# Run example
python examples/basic_usage.py

# Or use programmatically
python
>>> from testdata_ai import TestDataGenerator
>>> gen = TestDataGenerator()
>>> customers = gen.generate("ecommerce_customer", count=5)
>>> print(f"Generated {len(customers)} customers!")
```

---

## 🎯 Features

**Currently Working:**
- ✅ **Provider-agnostic** - Use OpenAI (GPT-4) or Anthropic (Claude)
- ✅ **Context-aware** - E-commerce, Banking, SaaS (more coming!)
- ✅ **Realistic data** - Diverse names, valid emails, behavioral patterns
- ✅ **Python API** - Use in your tests programmatically
- ✅ **Type hints** - Full type safety with mypy

**Coming Soon:**
- ⏳ **CLI interface** - `testdata-ai generate --context banking --count 100`
- ⏳ **10+ contexts** - Healthcare, Education, Social, etc.
- ⏳ **Multiple formats** - JSON, CSV, SQL inserts
- ⏳ **Pytest plugin** - Auto-generate test fixtures
- ⏳ **PyPI package** - `pip install testdata-ai`

---

## 💡 Why testdata-ai?

### Traditional Approach
```python
from faker import Faker
fake = Faker()

user = {
    "name": "John Doe",           # Generic!
    "email": "test123@example.com", # Obvious fake!
    "age": 42                       # Random, no context
}
```

**Problems:**
- ❌ Everyone knows it's fake
- ❌ No context awareness (e-commerce vs banking?)
- ❌ No behavioral data (shopping patterns, preferences)
- ❌ Manual effort for edge cases

### testdata-ai Approach
```python
from testdata_ai import TestDataGenerator

gen = TestDataGenerator()
users = gen.generate("ecommerce_customer", count=100)
```

**Result:** 100 unique, realistic, context-aware profiles in ~10 seconds!

**Advantages:**
- ✅ Realistic profiles (diverse cultures, realistic emails)
- ✅ Context-aware (shopping behavior matches demographics)
- ✅ Behavioral patterns (age → preferences, location → payment methods)
- ✅ Massive time savings (AI generates in seconds)

---

## 📖 Documentation

- [Installation Guide](docs/installation.md) *(coming soon)*
- [API Reference](docs/api-reference.md) *(coming soon)*
- [Context Definitions](docs/contexts.md) *(coming soon)*

For now, check:
- [`examples/basic_usage.py`](examples/basic_usage.py) - Working example
- [`.env.example`](.env.example) - Configuration template

---

## 🛠️ Configuration

**Supported AI Providers:**
- OpenAI (GPT-4)
- Anthropic (Claude)

**Setup:**
```bash
# .env file
AI_PROVIDER=openai  # or 'anthropic'
OPENAI_API_KEY=sk-your-key-here
# or
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

See [`.env.example`](.env.example) for all options.

---

## 📊 Development Roadmap

**Week 1** (Done):
- [x] GitHub setup
- [x] OpenAI/Anthropic API integration
- [x] Core generator architecture
- [x] E-commerce, Banking, SaaS contexts
- [x] Data-driven prompt builder

**Week 2-3:**
- [ ] CLI interface (Click)
- [ ] 10+ context templates
- [ ] Output formatters (CSV, SQL)
- [ ] Basic documentation

**Week 4-6:**
- [ ] Pytest plugin
- [ ] PyPI package
- [ ] Comprehensive docs
- [ ] v1.0 launch!

---

## 🤝 Contributing

This is an active development project! Contributions welcome:

- 🐛 **Found a bug?** Open an issue!
- 💡 **Have an idea?** Start a discussion!
- 🔧 **Want to code?** Fork and PR!

*(CONTRIBUTING.md coming soon)*

---

## 📝 License

MIT License - see [LICENSE](LICENSE)

---

## 🌟 Star History

⭐ **Star this repo** to follow progress and get notified when we launch v1.0!

Building in public - watch this space! 🚀

---

## 📬 Contact

- **GitHub Issues:** Bug reports & feature requests
- **Discussions:** Questions & ideas

---

**Built with 💙 by [TestCraft AI](https://github.com/testcraft-ai)**

*Building in public -- follow the progress!*
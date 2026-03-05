"""
generate_from_model examples — testdata-ai.

Covers:
  - Pydantic v2 model (simple + nested)
  - Raw JSON Schema dict
  - Inspecting the derived ContextSchema before generating
  - locale support
  - validate=False for optional-field schemas
  - DataGenerator (reuse across multiple models)
"""

import json
from typing import List, Optional

from pydantic import BaseModel, Field

from testdata_ai import DataGenerator, generate_from_model
from testdata_ai.schema_adapter import model_to_context_schema


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


# ── Pydantic models used throughout ──────────────────────────────

class Address(BaseModel):
    street: str
    city: str
    country: str


class Customer(BaseModel):
    name: str
    email: str = Field(description="Valid email address")
    age: int = Field(ge=18, le=99, description="Age in years")
    is_active: bool
    address: Address
    tags: List[str]


class SupportTicket(BaseModel):
    ticket_id: str
    priority: str = Field(description="Urgency level")
    summary: str = Field(description="Short one-line issue description")
    resolved: bool
    assignee: Optional[str] = None


# ── Helpers ───────────────────────────────────────────────────────

def pp(data: list) -> None:
    """Pretty-print a list of records."""
    for i, record in enumerate(data, 1):
        print(f"\n  [{i}] {json.dumps(record, indent=4, ensure_ascii=False)}")


# ── Examples ──────────────────────────────────────────────────────

def main():

    # 1. Simple Pydantic model
    section("1. Simple Pydantic model → generate_from_model")
    data = generate_from_model(Customer, count=2)
    print(f"  Generated {len(data)} Customer records:")
    pp(data)

    # 2. Inspect the derived ContextSchema (no AI call)
    section("2. Inspect derived ContextSchema (no AI)")
    cs = model_to_context_schema(Customer)
    print(f"  description  : {cs.description}")
    print(f"  fields       : {cs.fields}")
    print(f"  sample       : {json.dumps(cs.sample, indent=4)}")
    print(f"  prompt_hints :")
    for h in cs.prompt_hints:
        print(f"    • {h}")

    # 3. Nested Pydantic model (Address inside Customer)
    section("3. Nested Pydantic model")
    data = generate_from_model(Customer, count=2)
    print(f"  Generated {len(data)} nested Customer records:")
    pp(data)

    # 4. Optional fields — validate=False avoids errors when AI omits nullable fields
    section("4. Optional fields → validate=False")
    data = generate_from_model(SupportTicket, count=3, validate=False)
    print(f"  Generated {len(data)} SupportTicket records (assignee may be null):")
    pp(data)

    # 5. Raw JSON Schema dict (no Pydantic)
    section("5. Raw JSON Schema dict")
    product_schema = {
        "title": "Product",
        "description": "E-commerce product listing",
        "properties": {
            "sku":      {"type": "string"},
            "name":     {"type": "string", "description": "Display name"},
            "price":    {"type": "number", "minimum": 0, "description": "Price in USD"},
            "category": {"enum": ["electronics", "clothing", "food", "books"]},
            "in_stock": {"type": "boolean"},
            "tags":     {"type": "array", "items": {"type": "string"}},
        },
    }
    data = generate_from_model(product_schema, count=3)
    print(f"  Generated {len(data)} Product records:")
    pp(data)

    # 6. Locale support
    section("6. Locale: pl (Polish)")
    data = generate_from_model(Customer, count=2, locale="pl")
    print(f"  Generated {len(data)} Polish Customer records:")
    pp(data)

    # 7. DataGenerator reuse across multiple models
    section("7. DataGenerator — multiple models, single init")
    gen = DataGenerator()
    print(f"  Provider: {gen.config.provider} / {gen.config.model}")

    customers = gen.generate_from_model(Customer, count=2)
    tickets   = gen.generate_from_model(SupportTicket, count=2, validate=False)

    print(f"\n  Customers ({len(customers)}):")
    pp(customers)
    print(f"\n  Tickets ({len(tickets)}):")
    pp(tickets)


if __name__ == "__main__":
    main()

"""
Unique field constraints — testdata-ai.

For large datasets, AI can produce duplicate emails/IDs.
``unique_fields`` guarantees no duplicates within a generated batch
by delegating those fields to Faker's uniqueness proxy (``fake.unique``).

Requirements:
    pip install testdata-ai[faker]

Run:
    python examples/unique_fields.py
"""

import json
from pathlib import Path

from testdata_ai import (
    DataGenerator,
    ContextSchema,
    generate_from_model,
    load_contexts_from_file,
    register_context,
)


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def check_uniqueness(records: list, field: str) -> None:
    values = [r[field] for r in records]
    dupes = len(values) - len(set(values))
    status = "OK — no duplicates" if dupes == 0 else f"FAIL — {dupes} duplicate(s)"
    print(f"  {field:20s} unique check: {status}")


# ── 1. Basic: unique email in a SaaS user context ─────────────────────────────

def example_saas_users():
    section("1. SaaS users — unique email across 20 records")

    register_context(
        "saas_user_unique",
        ContextSchema(
            description="SaaS trial user with guaranteed unique email",
            category="saas",
            sample={
                "name": "Alice Chen",
                "email": "alice@startup.io",
                "company": "Acme Inc",
                "plan": "trial",
                "signup_date": "2026-01-15",
            },
            prompt_hints=[
                "Diverse professional names",
                "Realistic company names",
                "Plans: trial / starter / pro / enterprise",
                "Signup dates within the last 90 days",
            ],
            field_providers={
                "email": "faker:email",
            },
            unique_fields=["email"],   # ← no duplicate emails in the batch
        ),
    )

    gen = DataGenerator()
    records = gen.generate("saas_user_unique", count=20)

    print(f"\n  Generated {len(records)} records.")
    check_uniqueness(records, "email")
    print("\n  First 3 records:")
    for r in records[:3]:
        print(f"    {r['name']:25s}  {r['email']}")


# ── 2. Multiple unique fields ──────────────────────────────────────────────────

def example_ecommerce():
    section("2. E-commerce orders — unique order_id AND customer_email")

    register_context(
        "order_unique",
        ContextSchema(
            description="E-commerce order with unique ID and unique customer email",
            category="ecommerce",
            sample={
                "order_id": "550e8400-e29b-41d4-a716-446655440000",
                "customer_email": "buyer@shop.com",
                "product": "Wireless Headphones",
                "quantity": 2,
                "total_price": 299.99,
                "status": "pending",
            },
            prompt_hints=[
                "Realistic product names (electronics, clothing, home)",
                "Quantity 1-5, total_price matching unit price",
                "Status: pending / shipped / delivered / returned",
            ],
            field_providers={
                "order_id": "faker:uuid4",
                "customer_email": "faker:email",
            },
            unique_fields=["order_id", "customer_email"],
        ),
    )

    gen = DataGenerator()
    records = gen.generate("order_unique", count=15)

    print(f"\n  Generated {len(records)} records.")
    check_uniqueness(records, "order_id")
    check_uniqueness(records, "customer_email")
    print("\n  First 3 records:")
    for r in records[:3]:
        print(f"    {r['order_id']}  {r['customer_email']}")


# ── 3. File-based context with unique_fields in YAML ──────────────────────────

def example_yaml():
    section("3. File-based context (YAML) with unique_fields")

    yaml_path = Path(__file__).parent / "unique_contexts.yaml"
    registered = load_contexts_from_file(yaml_path)
    print(f"\n  Loaded context(s): {registered}")

    gen = DataGenerator()
    records = gen.generate("employee_unique", count=10)

    print(f"  Generated {len(records)} records.")
    check_uniqueness(records, "email")
    print("\n  First 3 records:")
    for r in records[:3]:
        print(f"    {r['name']:30s}  {r['email']}")


# ── 4. generate_from_model with unique_fields ──────────────────────────────────

def example_from_model():
    section("4. generate_from_model + unique_fields kwarg")

    json_schema = {
        "title": "APIUser",
        "description": "Application user accounts",
        "properties": {
            "user_id":  {"type": "string"},
            "username": {"type": "string"},
            "email":    {"type": "string", "format": "email"},
            "role":     {"type": "string"},
            "active":   {"type": "boolean"},
        },
    }

    records = generate_from_model(
        json_schema,
        count=10,
        field_providers={
            "user_id": "faker:uuid4",
            "email":   "faker:email",
        },
        unique_fields=["user_id", "email"],
    )

    print(f"\n  Generated {len(records)} records.")
    check_uniqueness(records, "user_id")
    check_uniqueness(records, "email")
    print("\n  First 3 records:")
    for r in records[:3]:
        print(f"    {r['user_id']}  {r['email']:35s}  role={r.get('role', '?')}")


# ── 5. Batching note ──────────────────────────────────────────────────────────

def example_batching_note():
    section("5. Uniqueness scope: per-batch (important for large counts)")

    print("""
  unique_fields guarantees uniqueness WITHIN a single generate() call.
  When using generate_batched() or generate(count > batch_size), each
  batch gets a fresh Faker instance, so values CAN repeat across batches.

  For cross-batch uniqueness, combine unique_fields with a post-processing
  deduplication step, or use a single large generate() call.

  Example (safe — all 50 records in one call):
      gen.generate("saas_user_unique", count=50)

  Example (per-batch unique only):
      for batch in gen.generate_batched("saas_user_unique", count=200, batch_size=50):
          process(batch)  # each batch is internally unique
    """)


if __name__ == "__main__":
    example_saas_users()
    example_ecommerce()
    example_yaml()
    example_from_model()
    example_batching_note()

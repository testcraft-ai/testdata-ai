"""
Faker hybrid mode examples.

AI generates semantically coherent records; Faker overwrites critical fields
with format-guaranteed values (email, IBAN, phone, etc.).

Requirements:
    pip install testdata-ai[faker]

Run:
    python examples/faker_hybrid.py
"""

import json

from testdata_ai import DataGenerator, generate_from_model, register_context, ContextSchema


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


# ── 1. Built-in context + field_providers via register_context ────────────────

def example_banking():
    section("1. Banking user — AI names + Faker email/IBAN")

    register_context(
        "banking_pl",
        ContextSchema(
            description="Polish retail banking customer",
            category="banking",
            sample={
                "name": "Jan Kowalski",
                "email": "jan.kowalski@bank.pl",
                "iban": "PL61109010140000071219812874",
                "phone": "+48 123 456 789",
                "balance": 4250.00,
                "currency": "PLN",
            },
            prompt_hints=[
                "Use realistic Polish full names",
                "Balance between 500 and 50000 PLN",
                "Currency always PLN",
            ],
            field_providers={
                "email": "faker:email",
                "iban": "faker:iban",
                "phone": "faker:phone_number",
            },
        ),
    )

    gen = DataGenerator(locale="pl_PL")
    records = gen.generate("banking_pl", count=3)

    print("  AI generated names/balance, Faker generated email/iban/phone:\n")
    for r in records:
        print(f"  name   : {r['name']}")
        print(f"  email  : {r['email']}")
        print(f"  iban   : {r['iban']}")
        print(f"  phone  : {r['phone']}")
        print(f"  balance: {r['balance']} {r['currency']}")
        print()


# ── 2. E-commerce — Faker email + uuid ───────────────────────────────────────

def example_ecommerce():
    section("2. E-commerce order — Faker email + order_id")

    register_context(
        "order_with_faker",
        ContextSchema(
            description="E-commerce order with guaranteed-valid email and UUID order ID",
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
                "Realistic product names from electronics/clothing/home categories",
                "Quantity 1-5, total_price matching quantity * unit price",
                "Status: pending / shipped / delivered / returned",
            ],
            field_providers={
                "order_id": "faker:uuid4",
                "customer_email": "faker:email",
            },
        ),
    )

    gen = DataGenerator()
    records = gen.generate("order_with_faker", count=3)

    print("  order_id and customer_email always valid format:\n")
    for r in records:
        print(f"  order_id       : {r['order_id']}")
        print(f"  customer_email : {r['customer_email']}")
        print(f"  product        : {r['product']} x{r['quantity']} = ${r['total_price']}")
        print(f"  status         : {r['status']}")
        print()


# ── 3. generate_from_model with field_providers parameter ─────────────────────

def example_pydantic():
    section("3. generate_from_model + field_providers kwarg")

    # Simulate a Pydantic-like schema as plain JSON Schema dict
    json_schema = {
        "title": "UserAccount",
        "properties": {
            "username": {"type": "string"},
            "email": {"type": "string", "format": "email"},
            "phone": {"type": "string"},
            "age": {"type": "integer"},
            "country": {"type": "string"},
        },
    }

    records = generate_from_model(
        json_schema,
        count=3,
        field_providers={
            "email": "faker:email",
            "phone": "faker:phone_number",
        },
    )

    print("  AI generates username/age/country, Faker handles email/phone:\n")
    for r in records:
        print(f"  {json.dumps(r, indent=4)}")
        print()


# ── 4. Locale-aware: Faker follows DataGenerator.locale ──────────────────────

def example_locale():
    section("4. Locale-aware Faker (pl_PL)")

    register_context(
        "hr_employee_pl",
        ContextSchema(
            description="Polish HR employee record",
            category="hr",
            sample={
                "name": "Anna Wiśniewska",
                "email": "a.wisniewska@firma.pl",
                "phone": "+48 512 345 678",
                "department": "Engineering",
                "salary": 8500,
            },
            prompt_hints=[
                "Polish names with correct Polish diacritics",
                "Department: Engineering / Marketing / HR / Finance / Sales",
                "Monthly salary in PLN between 4000 and 20000",
            ],
            field_providers={
                "email": "faker:email",
                "phone": "faker:phone_number",
            },
        ),
    )

    # locale="pl_PL" propagates to both AI prompt AND Faker instance
    gen = DataGenerator(locale="pl_PL")
    records = gen.generate("hr_employee_pl", count=3)

    print("  Both AI (names) and Faker (email/phone) use pl_PL locale:\n")
    for r in records:
        print(f"  {r['name']:30s}  {r['email']:35s}  {r['phone']}")


if __name__ == "__main__":
    example_banking()
    example_ecommerce()
    example_pydantic()
    example_locale()

"""
Relationship generation examples.

generate() with a {"nodes": {...}} graph generates multiple related entity
datasets with referential integrity. Unlike Faker's sequential approach,
child prompts include sample parent records so the AI produces semantically
coherent data:
  - Order amounts match parent customer's income tier
  - Shipment addresses match the order's destination
  - Employee salaries fit the parent company's size

Covered:
  1. E-commerce: customers → orders (2-level)
  2. Logistics: customers → orders → shipments (3-level chain)
  3. B2B: leads → employees (locale-aware, pl_PL)
  4. Module-level unified generate() function
  5. Graph YAML file + CLI (see examples/ecommerce_graph.yaml)

Run:
    python examples/relationships.py
    testdata-ai generate-related --graph-file examples/ecommerce_graph.yaml -o json
"""

import json

from testdata_ai import DataGenerator, generate


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


# ── 1. E-commerce: customers → orders ────────────────────────────────────────

def example_ecommerce():
    section("1. E-commerce: customers → orders")

    gen = DataGenerator()
    result = gen.generate_with_relationships({
        "customers": {
            "context": "ecommerce_customer",
            "count": 3,
        },
        "orders": {
            "context": "restaurant_order",
            "count": 9,
            "parent": "customers",
            "fk_field": "customer_email",
            "parent_pk": "email",
            "parent_sample_size": 3,   # all 3 parents shown to AI for coherence
        },
    })

    customer_emails = {c["email"] for c in result["customers"]}
    order_fks = {o["customer_email"] for o in result["orders"]}

    print(f"\n  Generated {len(result['customers'])} customers, {len(result['orders'])} orders")
    print(f"\n  Customers:")
    for c in result["customers"]:
        print(f"    {c['email']:40s}  {c.get('subscription_tier', c.get('loyalty_tier', ''))}")

    print(f"\n  Orders (customer_email → referential integrity guaranteed):")
    for o in result["orders"][:5]:
        print(f"    {o.get('order_id', o.get('order_number', '?')):15s}  "
              f"→ {o['customer_email']}")
    if len(result["orders"]) > 5:
        print(f"    ... and {len(result['orders']) - 5} more")

    assert order_fks.issubset(customer_emails), "FK violation!"
    print(f"\n  ✓ All {len(result['orders'])} order.customer_email values are valid customer emails")


# ── 2. Logistics: customers → orders → shipments (3-level) ───────────────────

def example_three_level():
    section("2. Logistics: customers → orders → shipments (3-level)")

    gen = DataGenerator()
    result = gen.generate_with_relationships({
        "customers": {
            "context": "ecommerce_customer",
            "count": 2,
        },
        "orders": {
            "context": "restaurant_order",
            "count": 6,
            "parent": "customers",
            "fk_field": "customer_email",
            "parent_pk": "email",
        },
        "shipments": {
            "context": "logistics_shipment",
            "count": 6,
            "parent": "orders",
            "fk_field": "reference_order_id",
            "parent_pk": "order_id",
            "parent_sample_size": 3,
        },
    })

    print(f"\n  Generated:")
    for entity, records in result.items():
        print(f"    {entity:12s}: {len(records)} records")

    # Verify chain integrity
    customer_emails = {c["email"] for c in result["customers"]}
    order_ids = {o.get("order_id", o.get("order_number")) for o in result["orders"]}
    for o in result["orders"]:
        assert o["customer_email"] in customer_emails
    for s in result["shipments"]:
        assert s["reference_order_id"] in order_ids

    print(f"\n  ✓ Full chain referential integrity: customers → orders → shipments")


# ── 3. B2B: leads → employees, locale pl_PL ──────────────────────────────────

def example_b2b_locale():
    section("3. B2B: leads → employees  (locale pl_PL)")

    gen = DataGenerator(locale="pl_PL")
    result = gen.generate_with_relationships({
        "companies": {
            "context": "b2b_lead",
            "count": 2,
        },
        "employees": {
            "context": "hr_employee",
            "count": 6,
            "parent": "companies",
            "fk_field": "company",       # field injected into each employee record
            "parent_pk": "company",      # b2b_lead uses "company" as the company name field
            "parent_sample_size": 2,
        },
    })

    print(f"\n  Companies:")
    for c in result["companies"]:
        print(f"    {c['company']:40s}  {c.get('industry', '')}")

    print(f"\n  Employees (company matches parent, AI maintains salary coherence):")
    for e in result["employees"]:
        print(f"    {e.get('name', '?'):30s}  → {e['company']}")


# ── 4. Unified generate() function ───────────────────────────────────────────

def example_unified_generate():
    section("4. Unified generate() — {'nodes': {...}} dispatch")

    result = generate(
        {
            "nodes": {
                "customers": {"context": "ecommerce_customer", "count": 2},
                "orders": {
                    "context": "restaurant_order",
                    "count": 4,
                    "parent": "customers",
                    "fk_field": "customer_email",
                    "parent_pk": "email",
                },
            }
        },
        validate=True,
    )

    # result is a RelationshipResult (dict subclass)
    print(f"\n  result.keys() = {list(result.keys())}")
    print(f"  customers    : {len(result['customers'])} records")
    print(f"  orders       : {len(result['orders'])} records")
    print(f"\n  First order:")
    print(f"  {json.dumps(result['orders'][0], indent=4)}")

    # Convert to pandas DataFrames (requires pip install testdata-ai[pandas])
    # dfs = result.to_dataframes()

    # Export to JSON
    # result.to_json("output.json")


# ── 5. CLI usage hint ─────────────────────────────────────────────────────────

def show_cli_hint():
    section("5. CLI usage — generate-related")
    print("""
  # Generate from a graph YAML file, output JSON:
  testdata-ai generate-related --graph-file examples/ecommerce_graph.yaml

  # JSONL format — one line per entity (useful for streaming / jq):
  testdata-ai generate-related --graph-file examples/ecommerce_graph.yaml \\
      -o jsonl-per-entity | jq '.records | length'

  # Pipe orders to jq:
  testdata-ai generate-related --graph-file examples/ecommerce_graph.yaml -q \\
      | jq '.orders[] | {id: .order_id, email: .customer_email}'

  # Graph YAML format (see examples/ecommerce_graph.yaml):
  #
  #   customers:
  #     context: ecommerce_customer
  #     count: 5
  #
  #   orders:
  #     context: restaurant_order
  #     count: 20
  #     parent: customers
  #     fk_field: customer_email
  #     parent_pk: email
  #     parent_sample_size: 3
""")


if __name__ == "__main__":
    example_ecommerce()
    example_three_level()
    example_b2b_locale()
    example_unified_generate()
    show_cli_hint()

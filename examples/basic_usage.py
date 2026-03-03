"""
Basic usage examples of testdata-ai.

Covers:
  - Listing available contexts
  - Inspecting a context schema
  - Generating data with DataGenerator (repeated use)
  - Generating data with generate() (one-off convenience)
"""

import json
from pathlib import Path

from testdata_ai import DataGenerator, generate, list_contexts, get_context_schema


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def main():
    # ── 1. Available contexts ─────────────────────────────────────
    section("Available contexts")
    contexts = list_contexts()
    print(f"  {len(contexts)} built-in contexts:")
    for name in contexts:
        print(f"    • {name}")

    # ── 2. Inspect a context schema ───────────────────────────────
    section("Context schema: ecommerce_customer")
    schema = get_context_schema("ecommerce_customer")
    print(f"  Description : {schema.description}")
    print(f"  Category    : {schema.category}")
    print(f"  Fields      : {', '.join(schema.fields)}")

    # ── 3. Generate with DataGenerator (recommended for repeated use)
    section("Generate 3 ecommerce_customer records")
    gen = DataGenerator()
    print(f"  Provider : {gen.config.provider}")
    print(f"  Model    : {gen.config.model}\n")

    customers = gen.generate("ecommerce_customer", count=3)
    print(f"  Generated {len(customers)} records:")
    for i, c in enumerate(customers, 1):
        print(f"\n  [{i}] {json.dumps(c, indent=4)}")

    output_path = Path(__file__).parent / "output_customers.json"
    output_path.write_text(json.dumps(customers, indent=2))
    print(f"\n  Saved -> {output_path}")

    # ── 4. One-off convenience function ───────────────────────────
    section("One-off: generate() convenience function")
    employees = generate("hr_employee", count=2)
    print(f"  Generated {len(employees)} hr_employee records:")
    for i, e in enumerate(employees, 1):
        print(f"\n  [{i}] {json.dumps(e, indent=4)}")


if __name__ == "__main__":
    main()

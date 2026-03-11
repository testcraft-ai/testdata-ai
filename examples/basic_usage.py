"""
Basic usage examples of testdata-ai.

Covers:
  - Listing available contexts
  - Inspecting a context schema
  - Generating data with DataGenerator (repeated use)
  - Generating data with generate() (one-off convenience, returns GenerateResult)
  - GenerateResult export helpers: to_json(), to_csv(), to_batches()
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

    # ── 4. One-off convenience function → GenerateResult ──────────
    section("One-off: generate() returns GenerateResult")
    result = generate("hr_employee", count=2)

    # GenerateResult is iterable and indexable like a list
    print(f"  Generated {len(result)} hr_employee records:")
    for i, e in enumerate(result, 1):
        print(f"\n  [{i}] {json.dumps(e, indent=4)}")

    # ── 5. GenerateResult export helpers ──────────────────────────
    section("GenerateResult export helpers")
    result = generate("ecommerce_customer", count=3)

    # Save to JSON file
    result.to_json(str(Path(__file__).parent / "output_customers.json"))
    print("  Saved JSON -> examples/output_customers.json")

    # Get CSV string
    csv_text = result.to_csv()
    print(f"\n  CSV (first 200 chars):\n{csv_text[:200]}")

    # Iterate in batches (no extra AI calls)
    print(f"\n  to_batches(batch_size=2):")
    for i, batch in enumerate(result.to_batches(batch_size=2), 1):
        print(f"    batch {i}: {len(batch)} record(s)")


if __name__ == "__main__":
    main()

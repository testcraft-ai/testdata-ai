"""
Async parallel generation examples.

generate_parallel() and async_generate() run multiple AI calls concurrently,
dramatically reducing wall-clock time when generating large datasets.

Uniqueness across parallel calls is guaranteed via two layers:
  1. Prompt injection — each batch gets a unique UUID prefix in the prompt
     (statistical; reduces AI from producing overlapping values)
  2. global_unique_fields + Faker — post-generation dedup replaces confirmed
     duplicates for the specified fields (guaranteed, requires faker extra)

Covered:
  1. Multi-context parallel — customers + accounts + IoT devices simultaneously
  2. Single-context parallel — 3000 customers split into parallel batches
  3. Batches + concurrency — large dataset with batch_size + parallelism cap
  4. Cross-call uniqueness — global_unique_fields dedup example
  5. Explicit labels — two specs for the same context with separate result keys
  6. Locale-aware parallel — different locales per spec

Run:
    python examples/async_generation.py
"""

import asyncio
import time

from testdata_ai import GenerateSpec, async_generate, generate_parallel


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def elapsed(start: float) -> str:
    return f"{time.perf_counter() - start:.1f}s"


# ── 1. Multi-context parallel ─────────────────────────────────────────────────

async def example_multi_context():
    section("1. Multi-context parallel")

    t = time.perf_counter()
    results = await generate_parallel([
        GenerateSpec("ecommerce_customer", count=3, label="customers"),
        GenerateSpec("banking_user",        count=3, label="accounts"),
        GenerateSpec("iot_device",          count=3, label="devices"),
    ])

    print(f"\n  Generated in {elapsed(t)} (all 3 AI calls ran concurrently)")
    for label, records in results.items():
        print(f"  {label:12s}: {len(records)} records")
        first = records[0]
        print(f"             first: {list(first.items())[:2]}")


# ── 2. Single-context parallel (auto-merge) ───────────────────────────────────

async def example_single_context_parallel():
    section("2. Single-context parallel — 3 batches of 5 merged automatically")

    # No label → all three batches merged into results["ecommerce_customer"]
    t = time.perf_counter()
    results = await generate_parallel([
        GenerateSpec("ecommerce_customer", count=5),
        GenerateSpec("ecommerce_customer", count=5),
        GenerateSpec("ecommerce_customer", count=5),
    ])

    print(f"\n  Generated in {elapsed(t)}")
    print(f"  results.keys() = {list(results.keys())}")
    print(f"  Total records  = {len(results['ecommerce_customer'])} (3 × 5 merged)")


# ── 3. async_generate — single context, batches + concurrency cap ─────────────

async def example_async_generate_batched():
    section("3. async_generate() — batch_size + parallelism semaphore")

    # 15 records split into 5 batches of 3, max 2 running at once
    t = time.perf_counter()
    records = await async_generate(
        "ecommerce_customer",
        count=15,
        parallelism=2,
        batch_size=3,
    )

    print(f"\n  Generated in {elapsed(t)}")
    print(f"  Count   : {len(records)} records")
    print(f"  Batching: ceil(15/3) = 5 batches × 3, max 2 concurrent")
    print(f"  First record keys: {list(records[0].keys())}")


# ── 4. Cross-call uniqueness with global_unique_fields ───────────────────────

async def example_unique_fields():
    section("4. Cross-call uniqueness — global_unique_fields=['email']")

    results = await generate_parallel(
        [
            GenerateSpec("ecommerce_customer", count=5, label="segment_a"),
            GenerateSpec("ecommerce_customer", count=5, label="segment_b"),
        ],
        global_unique_fields=["email"],
    )

    all_emails = [
        r["email"]
        for records in results.values()
        for r in records
        if "email" in r
    ]
    unique_emails = set(all_emails)

    print(f"\n  segment_a: {len(results['segment_a'])} records")
    print(f"  segment_b: {len(results['segment_b'])} records")
    print(f"\n  Total emails : {len(all_emails)}")
    print(f"  Unique emails: {len(unique_emails)}")
    if len(all_emails) == len(unique_emails):
        print("  All emails are unique across both segments")
    else:
        print(f"  {len(all_emails) - len(unique_emails)} duplicates remain (field absent in some records)")


# ── 5. Explicit labels — same context, separate result keys ───────────────────

async def example_explicit_labels():
    section("5. Explicit labels — buyers vs sellers (same context)")

    results = await generate_parallel([
        GenerateSpec("ecommerce_customer", count=2, label="buyers"),
        GenerateSpec("ecommerce_customer", count=2, label="sellers"),
    ])

    print(f"\n  results.keys() = {list(results.keys())}")
    print(f"\n  buyers  ({len(results['buyers'])} records):")
    for r in results["buyers"]:
        print(f"    {r.get('email', r.get('name', '?'))}")
    print(f"\n  sellers ({len(results['sellers'])} records):")
    for r in results["sellers"]:
        print(f"    {r.get('email', r.get('name', '?'))}")


# ── 6. Locale-aware parallel ──────────────────────────────────────────────────

async def example_locale_parallel():
    section("6. Locale-aware parallel — mixed locales per spec")

    results = await generate_parallel([
        GenerateSpec("ecommerce_customer", count=2, locale="pl_PL", label="pl_customers"),
        GenerateSpec("ecommerce_customer", count=2, locale="ja_JP", label="jp_customers"),
        GenerateSpec("hr_employee",        count=2, locale="de_DE", label="de_employees"),
    ])

    print(f"\n  Each context generated with its own locale:")
    for label, records in results.items():
        print(f"\n  {label} ({len(records)} records):")
        for r in records:
            name = r.get("name") or r.get("full_name") or r.get("email", "?")
            print(f"    {name}")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    await example_multi_context()
    await example_single_context_parallel()
    await example_async_generate_batched()
    await example_unique_fields()
    await example_explicit_labels()
    await example_locale_parallel()

    section("Summary — API cheatsheet")
    print("""
  # Multi-context, all parallel:
  results = await generate_parallel([
      GenerateSpec("ecommerce_customer", 100, label="buyers"),
      GenerateSpec("banking_user",        50, label="accounts"),
  ], global_unique_fields=["email"])

  # Single-context, auto-merge (no label):
  results = await generate_parallel([
      GenerateSpec("ecommerce_customer", 1000),
      GenerateSpec("ecommerce_customer", 1000),
      GenerateSpec("ecommerce_customer", 1000),
  ])
  records = results["ecommerce_customer"]  # 3000 merged records

  # Single-context convenience wrapper:
  records = await async_generate(
      "ecommerce_customer",
      count=9000,
      parallelism=3,      # max concurrent AI calls
      batch_size=1000,    # records per AI call (9 batches total, 3 waves)
      global_unique_fields=["email"],
  )
""")


if __name__ == "__main__":
    asyncio.run(main())

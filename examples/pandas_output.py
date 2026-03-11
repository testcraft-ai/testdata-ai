"""
Pandas DataFrame output examples.

testdata-ai can convert generated records to pandas DataFrames directly,
useful for data analysis, ML pipelines, CSV export, or notebook workflows.

Requirements:
    pip install testdata-ai[pandas]
    pip install testdata-ai[pandas,openai]   # with an AI provider

Run:
    python examples/pandas_output.py
"""

from testdata_ai import (
    DataGenerator,
    generate,
    register_context,
    ContextSchema,
)


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


# ── 1. result.to_dataframe() on any GenerateResult ────────────────────────────

def example_basic():
    section("1. result.to_dataframe() — flat records")

    result = generate("ecommerce_customer", count=5)
    df = result.to_dataframe()

    print(f"  shape : {df.shape}")
    print(f"  cols  : {list(df.columns)}")
    print()
    print(df[["name", "email", "loyalty_tier"]].to_string(index=False))


# ── 2. One-liner with method chaining ─────────────────────────────────────────

def example_one_liner():
    section("2. One-liner — generate().to_dataframe()")

    df = generate("banking_user", count=10).to_dataframe()

    print(f"  shape  : {df.shape}")
    print(f"  dtypes :\n{df.dtypes.to_string()}")


# ── 3. DataGenerator method ───────────────────────────────────────────────────

def example_generator_method():
    section("3. DataGenerator.generate_as_dataframe()")

    gen = DataGenerator()
    df = gen.generate_as_dataframe("hr_employee", count=8)

    print(f"  Records : {len(df)}")
    print()
    # Quick aggregation example
    if "department" in df.columns:
        print("  Count by department:")
        print(df["department"].value_counts().to_string())


# ── 4. Nested data — flatten=True vs flatten=False ───────────────────────────

def example_nested():
    section("4. Nested fields — flatten=True (default) vs flatten=False")

    register_context(
        "product_nested",
        ContextSchema(
            description="E-commerce product with nested dimensions",
            category="ecommerce",
            sample={
                "sku": "PROD-001",
                "name": "Wireless Headphones",
                "price": 149.99,
                "stock": 42,
                "dimensions": {
                    "weight_kg": 0.3,
                    "width_cm": 18,
                    "height_cm": 20,
                },
            },
            prompt_hints=[
                "Realistic consumer electronics or home goods",
                "Price between 9.99 and 999.99",
                "Stock between 0 and 500",
            ],
        ),
    )

    result = generate("product_nested", count=3)

    df_flat = result.to_dataframe(flatten=True)
    df_nested = result.to_dataframe(flatten=False)

    print("  flatten=True  columns:", list(df_flat.columns))
    print("  flatten=False columns:", list(df_nested.columns))
    print()
    print("  Flat — nested dict expanded to dot-separated columns:")
    dim_cols = [c for c in df_flat.columns if c.startswith("dimensions.")]
    print(df_flat[["sku", "name"] + dim_cols].to_string(index=False))


# ── 5. RelationshipResult.to_dataframes() — multi-entity ─────────────────────

def example_relationships():
    section("5. RelationshipResult.to_dataframes() — multi-entity")

    result = generate({
        "nodes": {
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
            },
        }
    })

    dfs = result.to_dataframes()

    for entity, df in dfs.items():
        print(f"\n  [{entity}]  shape={df.shape}")
        print(f"  cols: {list(df.columns)}")

    # Example join
    customers_df = dfs["customers"]
    orders_df = dfs["orders"]
    if "customer_email" in orders_df.columns and "email" in customers_df.columns:
        merged = orders_df.merge(
            customers_df[["email", "name"]],
            left_on="customer_email",
            right_on="email",
            how="left",
        )
        print(f"\n  Joined orders+customers: {merged.shape}")
        print(f"  cols: {list(merged.columns)}")


if __name__ == "__main__":
    example_basic()
    example_one_liner()
    example_generator_method()
    example_nested()
    example_relationships()

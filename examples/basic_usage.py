"""
Basic usage example of testdata-ai generator.
"""

import json
from pathlib import Path

from testdata_ai import TestDataGenerator


def main():
    """Run basic example."""
    print("=" * 60)
    print("🤖 testdata-ai - Basic Usage Example")
    print("=" * 60)
    
    # Initialize generator (reads from .env)
    gen = TestDataGenerator()
    
    print(f"\nUsing provider: {gen.config.provider}")
    print(f"Model: {gen.config.model}")
    
    # Generate customers
    print("\n📊 Generating 3 ecommerce customers...")
    customers = gen.generate("ecommerce_customer", count=3)
    
    print(f"\n✅ Success! Generated {len(customers)} customers\n")
    
    # Display results
    for i, customer in enumerate(customers, 1):
        print(f"\n{'='*60}")
        print(f"Customer {i}:")
        print(f"{'='*60}")
        print(json.dumps(customer, indent=2))
    
    # Save to file
    output_file = Path(__file__).parent / "output_customers.json"
    with open(output_file, "w") as f:
        json.dump(customers, f, indent=2)
    
    print(f"\n💾 Saved to: {output_file}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
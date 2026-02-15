"""
Context schemas for AI test data generation.

Each context defines:
- Description and category
- Expected fields for validation
- Sample record (used as few-shot example in prompts)
- Prompt hints (requirements for realistic data generation)
"""

__all__ = [
    "ContextSchema",
    "get_context_schema",
    "list_contexts",
    "validate_generated_data",
]

from dataclasses import dataclass
from typing import Dict, List, Any, Optional


@dataclass
class ContextSchema:
    """Schema definition for a test data context.

    The required fields are derived from the top-level keys of ``sample``,
    so there is a single source of truth for the record structure.
    """

    description: str
    sample: Dict[str, Any]
    prompt_hints: List[str]
    category: str = "general"

    @property
    def fields(self) -> List[str]:
        """Required field names, derived from sample keys."""
        return list(self.sample.keys())

    def validate_record(self, record: Dict[str, Any]) -> bool:
        """Check if a record has all required fields."""
        return all(f in record for f in self.fields)

    def missing_fields(self, record: Dict[str, Any]) -> List[str]:
        """Return list of required fields missing from a record."""
        return [f for f in self.fields if f not in record]


CONTEXTS: Dict[str, ContextSchema] = {
    "ecommerce_customer": ContextSchema(
        description="e-commerce customer profiles",
        category="ecommerce",
        sample={
            "name": "Aisha Patel",
            "email": "aisha.patel.2024@gmail.com",
            "age": 28,
            "location": {
                "city": "Mumbai",
                "country": "India",
                "timezone": "Asia/Kolkata",
            },
            "shopping_behavior": {
                "frequency": "weekly",
                "avg_order_value": "$45-80",
                "preferred_categories": ["electronics", "books"],
                "device": "mobile",
                "payment_method": "upi",
            },
            "joined_date": "2023-04-15",
            "loyalty_tier": "silver",
        },
        prompt_hints=[
            "Diverse names from different cultures and countries",
            "Age range: 18-75 years old",
            "Realistic email addresses (avoid test@test.com patterns)",
            "Valid location data (city, country, timezone)",
            "Shopping behavior should match demographics (e.g., students shop differently than seniors)",
            "Payment methods should match location (e.g., UPI in India, credit cards in USA)",
        ],
    ),

    "banking_user": ContextSchema(
        description="banking customer profiles",
        category="finance",
        sample={
            "name": "Marcus Johnson",
            "email": "mjohnson87@outlook.com",
            "age": 42,
            "account_type": "checking",
            "balance": 15420.50,
            "monthly_income": 5200,
            "credit_score": 740,
            "branch": "Austin-Downtown",
            "account_opened": "2021-11-03",
        },
        prompt_hints=[
            "Diverse names from different cultures",
            "Age range: 18-80 years old",
            "Realistic email addresses",
            "Valid account types (checking, savings, business, investment)",
            "Realistic balance ranges based on account type and customer demographics",
            "Credit scores: 300-850 range",
            "Monthly income should correlate with age and account balance",
        ],
    ),

    "saas_trial": ContextSchema(
        description="SaaS trial user profiles",
        category="saas",
        sample={
            "name": "Sarah Chen",
            "email": "sarah.chen@techstartup.io",
            "company": "TechStartup Inc",
            "role": "CTO",
            "plan": "business",
            "signup_date": "2026-02-01",
            "trial_expires": "2026-03-01",
            "usage_stats": {
                "logins": 12,
                "features_used": ["api", "analytics", "integrations"],
            },
        },
        prompt_hints=[
            "Business-focused names and emails (company domains)",
            "Age range: 25-60 years old (professional users)",
            "Realistic company names and roles",
            "Trial plan types: free, professional, business, enterprise",
            "Signup dates within last 3 months",
            "Trial expiry dates (typically 14-30 days from signup)",
        ],
    ),
}


def get_context_schema(context: str) -> ContextSchema:
    """Get schema definition for a given context.

    Raises:
        ValueError: If context is not recognized
    """
    if context not in CONTEXTS:
        available = list(CONTEXTS.keys())
        raise ValueError(
            f"Unknown context: '{context}'. Available contexts: {available}"
        )
    return CONTEXTS[context]


def list_contexts(category: Optional[str] = None) -> List[str]:
    """List available context names, optionally filtered by category."""
    if category is None:
        return list(CONTEXTS.keys())
    return [
        name for name, schema in CONTEXTS.items()
        if schema.category == category
    ]


def validate_generated_data(
    context: str, records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Validate generated records against the context schema.

    Args:
        context: Context identifier
        records: List of generated records to validate

    Returns:
        List of invalid-record dicts, each with 'record_index' and
        'missing_fields'. Empty list means all records are valid.
    """
    schema = get_context_schema(context)
    return [
        {"record_index": i, "missing_fields": schema.missing_fields(record)}
        for i, record in enumerate(records)
        if not schema.validate_record(record)
    ]

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

    "healthcare_patient": ContextSchema(
        description="healthcare patient records",
        category="healthcare",
        sample={
            "patient_id": "PT-2024-00847",
            "name": "Elena Rodriguez",
            "date_of_birth": "1985-06-12",
            "gender": "female",
            "blood_type": "A+",
            "primary_diagnosis": "Type 2 Diabetes",
            "medications": ["Metformin 500mg", "Lisinopril 10mg"],
            "allergies": ["Penicillin"],
            "insurance_provider": "Blue Cross Blue Shield",
            "last_visit": "2026-01-20",
            "attending_physician": "Dr. Kwame Asante",
        },
        prompt_hints=[
            "Diverse patient names from different cultures",
            "Realistic date of birth (age range 0-100)",
            "Valid blood types: A+, A-, B+, B-, AB+, AB-, O+, O-",
            "Common real-world diagnoses (diabetes, hypertension, asthma, etc.)",
            "Medications should match the diagnosis",
            "Allergies should be common drug/food allergies",
            "Use realistic insurance provider names",
        ],
    ),

    "education_student": ContextSchema(
        description="university student profiles",
        category="education",
        sample={
            "student_id": "STU-2023-1042",
            "name": "Jordan Williams",
            "email": "jwilliams@university.edu",
            "age": 20,
            "major": "Computer Science",
            "minor": "Mathematics",
            "year": "junior",
            "gpa": 3.45,
            "enrollment_status": "full-time",
            "courses": ["CS301 Algorithms", "MATH240 Linear Algebra", "CS350 Databases"],
            "advisor": "Dr. Priya Sharma",
        },
        prompt_hints=[
            "Diverse student names from different backgrounds",
            "Age range: 17-30 for undergrads, 22-45 for graduate students",
            "University email addresses (.edu domain)",
            "Realistic majors and minors that pair logically",
            "Year: freshman, sophomore, junior, senior, or graduate",
            "GPA range: 0.0-4.0, normally distributed around 3.0",
            "Course codes should follow a realistic format (dept + number)",
        ],
    ),

    "b2b_lead": ContextSchema(
        description="B2B sales lead profiles",
        category="b2b",
        sample={
            "lead_id": "LD-2026-0293",
            "contact_name": "David Nguyen",
            "email": "d.nguyen@acmelogistics.com",
            "phone": "+1-312-555-0184",
            "company": "Acme Logistics",
            "industry": "Supply Chain & Logistics",
            "company_size": "200-500",
            "job_title": "VP of Operations",
            "lead_source": "webinar",
            "lead_score": 72,
            "deal_value": 48000,
            "stage": "qualified",
            "notes": "Interested in warehouse automation. Follow up after Q1 budget review.",
        },
        prompt_hints=[
            "Professional names with company email domains",
            "Realistic company names across industries",
            "Industries: tech, healthcare, logistics, finance, manufacturing, retail, etc.",
            "Company size brackets: 1-10, 11-50, 51-200, 200-500, 500-1000, 1000+",
            "Job titles should reflect decision-makers (VP, Director, Head of, Manager)",
            "Lead sources: webinar, referral, inbound, cold outreach, trade show, content download",
            "Lead scores: 0-100, higher means more qualified",
            "Stages: new, contacted, qualified, proposal, negotiation, closed-won, closed-lost",
        ],
    ),

    "hr_employee": ContextSchema(
        description="HR employee records",
        category="hr",
        sample={
            "employee_id": "EMP-004231",
            "name": "Fatima Al-Rashid",
            "email": "f.alrashid@globecorp.com",
            "department": "Engineering",
            "job_title": "Senior Software Engineer",
            "hire_date": "2022-03-14",
            "salary": 125000,
            "employment_type": "full-time",
            "manager": "Tomoko Hayashi",
            "location": "San Francisco, CA",
            "performance_rating": 4,
        },
        prompt_hints=[
            "Diverse employee names reflecting a global workforce",
            "Company email addresses matching a consistent domain",
            "Departments: Engineering, Sales, Marketing, HR, Finance, Operations, Legal, Product",
            "Job titles should match department and seniority level",
            "Hire dates within the last 0-20 years",
            "Salary should correlate with role seniority and department",
            "Employment types: full-time, part-time, contract, intern",
            "Performance ratings: 1-5 scale, normally distributed around 3",
        ],
    ),

    "real_estate_listing": ContextSchema(
        description="real estate property listings",
        category="real_estate",
        sample={
            "listing_id": "MLS-782451",
            "address": "1247 Oak Street, Portland, OR 97205",
            "property_type": "single-family",
            "bedrooms": 3,
            "bathrooms": 2,
            "sqft": 1850,
            "year_built": 1998,
            "list_price": 485000,
            "status": "active",
            "days_on_market": 14,
            "agent": "Rebecca Torres",
            "features": ["hardwood floors", "updated kitchen", "fenced yard"],
        },
        prompt_hints=[
            "Realistic US addresses with valid city/state/zip combinations",
            "Property types: single-family, condo, townhouse, multi-family, land",
            "Bedrooms: 1-6, bathrooms: 1-4 (should correlate with sqft)",
            "Square footage should match property type (condos smaller, houses larger)",
            "Year built: 1900-2026",
            "List prices should match location and property characteristics",
            "Status: active, pending, sold, withdrawn",
            "Features should be realistic and match the property type",
        ],
    ),

    "iot_device": ContextSchema(
        description="IoT device telemetry records",
        category="iot",
        sample={
            "device_id": "IOT-TH-00412",
            "device_type": "temperature_humidity_sensor",
            "manufacturer": "SenseTech",
            "firmware_version": "2.4.1",
            "location": "Warehouse B - Aisle 3",
            "status": "online",
            "battery_level": 87,
            "last_reading": {
                "temperature_c": 22.4,
                "humidity_pct": 58.1,
                "timestamp": "2026-02-15T08:32:00Z",
            },
            "alert_threshold": {"temp_max": 30.0, "humidity_max": 70.0},
            "installed_date": "2025-06-10",
        },
        prompt_hints=[
            "Device types: temperature_humidity_sensor, motion_detector, smart_meter, air_quality, pressure_gauge",
            "Realistic manufacturer names for industrial IoT",
            "Firmware versions in semver format (major.minor.patch)",
            "Status: online, offline, maintenance, error",
            "Battery levels: 0-100, some devices should be low",
            "Sensor readings should be physically plausible for the device type",
            "Timestamps in ISO 8601 format with timezone",
        ],
    ),

    "social_media_profile": ContextSchema(
        description="social media user profiles",
        category="social_media",
        sample={
            "username": "travel_with_miko",
            "display_name": "Miko Tanaka",
            "bio": "Exploring the world one city at a time. Tokyo -> NYC -> ???",
            "followers": 12400,
            "following": 843,
            "posts": 327,
            "verified": False,
            "joined": "2023-08-15",
            "category": "travel",
            "engagement_rate": 3.2,
            "top_hashtags": ["#wanderlust", "#streetphotography", "#foodie"],
        },
        prompt_hints=[
            "Creative usernames that feel organic (underscores, numbers, abbreviations)",
            "Bios should feel authentic and match the category",
            "Follower counts: realistic distribution (most users < 5k, some 5k-100k, rare > 100k)",
            "Following count usually lower than or comparable to followers for larger accounts",
            "Categories: travel, food, fitness, tech, fashion, art, music, gaming, lifestyle",
            "Engagement rates: 1-8% is typical, higher for smaller accounts",
            "Hashtags should match the profile category",
        ],
    ),

    "travel_booking": ContextSchema(
        description="travel booking records",
        category="travel",
        sample={
            "booking_id": "BK-2026-18743",
            "passenger_name": "Carlos Mendez",
            "email": "carlos.mendez@email.com",
            "trip_type": "round-trip",
            "origin": "LAX",
            "destination": "NRT",
            "departure_date": "2026-04-10",
            "return_date": "2026-04-24",
            "cabin_class": "economy",
            "total_price": 1245.00,
            "currency": "USD",
            "travelers": 2,
            "status": "confirmed",
            "add_ons": ["extra_baggage", "travel_insurance"],
        },
        prompt_hints=[
            "Diverse passenger names from different nationalities",
            "Trip types: one-way, round-trip, multi-city",
            "Use real IATA airport codes (LAX, JFK, LHR, NRT, CDG, etc.)",
            "Departure dates within the next 6 months",
            "Return dates after departure (duration 2-30 days)",
            "Cabin classes: economy, premium_economy, business, first",
            "Prices should match route distance and cabin class",
            "Status: confirmed, pending, cancelled, checked-in",
        ],
    ),

    "restaurant_order": ContextSchema(
        description="restaurant / food delivery orders",
        category="food",
        sample={
            "order_id": "ORD-20260215-0042",
            "customer_name": "Lisa Park",
            "restaurant": "Nonna's Trattoria",
            "cuisine": "Italian",
            "items": [
                {"name": "Margherita Pizza", "qty": 1, "price": 14.50},
                {"name": "Caesar Salad", "qty": 1, "price": 9.00},
                {"name": "Tiramisu", "qty": 2, "price": 7.50},
            ],
            "subtotal": 38.50,
            "delivery_fee": 3.99,
            "tip": 6.00,
            "total": 48.49,
            "payment_method": "credit_card",
            "order_type": "delivery",
            "status": "delivered",
            "ordered_at": "2026-02-15T19:22:00Z",
        },
        prompt_hints=[
            "Diverse customer names",
            "Creative but realistic restaurant names matching the cuisine",
            "Cuisines: Italian, Japanese, Mexican, Indian, Chinese, Thai, American, Mediterranean, Korean",
            "Menu items should match the cuisine and restaurant style",
            "Item prices: appetizers $5-12, mains $10-25, desserts $6-12, drinks $3-8",
            "Subtotal must equal sum of (qty * price) for all items",
            "Order types: delivery, pickup, dine-in",
            "Status: placed, preparing, out_for_delivery, delivered, cancelled",
        ],
    ),

    "logistics_shipment": ContextSchema(
        description="logistics shipment tracking records",
        category="logistics",
        sample={
            "tracking_number": "TRK-9827461053",
            "carrier": "FastFreight Global",
            "origin": {"city": "Shenzhen", "country": "China"},
            "destination": {"city": "Chicago", "country": "United States"},
            "ship_date": "2026-01-28",
            "estimated_delivery": "2026-02-18",
            "actual_delivery": None,
            "weight_kg": 245.0,
            "dimensions_cm": {"length": 120, "width": 80, "height": 90},
            "contents": "Consumer Electronics",
            "status": "in_transit",
            "last_checkpoint": "Port of Long Beach, CA",
        },
        prompt_hints=[
            "Tracking numbers: alphanumeric, 10-15 characters",
            "Realistic carrier names (mix of real-sounding freight companies)",
            "International routes with realistic origin/destination pairs",
            "Ship dates within the last 30 days",
            "Delivery estimates: 2-5 days domestic, 7-30 days international",
            "actual_delivery is null for in-transit shipments",
            "Weight and dimensions should match the contents category",
            "Status: pending_pickup, picked_up, in_transit, customs_hold, out_for_delivery, delivered, exception",
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

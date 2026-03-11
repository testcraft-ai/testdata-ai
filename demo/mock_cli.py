#!/usr/bin/env python3
"""Demo driver — runs the real testdata-ai CLI with fixture data.

Instead of calling any AI provider the mock:
  1. Patches DataGenerator.generate() to return pre-canned records.
  2. Sleeps 1.5 s per call so the spinner is clearly visible in the GIF.
  3. Returns exactly `count` records every time.

Usage (mirrors the real CLI):
  python3 demo/mock_cli.py list-contexts
  python3 demo/mock_cli.py generate --context ecommerce_customer --count 3
  python3 demo/mock_cli.py generate --context ecommerce_customer --count 4 --batch-size 2 -o jsonl
  python3 demo/mock_cli.py generate --context hr_employee --count 3 -o sql --table employees
  python3 demo/mock_cli.py show-context banking_user
"""

import os
import sys
import time
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Fixture data — realistic pre-canned records (20 per context).
# Sliced to exactly `count` at call time so counts are always correct.
# ---------------------------------------------------------------------------

_ECOMMERCE_CUSTOMERS = [
    {"name": "Aisha Patel", "email": "aisha.patel@gmail.com", "age": 28,
     "location": {"city": "Mumbai", "country": "India", "timezone": "Asia/Kolkata"},
     "shopping_behavior": {"frequency": "weekly", "avg_order_value": "$45-80",
                           "preferred_categories": ["electronics", "books"],
                           "device": "mobile", "payment_method": "upi"},
     "joined_date": "2023-04-15", "loyalty_tier": "silver"},
    {"name": "Carlos Rivera", "email": "c.rivera94@hotmail.com", "age": 31,
     "location": {"city": "Mexico City", "country": "Mexico", "timezone": "America/Mexico_City"},
     "shopping_behavior": {"frequency": "biweekly", "avg_order_value": "$30-60",
                           "preferred_categories": ["clothing", "sports"],
                           "device": "mobile", "payment_method": "credit_card"},
     "joined_date": "2022-11-08", "loyalty_tier": "bronze"},
    {"name": "Emma Johansson", "email": "emma.johansson@outlook.se", "age": 26,
     "location": {"city": "Stockholm", "country": "Sweden", "timezone": "Europe/Stockholm"},
     "shopping_behavior": {"frequency": "monthly", "avg_order_value": "$90-150",
                           "preferred_categories": ["home_decor", "fashion"],
                           "device": "desktop", "payment_method": "swish"},
     "joined_date": "2024-01-20", "loyalty_tier": "gold"},
    {"name": "Kwame Asante", "email": "kwame.asante@gmail.com", "age": 34,
     "location": {"city": "Accra", "country": "Ghana", "timezone": "Africa/Accra"},
     "shopping_behavior": {"frequency": "weekly", "avg_order_value": "$20-40",
                           "preferred_categories": ["electronics", "clothing"],
                           "device": "mobile", "payment_method": "mobile_money"},
     "joined_date": "2023-07-03", "loyalty_tier": "bronze"},
    {"name": "Li Wei", "email": "liwei2024@163.com", "age": 22,
     "location": {"city": "Shanghai", "country": "China", "timezone": "Asia/Shanghai"},
     "shopping_behavior": {"frequency": "daily", "avg_order_value": "$15-35",
                           "preferred_categories": ["beauty", "snacks", "electronics"],
                           "device": "mobile", "payment_method": "alipay"},
     "joined_date": "2023-12-01", "loyalty_tier": "platinum"},
    {"name": "Priya Sharma", "email": "priya.sharma88@yahoo.in", "age": 38,
     "location": {"city": "Bengaluru", "country": "India", "timezone": "Asia/Kolkata"},
     "shopping_behavior": {"frequency": "biweekly", "avg_order_value": "$55-100",
                           "preferred_categories": ["books", "kitchen", "clothing"],
                           "device": "desktop", "payment_method": "net_banking"},
     "joined_date": "2021-09-14", "loyalty_tier": "gold"},
    {"name": "Omar Hassan", "email": "omar.h@protonmail.com", "age": 29,
     "location": {"city": "Cairo", "country": "Egypt", "timezone": "Africa/Cairo"},
     "shopping_behavior": {"frequency": "monthly", "avg_order_value": "$40-70",
                           "preferred_categories": ["electronics", "automotive"],
                           "device": "mobile", "payment_method": "cash_on_delivery"},
     "joined_date": "2022-05-30", "loyalty_tier": "silver"},
    {"name": "Maria Santos", "email": "mariasantos@gmail.com", "age": 45,
     "location": {"city": "São Paulo", "country": "Brazil", "timezone": "America/Sao_Paulo"},
     "shopping_behavior": {"frequency": "weekly", "avg_order_value": "$60-120",
                           "preferred_categories": ["fashion", "beauty", "home_decor"],
                           "device": "mobile", "payment_method": "pix"},
     "joined_date": "2020-03-18", "loyalty_tier": "platinum"},
    {"name": "James O'Brien", "email": "jobrien@eircom.net", "age": 52,
     "location": {"city": "Dublin", "country": "Ireland", "timezone": "Europe/Dublin"},
     "shopping_behavior": {"frequency": "monthly", "avg_order_value": "$80-160",
                           "preferred_categories": ["sports", "books", "garden"],
                           "device": "desktop", "payment_method": "credit_card"},
     "joined_date": "2019-08-22", "loyalty_tier": "gold"},
    {"name": "Yuki Tanaka", "email": "yuki.tanaka@docomo.ne.jp", "age": 27,
     "location": {"city": "Tokyo", "country": "Japan", "timezone": "Asia/Tokyo"},
     "shopping_behavior": {"frequency": "weekly", "avg_order_value": "$35-65",
                           "preferred_categories": ["anime_merch", "electronics", "fashion"],
                           "device": "mobile", "payment_method": "ic_card"},
     "joined_date": "2023-02-11", "loyalty_tier": "silver"},
    {"name": "Fatima Al-Rashid", "email": "fatima.rashid@gmail.com", "age": 33,
     "location": {"city": "Dubai", "country": "UAE", "timezone": "Asia/Dubai"},
     "shopping_behavior": {"frequency": "weekly", "avg_order_value": "$120-250",
                           "preferred_categories": ["luxury_fashion", "beauty", "home"],
                           "device": "mobile", "payment_method": "credit_card"},
     "joined_date": "2022-10-05", "loyalty_tier": "platinum"},
    {"name": "Liam Murphy", "email": "l.murphy@bigpond.com", "age": 41,
     "location": {"city": "Melbourne", "country": "Australia", "timezone": "Australia/Melbourne"},
     "shopping_behavior": {"frequency": "biweekly", "avg_order_value": "$70-130",
                           "preferred_categories": ["outdoor", "electronics", "sports"],
                           "device": "desktop", "payment_method": "paypal"},
     "joined_date": "2021-06-07", "loyalty_tier": "gold"},
    {"name": "Ana García", "email": "anagarcia92@hotmail.es", "age": 34,
     "location": {"city": "Madrid", "country": "Spain", "timezone": "Europe/Madrid"},
     "shopping_behavior": {"frequency": "weekly", "avg_order_value": "$45-90",
                           "preferred_categories": ["fashion", "home_decor", "food"],
                           "device": "mobile", "payment_method": "bizum"},
     "joined_date": "2022-04-19", "loyalty_tier": "silver"},
    {"name": "David Kim", "email": "dkim.seoul@naver.com", "age": 25,
     "location": {"city": "Seoul", "country": "South Korea", "timezone": "Asia/Seoul"},
     "shopping_behavior": {"frequency": "daily", "avg_order_value": "$25-50",
                           "preferred_categories": ["kbeauty", "fashion", "electronics"],
                           "device": "mobile", "payment_method": "kakao_pay"},
     "joined_date": "2024-02-28", "loyalty_tier": "bronze"},
    {"name": "Amara Diallo", "email": "amara.diallo@orange.sn", "age": 30,
     "location": {"city": "Dakar", "country": "Senegal", "timezone": "Africa/Dakar"},
     "shopping_behavior": {"frequency": "monthly", "avg_order_value": "$25-55",
                           "preferred_categories": ["clothing", "food", "electronics"],
                           "device": "mobile", "payment_method": "wave"},
     "joined_date": "2023-01-15", "loyalty_tier": "bronze"},
    {"name": "Noah Bergmann", "email": "n.bergmann@web.de", "age": 36,
     "location": {"city": "Berlin", "country": "Germany", "timezone": "Europe/Berlin"},
     "shopping_behavior": {"frequency": "biweekly", "avg_order_value": "$65-110",
                           "preferred_categories": ["electronics", "outdoor", "books"],
                           "device": "desktop", "payment_method": "sofort"},
     "joined_date": "2021-11-30", "loyalty_tier": "silver"},
    {"name": "Isabel Fernandes", "email": "isabel.f@sapo.pt", "age": 44,
     "location": {"city": "Lisbon", "country": "Portugal", "timezone": "Europe/Lisbon"},
     "shopping_behavior": {"frequency": "weekly", "avg_order_value": "$50-95",
                           "preferred_categories": ["fashion", "beauty", "kitchen"],
                           "device": "mobile", "payment_method": "mbway"},
     "joined_date": "2020-07-12", "loyalty_tier": "gold"},
    {"name": "Raj Patel", "email": "raj.patel@gmail.co.uk", "age": 39,
     "location": {"city": "London", "country": "UK", "timezone": "Europe/London"},
     "shopping_behavior": {"frequency": "weekly", "avg_order_value": "$75-140",
                           "preferred_categories": ["electronics", "fashion", "sports"],
                           "device": "desktop", "payment_method": "contactless"},
     "joined_date": "2019-05-04", "loyalty_tier": "platinum"},
    {"name": "Mei Ling", "email": "meiling1998@yahoo.com.tw", "age": 26,
     "location": {"city": "Taipei", "country": "Taiwan", "timezone": "Asia/Taipei"},
     "shopping_behavior": {"frequency": "weekly", "avg_order_value": "$30-60",
                           "preferred_categories": ["fashion", "beauty", "stationery"],
                           "device": "mobile", "payment_method": "line_pay"},
     "joined_date": "2023-09-17", "loyalty_tier": "silver"},
    {"name": "Aleksei Volkov", "email": "volkov.aleksei@mail.ru", "age": 47,
     "location": {"city": "Warsaw", "country": "Poland", "timezone": "Europe/Warsaw"},
     "shopping_behavior": {"frequency": "monthly", "avg_order_value": "$55-100",
                           "preferred_categories": ["tools", "electronics", "automotive"],
                           "device": "desktop", "payment_method": "blik"},
     "joined_date": "2020-12-09", "loyalty_tier": "gold"},
]

_HR_EMPLOYEES = [
    {"employee_id": "EMP-001042", "name": "Aisha Patel", "email": "a.patel@globecorp.com",
     "department": "Engineering", "job_title": "Senior Software Engineer",
     "hire_date": "2021-03-08", "salary": 128000, "employment_type": "full-time",
     "manager": "Tomoko Hayashi", "location": "San Francisco, CA", "performance_rating": 5},
    {"employee_id": "EMP-002187", "name": "Carlos Rivera", "email": "c.rivera@globecorp.com",
     "department": "Product", "job_title": "Product Manager",
     "hire_date": "2020-07-14", "salary": 135000, "employment_type": "full-time",
     "manager": "Sarah Chen", "location": "New York, NY", "performance_rating": 4},
    {"employee_id": "EMP-003356", "name": "Emma Johansson", "email": "e.johansson@globecorp.com",
     "department": "Design", "job_title": "UX Designer",
     "hire_date": "2022-11-01", "salary": 98000, "employment_type": "full-time",
     "manager": "David Kim", "location": "Remote", "performance_rating": 4},
    {"employee_id": "EMP-004521", "name": "Kwame Asante", "email": "k.asante@globecorp.com",
     "department": "Data", "job_title": "Data Engineer",
     "hire_date": "2023-02-20", "salary": 112000, "employment_type": "full-time",
     "manager": "Tomoko Hayashi", "location": "Chicago, IL", "performance_rating": 3},
    {"employee_id": "EMP-005670", "name": "Yuki Tanaka", "email": "y.tanaka@globecorp.com",
     "department": "Engineering", "job_title": "Staff Engineer",
     "hire_date": "2019-05-17", "salary": 165000, "employment_type": "full-time",
     "manager": "Sarah Chen", "location": "Seattle, WA", "performance_rating": 5},
    {"employee_id": "EMP-006811", "name": "Maria Santos", "email": "m.santos@globecorp.com",
     "department": "Marketing", "job_title": "Growth Lead",
     "hire_date": "2022-08-09", "salary": 105000, "employment_type": "full-time",
     "manager": "Ana García", "location": "Austin, TX", "performance_rating": 4},
]

_FIXTURES: dict = {
    "ecommerce_customer": _ECOMMERCE_CUSTOMERS,
    "hr_employee": _HR_EMPLOYEES,
}

# ---------------------------------------------------------------------------
# Mock patch
# ---------------------------------------------------------------------------

def _make_mock_generate(delay: float = 0.8):
    """Return a patched DataGenerator.generate() that uses fixture data."""
    def _mock(self, context: str, count: int = 10, validate: bool = True):
        time.sleep(delay)
        pool = (_FIXTURES.get(context) or _ECOMMERCE_CUSTOMERS) * 10
        return pool[:count]
    return _mock


# ---------------------------------------------------------------------------
# Entry point — patch and hand off to the real CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Ollama requires no package import, so set it as the provider.
    # The mock never actually calls the network — it patches generate() directly.
    os.environ.setdefault("AI_PROVIDER", "ollama")
    os.environ.setdefault("OLLAMA_MODEL", "demo")

    from testdata_ai.generator import DataGenerator
    from testdata_ai.cli import _Spinner, cli

    # Block-element quadrants (U+2596–U+259F) are in every terminal font
    # including agg's bundled JetBrains Mono (powerline uses them).
    # Braille / arrows are NOT in JetBrains Mono — they render as [?] in GIF.
    # Production CLI keeps its Braille spinner unchanged.
    _Spinner.FRAMES = "▁▂▃▄▅▆▇█▇▆▅▄▃▁"
    _Spinner.INTERVAL = 0.12

    with patch.object(DataGenerator, "generate", _make_mock_generate(delay=1.5)):
        cli()

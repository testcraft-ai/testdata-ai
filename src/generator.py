"""
Core test data generator - provider agnostic.
Supports OpenAI, Anthropic, and other AI providers.
"""

import json
from typing import Dict, List, Any, Optional
import logging

from src.config import Config, AIProviderConfig
from src.ai_providers import get_provider, AIProvider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestDataGenerator:
    """
    AI-powered test data generator.

    Generates realistic, context-aware test data using AI providers
    (OpenAI, Anthropic, or others).

    Example:
        >>> # Using default provider from .env
        >>> gen = TestDataGenerator()
        >>> data = gen.generate("ecommerance_customer", count=10)
        
        >>> # Using specified provider
        >>> gen = TestDataGenerator(provider="anthropic")
        >>> data = gen.generate("banking_user", count=5)
    """

    def __init__(
            self,
            provider: Optional[str] = None,
            api_key: Optional[str] = None,
            model: Optional[str] = None,
            temperature: Optional[float] = None
    ):
        """
        Initialize generator.
        
        Args:
            provider: AI provider name ('openai', 'anthropic', or None for default)
            api_key: API key (if None, reads from .env based on provider)
            model: Model name (if None, uses defauld provider)
            temperature: Sampling temperature 0.0-1.0 (if None, uses default)

        Note:
            If arguments are None, reads from .env file using Config.
            This allow flexible usage:
            - TestDataGenerator() = reads all from .env
            - TestDataGenerator(provider="anthropic") = use Anthropic with .env settings
            - TestDataGenerator(api-key="sk-...") = override API key
        """
        # Get config from .env or use overrides
        if api_key:
            # User provied explicit config
            provider = provider or "openai" # Default if not specified

            # Build config manually
            self.config = AIProviderConfig(
                provider=provider,
                api_key=api_key,
                model=model or ("gpt-4-turbo-preview" if provider == "openai" else "claude-3-sonnet-20240229"),
                temperature=temperature or 0.7
            )
        else:
            # Load from .env via Config
            self.config = Config.get_provider_config(provider)

        # Initialize AI provider
        self.provider: AIProvider = get_provider(
            provider_name=self.config.provider,
            api_key=self.config.api_key,
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        logger.info(f"Initialized generator with {self.config.provider} provider (model: {self.config.model})")

    def generate(
        self,
        context: str,
        count: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate test data for given context.
        
        Args:
            context: Type of data to generate (e.g., "ecommenrance_customer, "banking_user")
            count: Number of records to generate
            **kwargs: Additional parameters (persone, filters, etc.)

        Returns:
            List of generated data records as dictionaries

        Examples:
            >>> gen = TestDataGenerator()
            >>> customers = gen.generate("ecommerce_customer", count=5)
            >>> print(len(customers)) # 5
            >>> print(customers[0]["email"]) # realistic email
        """
        logger.info(f"Generating {count} {context} records...")

        prompt = self._build_prompt(context, count, **kwargs)
        system_prompt = "You are a test data generator. Generate realisitc, diverse data in JSON format."

        # Call AI provider
        try:
            response_text = self.provider.generate(prompt, system_prompt)

            data = json.loads(response_text)

            # Extract data array
            if "data" in data:
                records = data["data"]
            elif isinstance(data, list):
                records = data
            else:
                # Wrap single object in array
                records = [data]
            
            logger.info(f"Succesfully generated {len(records)} records")
            return records
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response as JSON: {e}")
            logger.error(f"Response was: {response_text[:200]}...")
            raise ValueError(f"AI Response is not valid JSON: {e}") from e
                
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise RuntimeError(f"Failed to generated test data: {e}") from e
            

    def _build_prompt(
            self,
            context: str,
            count: int,
            **kwargs
    ) -> str:
        """
        Build prompt for AI based on context.

        TOOD: Move to prompts.py with proper templates

        Args:
            context: Data context type
            count: Number of records
            **kwargs: Additional parameters

        Returns:
            Formatted prompt string
        """

        if context == "ecommerce_customer":
            return f"""
Generate {count} realistic e-commerce customer profiles.

For each customer, include:
- Full name (culturally diverse, realistic)
- Email address (realistic format, NOT test@test.com)
- Age (18-75, varied dstribution)
- Location (city, country)
- Shopping behavior:
    - Frequency (daily, weekly, monthly, occasional)
    - Average order value (realistic for region)
    - Preferred product categories (2-3 categories)
    - Preferred device (mobile, desktop, tablet)
    - Payment method preference

Make the data realistic and diverse:
- Mix of ages, locations, behaviours
- Names form various cultures
- Email formats look real (firstname.lastname@provider.com style)
- Shopping patterns match demographics (students vs professionals vs seniors)

Return ONLY a JSON object with this structure:
{{
    "data": [
        {{
            "name": "...",
            "email": "...",
            "age": "...",
            "location": {{"city": "...", "country": "..."}},
            "shopping_behavior": {{
                "frequency": "...",
                "avg_order_value": "..."
                "preferred_categories": "...",
                "device": "...",
                "payment_method": "..."
            }}
        }},
        ... more customers
    ]
}}

DO NOT include any markdown formatting or code blocks. Return only the raw JSON.
"""
        else:
            # Generic fallback
            return f"""
Generate {count} realistic test data records for: {context}

Make them diverse and realistic.
Return as JSON: {{"data": [...]}}

DO NOT include any markdown formatting or code blocks. Return only the raw JSON.
"""


if __name__ == "__main__":
    # Test with default provider from .env
    gen = TestDataGenerator()

    print("🤖 Using {gen.config.provider} provider")
    print("   Model: {gen.config.model}")
    print("\nGenerating 3 ecommerce customers...")
    
    customers = gen.generate("ecommerce_customer", count=3)

    print(f"\n✅ Generated {len(customers)} customers!")
    print("\nFirst customer:")
    print(json.dumps(customers[0], indent=2))


# class Generator:
#     """Main test data generator."""
    
#     def __init__(self, api_key=None, provider="openai"):
#         """Initialize with API credentials."""
        
#     def generate(self, context, count=10, persona=None, **kwargs):
#         """Generate test data for given context."""
        
#     def generate_with_relationships(self, primary_context, 
#                                     related_contexts, count=10):
#         """Generate related data (e.g., users + their orders)."""
        
#     def validate_data(self, data, context):
#         """Validate generated data against schema."""
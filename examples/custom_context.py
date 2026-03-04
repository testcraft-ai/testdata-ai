"""
Custom context definitions — two approaches.

Run this file directly to see both approaches in action (no real AI call is made;
set AI_PROVIDER / API keys in .env to actually generate data).
"""

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Approach 1: Programmatic — register a ContextSchema in Python
# ---------------------------------------------------------------------------
from testdata_ai import DataGenerator, ContextSchema, register_context

register_context(
    "game_npc",
    ContextSchema(
        description="RPG non-player character profiles",
        category="gaming",
        sample={
            "npc_id": "NPC-0011",
            "name": "Mira Dawnwhisper",
            "role": "innkeeper",
            "location": "Thornhaven",
            "disposition": "friendly",
            "gold": 80,
        },
        prompt_hints=[
            "Fantasy names from diverse real-world cultures",
            "Roles: innkeeper, blacksmith, guard, merchant, quest-giver",
            "Dispositions: friendly, neutral, hostile, fearful",
            "Gold: 10-500 depending on role",
        ],
    ),
)

print("Registered 'game_npc' context via Python API (ContextSchema).")

# Approach 1b: Programmatic — plain dict (no ContextSchema import needed)
register_context("game_item", {
    "description": "RPG inventory items",
    "category": "gaming",
    "sample": {
        "item_id": "ITM-0099",
        "name": "Elven Cloak",
        "type": "armor",
        "rarity": "rare",
        "value_gold": 250,
    },
    "prompt_hints": [
        "Fantasy item names (weapons, armor, potions, scrolls)",
        "Types: weapon, armor, accessory, consumable, quest",
        "Rarities: common, uncommon, rare, epic, legendary",
        "Value should match rarity (common < 50g, legendary > 5000g)",
    ],
})

print("Registered 'game_item' context via plain dict.")

# Generate data  (requires a configured AI provider in .env)
# gen = DataGenerator()
# npcs = gen.generate("game_npc", count=5)
# print(json.dumps(npcs, indent=2))


# ---------------------------------------------------------------------------
# Approach 2: File-based — load contexts from a YAML (or JSON) file
# ---------------------------------------------------------------------------
from testdata_ai import load_contexts_from_file

yaml_file = Path(__file__).parent / "game_characters.yaml"
registered = load_contexts_from_file(yaml_file)
print(f"Loaded context(s) from file: {registered}")

# The context is now available globally — generate with it:
# gen = DataGenerator()
# characters = gen.generate("game_character", count=5)
# print(json.dumps(characters, indent=2))


# ---------------------------------------------------------------------------
# CLI equivalents:
#
#   testdata-ai generate \
#       --context game_character \
#       --context-file examples/game_characters.yaml \
#       --count 5
#
#   testdata-ai list-contexts --context-file examples/game_characters.yaml
#   testdata-ai show-context game_character --context-file examples/game_characters.yaml
# ---------------------------------------------------------------------------
print("\nCLI usage:")
print("  testdata-ai generate --context game_character --context-file examples/game_characters.yaml --count 5")
print("  testdata-ai generate --context game_npc --count 5  # programmatic context, no file needed")
print("  testdata-ai list-contexts --context-file examples/game_characters.yaml")
print("  testdata-ai show-context game_character --context-file examples/game_characters.yaml")

"""
Mem0 OpenSource implementation does not have categorization features. I built a simple one.
"""

import logging
import os

from dotenv import load_dotenv
from openrouter import OpenRouter

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class Category:
    name: str
    description: str

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def __str__(self):
        return f"NAME: {self.name} | DESCRIPTION: {self.description}"


AVAILABLE_CATEGORIES = [
    Category(
        name="DB",
        description="""MySQL-specific insights. Includes indexing strategies, 
        complex query optimizations, migration status, and database-level constraints.""",
    ),
    Category(
        name="AUTH",
        description="""Centralize knowledge regarding the dedicated Auth Service. 
        Record details about JWT structures, specific Role-Based Access Control (RBAC) 
        permissions, and how other services should validate users and permissions.""",
    ),
    Category(
        name="DEPLOYMENT",
        description="""Deployment is manual on Linux VPS. 
        Record specific manual steps required to ship a specific service.""",
    ),
    Category(
        name="SCHEMAS",
        description="""Focus on FastAPI/Pydantic definitions. 
        Record facts about Pydantic model validation rules, prefered declarations 
        of fields, CamelModel and BaseModel use.""",
    ),
    Category(
        name="BUSINESS_LOGIC",
        description="""
        Capture the 'Domain Knowledge'—why a calculation is done a certain way, edge cases 
        in the code, or legacy logic that isn't immediately obvious from reading the code.""",
    ),
    Category(
        name="CROSS_SERVICE",
        description="""Document the 'connective tissue' between services. 
        This includes internal API endpoints, timeout settings between services, and 
        how Service A depends on Service B's state.""",
    ),
]

SYSTEM_PROMPT = f"""You are a categorization engine. Your task is to categorize 
technical facts into the following categories: {", ".join([str(category) for category in AVAILABLE_CATEGORIES])}. 
When given a technical fact, determine which category it belongs. 
If a fact does not fit into any of the categories, respond with 'UNCATEGORIZED'.
ALWAYS respond with ONLY the category name or 'UNCATEGORIZED'. 
DO NOT provide any explanations or additional text."""


def categorize_memories(fact: str) -> str:
    with OpenRouter(api_key=OPENROUTER_API_KEY) as client:
        logging.info("Categorizing...")
        response = client.chat.send(
            model="nvidia/nemotron-3-super-120b-a12b:free",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": fact},
            ],
        )
        result = response.choices[0].message.content.strip()
        logging.info(f"Categorization result: {result}")
        # Make sure the result is one of the valid categories or "UNCATEGORIZED"
        if result not in [category.name for category in AVAILABLE_CATEGORIES] + ["UNCATEGORIZED"]:
            return "UNCATEGORIZED"
        return result

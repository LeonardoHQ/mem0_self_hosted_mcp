import logging
import os
from datetime import datetime
from typing import Any, Dict, Literal, Optional

from dotenv import load_dotenv
from fastmcp import FastMCP
from mem0 import Memory
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
load_dotenv()

# This is the backend team's shared memory instance so it's hardcoded.
SHARED_USER_ID = "backend_team_shared"
AGENT_ID = "mcp_engineering_companion"


POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "postgres")
POSTGRES_PORT = os.environ.get("POSTGRES_PORT", "5432")
POSTGRES_DB = os.environ.get("POSTGRES_DB", "postgres")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "postgres")
POSTGRES_COLLECTION_NAME = os.environ.get("POSTGRES_COLLECTION_NAME", "memories")

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "mem0graph")

MEMGRAPH_URI = os.environ.get("MEMGRAPH_URI", "bolt://localhost:7687")
MEMGRAPH_USERNAME = os.environ.get("MEMGRAPH_USERNAME", "memgraph")
MEMGRAPH_PASSWORD = os.environ.get("MEMGRAPH_PASSWORD", "mem0graph")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_LLM_MODEL = os.environ.get("OPENROUTER_LLM_MODEL", "minimax/minimax-m2.5:free")
OPENROUTER_EMBEDDING_MODEL = os.environ.get("OPENROUTER_EMBEDDING_MODEL", "perplexity/pplx-embed-v1-0.6b")

HISTORY_DB_PATH = os.environ.get("HISTORY_DB_PATH", "/app/history/history.db")

DEFAULT_CONFIG = {
    "version": "v1.1",
    "custom_instructions": """
        You are a senior backend architect for a team of 4. Extract technical facts using the following strict taxonomy:

        1. 'db': Focus on MySQL-specific insights. This includes indexing strategies, complex query optimizations, migration status, and database-level constraints.
        2. 'auth': Centralize knowledge regarding the dedicated Auth Service. Record details about JWT structures, specific Role-Based Access Control (RBAC) permissions, and how other services should validate users.
        3. 'deployment': Since deployment is manual on Linux VPS, record specific environment variables, systemd service configurations, Nginx proxy rules, and manual steps required to ship a specific service.
        4. 'schemas': Focus on FastAPI/Pydantic definitions. Record facts about Pydantic model validation rules, expected input/output JSON structures, and breaking changes in API contracts.
        5. 'business-logic': Capture the 'Domain Knowledge'—why a calculation is done a certain way, edge cases in the code, or legacy logic that isn't immediately obvious from reading the code.
        6. 'cross-service': Document the 'connective tissue' between services. This includes internal API endpoints, timeout settings between services, and how Service A depends on Service B's state.

        EXTRACTION GUIDELINES:
        - If a developer explains a fix for a manual deployment error, tag it as 'deployment'.
        - If a developer mentions a specific Pydantic field constraint (e.g., "this must be a positive int"), tag it as 'schemas'.
        - Ignore all non-technical chatter. If no facts fit these categories, do not create a memory.

        DEDUPLICATION RULES:
        - If a fact is already present in the conversation history as something previously 'learned', DO NOT extract it again.
        - Only extract 'New' information or 'Corrections' to previous facts.
        - If the user is just confirming what you just said, ignore it.
    """,
    "vector_store": {
        "provider": "pgvector",
        "config": {
            "host": POSTGRES_HOST,
            "port": int(POSTGRES_PORT),
            "dbname": POSTGRES_DB,
            "user": POSTGRES_USER,
            "password": POSTGRES_PASSWORD,
            "collection_name": POSTGRES_COLLECTION_NAME,
        },
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {"url": NEO4J_URI, "username": NEO4J_USERNAME, "password": NEO4J_PASSWORD},
    },
    "llm": {
        "provider": "openai",
        "config": {
            "api_key": OPENROUTER_API_KEY,
            "model": OPENROUTER_LLM_MODEL,
            "openai_base_url": "https://openrouter.ai/api/v1",
        },
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "api_key": OPENROUTER_API_KEY,
            "model": OPENROUTER_EMBEDDING_MODEL,
            "openai_base_url": "https://openrouter.ai/api/v1",
        },
    },
    "history_db_path": HISTORY_DB_PATH,
}

# Initialize Mem0
MEMORY_INSTANCE = Memory.from_config(DEFAULT_CONFIG)

# Initialize FastMCP Server
mcp = FastMCP("Mem0 Server")


class Message(BaseModel):
    role: str = Field(..., description="Role of the message (user or assistant).")
    content: str = Field(..., description="Message content.")


@mcp.tool()
def add_memory(
    messages: list[Message],
    categories: list[
        Literal["db", "auth", "deployment", "schemas", "endpoint-output", "business-logic", "cross-service"]
    ] = Field(..., description="Topic category for the memory."),
) -> Any:
    """
    Store technical insights, architectural decisions, and 'gotchas'.
    Use this when a non-obvious fact about the codebase is discovered.
    """

    # We do not store unnecesary things.
    custom_extraction_prompt = f"""
    Extract only technical backend facts.
    Assigned Categories for this entry: {categories}
    If the conversation is just 'thanks' or 'ok', extract NOTHING.
    """

    params = {
        "user_id": SHARED_USER_ID,
        "agent_id": AGENT_ID,
        "metadata": {"categories": categories, "repo": "detect_from_context", "created_at": datetime.now().isoformat()},
        "infer": True,
    }
    params["prompt"] = custom_extraction_prompt

    try:
        dict_messages = [m.model_dump() for m in messages]
        return MEMORY_INSTANCE.add(messages=dict_messages, **params)
    except Exception as e:
        logging.exception("Error in add_memory:")
        raise ValueError(f"Failed to add memory: {str(e)}")


@mcp.tool()
def search_memory(
    query: str = Field(
        ..., description="The natural language search query (e.g., 'MySQL version' or 'how to deploy')."
    ),
    categories: Optional[
        list[Literal["db", "auth", "deployment", "schemas", "endpoint-output", "business-logic", "cross-service"]]
    ] = Field(None, description="Optional categories to narrow down the search."),
    limit: int = Field(5, description="Number of memories to return. Defaults to 5."),
) -> str:
    """
    Search the backend team's shared memory.
    Use this to find existing 'gotchas', architectural decisions, or deployment steps.
    """

    # Matches the hardcoded ID from your add_memory tool
    SHARED_USER_ID = "backend_team_shared"

    # Build filters based on metadata
    search_filters: dict[str, Any] = {"user_id": SHARED_USER_ID}

    # If the agent provides categories, we apply them to the metadata filter
    if categories:
        # Note: In Mem0 OSS, this matches the 'categories' key we put in metadata
        search_filters["metadata.categories"] = {"any": categories}

    try:
        results = MEMORY_INSTANCE.search(query=query, filters=search_filters, limit=limit)

        if not results or not results.get("results"):
            return f"No relevant memories found for: '{query}'"

        # Format the output so the Agent can read it easily
        formatted_results = []
        for res in results["results"]:
            mem_text = res["memory"]
            mem_id = res["id"]
            # Extract metadata for context (who added it, etc.)
            meta = res.get("metadata", {})
            cat_list = meta.get("categories", ["uncategorized"])
            dev = meta.get("added_by", "Unknown Dev")

            formatted_results.append(f"- [{', '.join(cat_list)}] {mem_text} (ID: {mem_id}, Added by: {dev})")

        return "\n".join(formatted_results)

    except Exception as e:
        logging.exception("Error in search_memory:")
        return f"Error performing search: {str(e)}"


@mcp.tool()
def update_memory(
    memory_id: str = Field(..., description="The unique ID of the memory to update."),
    text: str = Field(..., description="The new, updated technical fact or insight."),
    categories: Optional[
        list[Literal["db", "auth", "deployment", "schemas", "endpoint-output", "business-logic", "cross-service"]]
    ] = Field(None, description="Updated categories for this memory if they have changed."),
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Update an existing technical memory with new information.
    Use this when a 'gotcha' changes, a VPS config is updated, or a Pydantic schema is modified.
    """

    # Pre-build the update payload
    update_data: dict[str, Any] = {"data": text}

    # Maintain metadata consistency
    final_metadata = metadata or {}
    if categories:
        final_metadata["categories"] = categories

    # Add a 'last_updated' flag so devs know the info is fresh
    final_metadata["last_updated_at"] = datetime.now().isoformat()

    if final_metadata:
        update_data["metadata"] = final_metadata

    try:
        # Mem0 OSS update method
        MEMORY_INSTANCE.update(memory_id=memory_id, **update_data)
        return f"Successfully updated memory {memory_id}."

    except Exception as e:
        logging.exception(f"Error updating memory {memory_id}:")
        return f"Failed to update memory: {str(e)}"


# @mcp.tool()
# def delete_all_memories(
#     user_id: str,
#     agent_id: str,
#     run_id: Optional[str] = None,
# ) -> str:
#     """Delete all memories for a given identifier."""
#     logging.debug("trying to delete all memories with params: %s", {
#         "user_id": user_id,
#         "agent_id": agent_id,
#         "run_id": run_id
#     })
#     try:
#         params = {k: v for k, v in {"user_id": user_id, "run_id": run_id, "agent_id": agent_id}.items() if v is not None}
#         MEMORY_INSTANCE.delete_all(**params)
#         return "All relevant memories deleted"
#     except Exception as e:
#         logging.exception("Error in delete_all_memories:")
#         raise ValueError(f"Failed to delete all memories: {str(e)}")

# @mcp.tool()
# def reset_memory() -> str:
#     """Completely reset stored memories."""
#     logging.debug("trying to reset all memories")
#     try:
#         MEMORY_INSTANCE.reset()
#         return "All memories reset"
#     except Exception as e:
#         logging.exception("Error in reset_memory:")
#         raise ValueError(f"Failed to reset memory: {str(e)}")

if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)

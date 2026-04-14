import logging
import os
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from mem0 import Memory
from fastmcp import FastMCP

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
load_dotenv()

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
    "llm": {"provider": "openai", "config": {"api_key": OPENROUTER_API_KEY, "model": OPENROUTER_LLM_MODEL, "openai_base_url": "https://openrouter.ai/api/v1"}},
    "embedder": {"provider": "openai", "config": {"api_key": OPENROUTER_API_KEY, "model": OPENROUTER_EMBEDDING_MODEL,  "openai_base_url": "https://openrouter.ai/api/v1"}},
    "history_db_path": HISTORY_DB_PATH,
}

# Initialize Mem0
MEMORY_INSTANCE = Memory.from_config(DEFAULT_CONFIG)

# Initialize FastMCP Server
mcp = FastMCP("Mem0 Server")

class Message(BaseModel):
    role: str = Field(..., description="Role of the message (user or assistant).")
    content: str = Field(..., description="Message content.")

# @mcp.tool()
# def configure_mem0(config: Dict[str, Any]) -> str:
#     """Set memory configuration."""
#     global MEMORY_INSTANCE
#     MEMORY_INSTANCE = Memory.from_config(config)
#     return "Configuration set successfully"

@mcp.tool()
def add_memory(
    messages: List[Message],
    user_id: str,
    agent_id: str,
    run_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    infer: Optional[bool] = Field(True, description="Whether to extract facts from messages. Defaults to True."),
    memory_type: Optional[str] = Field(None, description="Type of memory to store (e.g. 'core')."),
    prompt: Optional[str] = Field(None, description="Custom prompt to use for fact extraction.")
) -> Any:
    """Store new memories."""
    logging.debug("trying to add memory with params: %s", {
        "user_id": user_id,
        "agent_id": agent_id,
        "run_id": run_id,
        "metadata": metadata,
        "infer": infer,
        "memory_type": memory_type,
        "prompt": prompt,
        "messages": [m.model_dump() for m in messages]    
    })

    if run_id is None:
        run_id = f"run_{user_id}_{agent_id}_{int(time.time())}"

    params = {
        "user_id": user_id,
        "agent_id": agent_id,
        "run_id": run_id,
        "metadata": metadata,
        "infer": infer,
        "memory_type": memory_type,
        "prompt": prompt
    }
    # Remove None values
    params = {k: v for k, v in params.items() if v is not None}
    
    try:
        dict_messages = [m.model_dump() for m in messages]
        return MEMORY_INSTANCE.add(messages=dict_messages, **params)
    except Exception as e:
        logging.exception("Error in add_memory:")
        raise ValueError(f"Failed to add memory: {str(e)}")

@mcp.tool()
def get_all_memories(
    user_id: str,
    agent_id: str,
    run_id: Optional[str] = None,
) -> Any:
    """Retrieve stored memories."""
    if run_id is None:
        run_id = f"run_{user_id}_{agent_id}_{int(time.time())}"

    logging.debug("trying to get all memories with params: %s", {
        "user_id": user_id,
        "agent_id": agent_id,
        "run_id": run_id
    })
    
    try:
        params = {k: v for k, v in {"user_id": user_id, "run_id": run_id, "agent_id": agent_id}.items() if v is not None}
        return MEMORY_INSTANCE.get_all(**params)
    except Exception as e:
        logging.exception("Error in get_all_memories:")
        raise ValueError(f"Failed to get memories: {str(e)}")

@mcp.tool()
def get_memory(memory_id: str) -> Any:
    """Retrieve a specific memory by ID."""
    try:
        return MEMORY_INSTANCE.get(memory_id)
    except Exception as e:
        logging.exception("Error in get_memory:")
        raise ValueError(f"Failed to get memory: {str(e)}")

@mcp.tool()
def search_memories(
    query: str,
    user_id: str,
    agent_id: str,
    run_id: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
    top_k: Optional[int] = Field(None, description="Maximum number of results to return."),
    threshold: Optional[float] = Field(None, description="Minimum similarity score for results.")
) -> Any:
    """Search for memories based on a query."""
    params = {
        "user_id": user_id,
        "run_id": run_id,
        "agent_id": agent_id,
        "filters": filters,
        "top_k": top_k,
        "threshold": threshold
    }

    logging.debug("trying to search memories with params: %s", params)

    params = {k: v for k, v in params.items() if v is not None}
    
    try:
        return MEMORY_INSTANCE.search(query=query, **params)
    except Exception as e:
        logging.exception("Error in search_memories:")
        raise ValueError(f"Failed to search memories: {str(e)}")

@mcp.tool()
def update_memory(
    memory_id: str, 
    text: str = Field(..., description="New content to update the memory with."),
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata to update.")
) -> Any:
    """Update an existing memory with new content."""

    logging.debug("trying to update memory with params: %s", {
        "memory_id": memory_id,
        "text": text,
        "metadata": metadata
    })
    try:
        return MEMORY_INSTANCE.update(memory_id=memory_id, data=text, metadata=metadata)
    except Exception as e:
        logging.exception("Error in update_memory:")
        raise ValueError(f"Failed to update memory: {str(e)}")

@mcp.tool()
def get_memory_history(memory_id: str) -> Any:
    """Retrieve memory history."""
    logging.debug("trying to get memory history with memory_id: %s", memory_id)
    try:
        return MEMORY_INSTANCE.history(memory_id=memory_id)
    except Exception as e:
        logging.exception("Error in memory_history:")
        raise ValueError(f"Failed to get memory history: {str(e)}")

# @mcp.tool()
# def delete_memory(memory_id: str) -> str:
#     """Delete a specific memory by ID."""
#     logging.debug("trying to delete memory with memory_id: %s", memory_id)
#     try:
#         MEMORY_INSTANCE.delete(memory_id=memory_id)
#         return "Memory deleted successfully"
#     except Exception as e:
#         logging.exception("Error in delete_memory:")
#         raise ValueError(f"Failed to delete memory: {str(e)}")

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
    # Run the server over Streamable HTTP transport
    # You can customize the host, port, and path as needed
    mcp.run(
        transport="http", 
        host="0.0.0.0", 
        port=8000
    )
# Mem0 Self-Hosted MCP Server

This project is a self-hosted implementation of the Mem0 REST API server, based on the [mem0ai/mem0](https://github.com/mem0ai/mem0) repository, specifically the server folder.

## What This Project Is

This is a modified version of the Mem0 server that:
- Works with OpenRoute instead of the original embedding/LLM providers
- Uses updated library versions compared to the original
- Provides the same REST API functionality for memory management

## Features

- **Create memories:** Create memories based on messages for a user, agent, or run.
- **Retrieve memories:** Get all memories for a given user, agent, or run.
- **Search memories:** Search stored memories based on a query.
- **Update memories:** Update an existing memory.
- **Delete memories:** Delete a specific memory or all memories for a user, agent, or run.
- **Reset memories:** Reset all memories for a user, agent, or run.
- **OpenAPI Documentation:** Accessible via `/docs` endpoint.

## Starting the Server

Do it using the `docker-compose.yml` file by executing:

```bash
docker compose up -d
```

Make sure you don't have other services running on the same ports declared on the `docker-compose.yml` file.

## Key Differences from Original mem0ai/mem0 Server

1. **OpenRoute Integration:** This implementation is configured to work with OpenRoute for embeddings and LLM operations instead of the default providers used in the original.
2. **Updated Libraries:** All Python dependencies have been updated to their latest compatible versions.
3. **Self-Hosted Focus:** Optimized for self-hosted deployment with MCP (Model Context Protocol) support.

## Available Entry Points

- `fastapi_main.py` - Standard FastAPI server implementation
- `fastmcp_main.py` - FastAPI server with MCP integration

Both servers provide the same Mem0 REST API endpoints.
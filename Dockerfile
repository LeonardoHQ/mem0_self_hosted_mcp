FROM cicirello/pyaction:3.13
ENV PYTHONUNBUFFERED=1

# installs uv
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:0.11.3 /uv /uvx /bin/

COPY pyproject.toml uv.lock /app/

# allows to install system-wide packages, evading .venv
ENV UV_PROJECT_ENVIRONMENT=/usr/local

COPY . .

RUN uv sync
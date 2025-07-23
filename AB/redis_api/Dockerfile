FROM python:3.11-slim

WORKDIR /app/redis_api

# Install uv and sync from redis_api folder
COPY pyproject.toml uv.lock ./

RUN pip install uv && uv sync

# Copy in actual code
COPY . .

CMD ["uv", "run", "uvicorn", "redis_api:app", "--host", "0.0.0.0", "--port", "8000"]
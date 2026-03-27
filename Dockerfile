FROM python:3.12-slim

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy project files needed for dependency resolution
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

# Install the package and all dependencies
RUN uv sync --frozen --no-dev

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "sentiment.main:app", "--host", "0.0.0.0", "--port", "8000"]

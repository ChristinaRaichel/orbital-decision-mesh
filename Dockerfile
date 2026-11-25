
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project
COPY . .

# Install Python package and dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Prometheus metrics and FastMCP HTTP
EXPOSE 8001 8000 8080

# Start MCP server
CMD ["python", "-u", "-m", "src.mcp.server"]

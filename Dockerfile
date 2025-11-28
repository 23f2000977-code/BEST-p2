FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml .

# Install Python dependencies with uv (MUCH faster than pip)
RUN uv pip install --system -r pyproject.toml

# Install Playwright and its dependencies
RUN playwright install --with-deps chromium

# Copy application files
COPY . .

# Create directory for file operations
RUN mkdir -p /app/hybrid_llm_files

# Expose port
EXPOSE 7860

# Run the application
CMD ["python", "-m", "uvicorn", "hybrid_main:app", "--host", "0.0.0.0", "--port", "7860"]

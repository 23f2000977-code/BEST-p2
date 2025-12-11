FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

# Install system dependencies
# ADDED: ffmpeg (Critical for Audio Processing)
RUN apt-get update && apt-get install -y \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy ALL files first (needed for uv to read pyproject.toml properly)
COPY . .

# Pre-install heavy libraries to avoid runtime timeouts
RUN uv pip install --system --no-cache pandas numpy scipy playwright beautifulsoup4 requests scikit-learn pillow python-dotenv fastapi uvicorn

# Install remaining Python dependencies from pyproject.toml
RUN uv pip install --system --no-cache .

# Install Playwright and its dependencies
RUN playwright install --with-deps chromium

# Create directory for file operations AND fix permissions for HF Spaces (user 1000)
RUN mkdir -p /app/hybrid_llm_files && chmod 777 /app/hybrid_llm_files

# Expose port (HF Spaces uses 7860)
EXPOSE 7860

# Run the application
CMD ["uvicorn", "hybrid_main:app", "--host", "0.0.0.0", "--port", "7860"]
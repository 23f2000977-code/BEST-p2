FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

# Install system dependencies
# CRITICAL: Added ffmpeg for audio processing
RUN apt-get update && apt-get install -y \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy ALL files first
COPY . .

# Pre-install heavy libraries
RUN uv pip install --system --no-cache pandas numpy scipy playwright beautifulsoup4 requests scikit-learn pillow python-dotenv fastapi uvicorn

# Install remaining Python dependencies
RUN uv pip install --system --no-cache .

# Install Playwright
RUN playwright install --with-deps chromium

# Create directory for file operations
RUN mkdir -p /app/hybrid_llm_files && chmod 777 /app/hybrid_llm_files

# Expose port
EXPOSE 7860

# Run the application
CMD ["uvicorn", "hybrid_main:app", "--host", "0.0.0.0", "--port", "7860"]
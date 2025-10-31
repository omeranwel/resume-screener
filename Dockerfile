# ---------- Stage 1: builder (installs build tools & wheels) ----------
FROM python:3.11-slim AS builder

# Set environment variables for non-interactive installs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies and libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Only copy requirements first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --upgrade pip && pip wheel --no-cache-dir --no-deps -r requirements.txt -w /wheels

# ---------- Stage 2: final runtime image ----------
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create a non-root user and home dir
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# System runtime tools we want available (curl for HEALTHCHECK)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels from builder and install them (fast & reproducible)
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*

# Copy the actual source code last (so code changes rebuild minimal layers)
COPY . /app

# Switch to non-root user for safer runtime
USER appuser

# Expose FastAPI port
EXPOSE 8000

# Docker will ping the app; if /health fails, container is considered unhealthy
HEALTHCHECK --interval=30s --timeout=5s --retries=5 \
  CMD curl --fail http://localhost:8000/health || exit 1

# Run the app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

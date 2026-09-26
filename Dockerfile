# ==============================================================================
# K.I.S.A.N. AI - Multi-Agent Agricultural Advisory Platform
# Production-ready, containerized Docker configuration
# ==============================================================================

FROM python:3.11-slim

# Set container working directory
WORKDIR /app

# Ensure Python outputs stdout/stderr unbuffered and skips bytecode generation
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install minimal essential runtime packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for optimal Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application configuration, models, data, and source code
COPY config/ /app/config/
COPY models/ /app/models/
COPY data/ /app/data/
COPY src/ /app/src/
COPY app.py /app/

# Expose web service port
EXPOSE 5000

# Container healthcheck to verify web dashboard is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

# Default command to launch the web dashboard
CMD ["python", "app.py"]

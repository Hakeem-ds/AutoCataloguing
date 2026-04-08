# ── Corporate Archives Auto-Classification App ──
# Multi-stage build: builder installs deps, runtime copies only what's needed

# ── Stage 1: Builder ──
FROM python:3.10-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install Python deps into a clean prefix
COPY src/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Download NLTK data (punkt, stopwords) at build time
RUN python -c "import os; os.makedirs('/nltk_data', exist_ok=True)"
ENV NLTK_DATA=/nltk_data
RUN pip install --no-cache-dir nltk \
    && python -m nltk.downloader -d /nltk_data punkt punkt_tab stopwords

# ── Stage 2: Runtime (no build-essential, ~200MB smaller) ──
FROM python:3.10-slim AS runtime

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local
COPY --from=builder /nltk_data /nltk_data
ENV NLTK_DATA=/nltk_data

# Copy app source (folder_name_map.json included via src/artifacts/)
COPY src/ ./src/

# Streamlit config
RUN mkdir -p /app/.streamlit \
    && echo "[server]\nport = 8000\nheadless = true\nenableCORS = false\nenableXsrfProtection = false\n" \
       > /app/.streamlit/config.toml \
    && chown -R appuser:appuser /app

ENV PYTHONPATH=/app/src
ENV HOME=/app
EXPOSE 8000

USER appuser
WORKDIR /app/src
CMD ["streamlit", "run", "main.py", "--server.port", "8000"]

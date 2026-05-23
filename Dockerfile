# Stage 1: Build python dependencies
FROM python:3.11-slim as builder

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir google-generativeai

# Stage 2: Clean and small runtime container
FROM python:3.11-slim

WORKDIR /app

# Copy pip installation from build stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy app code
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Health check to monitor Streamlit status
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch command
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

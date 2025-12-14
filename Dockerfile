FROM python:3.12-slim

WORKDIR /app

# Install system dependencies (if any needed for pandas/numpy compilation, though wheels usually suffice)
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# Create data directory for the volume
RUN mkdir data

CMD ["python", "-m", "src.main"]

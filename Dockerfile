FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# Install system tools for downloading dataset
RUN apt-get update && apt-get install -y --no-install-recommends curl unzip && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download and extract PaperBananaBench dataset
RUN mkdir -p data && \
    curl -L -o /tmp/PaperBananaBench.zip \
    "https://huggingface.co/datasets/dwzhu/PaperBananaBench/resolve/main/PaperBananaBench.zip" && \
    unzip -q /tmp/PaperBananaBench.zip -d data/ && \
    rm /tmp/PaperBananaBench.zip && \
    chmod -R 777 data

# Copy application code
COPY . .

# Expose port
EXPOSE 8080

# Run FastAPI
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]

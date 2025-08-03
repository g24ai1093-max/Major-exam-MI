# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Copy models directory (will be populated by GitHub Actions artifacts)
COPY models/ ./models/

# Set Python path
ENV PYTHONPATH=/app/src

# Default command to run predictions
CMD ["python", "src/predict.py"] 
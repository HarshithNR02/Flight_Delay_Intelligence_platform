FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY streamlit_app/ ./streamlit_app/
COPY api/ ./api/
COPY download_models.py .
COPY startup.sh .

RUN chmod +x startup.sh

# Create models directory
RUN mkdir -p models

EXPOSE 7860 8000

CMD ["./startup.sh"]

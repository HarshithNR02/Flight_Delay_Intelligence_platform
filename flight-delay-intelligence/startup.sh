#!/bin/bash
set -e

echo "=== Flight Delay Intelligence Platform ==="
echo "Downloading models from Azure Blob Storage..."
python /app/download_models.py

echo "Starting FastAPI..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 &

echo "Starting Streamlit..."
streamlit run streamlit_app/app.py \
  --server.port 7860 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.fileWatcherType none \
  --browser.gatherUsageStats false

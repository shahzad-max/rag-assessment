#!/bin/bash
# Start FastAPI server for EU AI Act RAG System

echo "Starting EU AI Act RAG API..."
echo "================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Please run: python -m venv venv"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Start server
echo "Starting FastAPI server on http://0.0.0.0:8000"
echo "API Documentation: http://localhost:8000/docs"
echo "================================"

cd "$(dirname "$0")/.."
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Made with Bob

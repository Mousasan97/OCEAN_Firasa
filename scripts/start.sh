#!/bin/bash
# Quick start script for OCEAN API

set -e

echo "=========================================="
echo "OCEAN Personality API - Quick Start"
echo "=========================================="

# Check if .env exists
if [ ! -f .env ]; then
    echo "Creating .env from template..."
    cp .env.example .env
    echo "✓ .env created. Please update it with your configuration."
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate || source venv/Scripts/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements/base.txt

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Update .env with your configuration"
echo "2. Ensure MODEL_CHECKPOINT_PATH points to your trained model"
echo "3. Run the API server:"
echo ""
echo "   python -m src.api.main"
echo ""
echo "4. Visit http://localhost:8000/docs for API documentation"
echo ""
echo "=========================================="

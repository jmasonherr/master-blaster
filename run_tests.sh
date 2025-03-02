#!/bin/bash

# Create a virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
echo "Installing test requirements..."
pip install -r test_requirements.txt

# Run tests
echo "Running tests..."
python -m pytest tests/ -v

# Show test coverage report path
echo "Test coverage report generated at htmlcov/index.html"
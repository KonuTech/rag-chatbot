#!/bin/bash
# Run linting checks

echo "Running flake8..."
uv run flake8 backend/ main.py

echo "Checking import order with isort..."
uv run isort --check-only --diff backend/ main.py

echo "Checking code format with black..."
uv run black --check backend/ main.py

echo "Linting complete!"
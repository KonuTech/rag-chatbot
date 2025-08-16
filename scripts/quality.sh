#!/bin/bash
# Run all code quality checks

echo "=== Running Code Quality Checks ==="

echo "1. Running tests..."
cd backend && uv run pytest

echo "2. Running linting..."
bash scripts/lint.sh

echo "=== Quality checks complete ==="
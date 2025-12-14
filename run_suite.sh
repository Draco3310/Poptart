#!/bin/bash

# Poptart Gal Friday V2 - Quality Assurance Pipeline
# Runs Linting, Type Checking, and Unit Tests

set -e  # Exit immediately if a command exits with a non-zero status.

echo "========================================"
echo "🤖 Poptart Gal Friday V2 - Test Suite"
echo "========================================"

echo ""
echo "--- Step 1: Linting & Formatting (Ruff) ---"
ruff check .
ruff format --check .
echo "✅ Linting & Formatting Passed"

echo ""
echo "--- Step 2: Type Checking (Mypy) ---"
mypy src/
echo "✅ Type Checking Passed"

echo ""
echo "--- Step 3: Unit Tests (Pytest) ---"
echo "Running tests for Hybrid Timeframe Logic (5m/1m)..."
python -m pytest tests/ -v
echo "✅ All Tests Passed"

echo ""
echo "--- Step 4: Cleanup ---"
echo "Cleaning up pycache and tool caches..."
find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} +
find . -type f -name "*.pyc" -not -path "./.venv/*" -delete
rm -rf .mypy_cache .pytest_cache .ruff_cache
echo "✅ Cleanup Complete"

echo ""
echo "🎉 SUITE COMPLETE: System is Stable."

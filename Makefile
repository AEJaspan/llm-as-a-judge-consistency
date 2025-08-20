# Makefile for LLM Confidence Experiment
.PHONY: help install format lint test run clean setup-env move-assets full-pipeline

# Default target
help:
	@echo "Available commands:"
	@echo "  setup-env      - Create virtual environment and install dependencies"
	@echo "  install        - Install dependencies"
	@echo "  format         - Format code with ruff"
	@echo "  lint           - Lint code with ruff"
	@echo "  test           - Run tests with pytest"
	@echo "  run            - Run the main experiment"
	@echo "  run-quick      - Run experiment with smaller sample size"
	@echo "  move-assets    - Move generated plots to assets/ directory"
	@echo "  clean          - Clean up generated files"
	@echo "  full-pipeline  - Run complete pipeline: format, lint, test, run, move-assets"

# Python and environment setup
PYTHON := python3
VENV := .venv
PIP := $(VENV)/bin/pip
PYTHON_VENV := $(VENV)/bin/python
PYTEST := $(VENV)/bin/pytest

# Setup virtual environment and install dependencies
setup-env:
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
	@echo "Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -e .
	$(PIP) install ruff pytest pytest-asyncio python-dotenv pyyaml
	@echo "Virtual environment created at $(VENV)"
	@echo "Activate with: source $(VENV)/bin/activate"

# Install dependencies (assumes venv is activated)
install:
	pip install --upgrade pip
	pip install -e .
	pip install ruff pytest pytest-asyncio python-dotenv pyyaml

# Format code with ruff
format:
	@echo "Formatting code with ruff..."
	ruff format src/ tests/
	@echo "Code formatting complete!"

# Lint code with ruff
lint:
	@echo "Linting code with ruff..."
	ruff check src/ tests/ --fix
	@echo "Linting complete!"

# Run tests
test:
	@echo "Running tests with pytest..."
	PYTHONPATH=src pytest tests/ -v --tb=short
	@echo "Tests complete!"

# Run the main experiment (default: quick test)
run:
	@echo "Running LLM confidence experiment..."
	@echo "Note: This requires OPENAI_API_KEY in your .env file"
	PYTHONPATH=src $(PYTHON) src/main.py
	@echo "Experiment complete!"

# Run experiment with quick test configuration
run-quick:
	@echo "Running quick test experiment..."
	PYTHONPATH=src $(PYTHON) src/main.py --config quick_test
	@echo "Quick test complete!"

# Create assets directory and move plots
move-assets:
	@echo "Creating assets directory and moving plots..."
	mkdir -p assets/plots
	mkdir -p assets/data
	# Move plot files
	-mv llm_confidence_experiment_results_*.png assets/plots/ 2>/dev/null || true
	-mv *.png assets/plots/ 2>/dev/null || true
	# Move data files  
	-mv llm_confidence_experiment_results*.csv assets/data/ 2>/dev/null || true
	-mv llm_confidence_analyses*.json assets/data/ 2>/dev/null || true
	-mv *.csv assets/data/ 2>/dev/null || true
	-mv *.json assets/data/ 2>/dev/null || true
	@echo "Assets moved to assets/ directory"
	@ls -la assets/plots/ assets/data/

# Clean up generated files
clean:
	@echo "Cleaning up generated files..."
	rm -f llm_confidence_experiment_results*.csv
	rm -f llm_confidence_analyses*.json
	rm -f llm_confidence_experiment_results*.png
	rm -rf __pycache__/
	rm -rf src/__pycache__/
	rm -rf src/*/__pycache__/
	rm -rf tests/__pycache__/
	rm -rf .pytest_cache/
	rm -rf logs/
	@echo "Cleanup complete!"

# Clean everything including venv
clean-all: clean
	rm -rf $(VENV)
	rm -rf *.egg-info/
	rm -rf build/
	rm -rf dist/

# Development helpers
check-env:
	@echo "Checking environment setup..."
	@if [ ! -f .env ]; then \
		echo "WARNING: .env file not found. Create one with OPENAI_API_KEY=your_key"; \
	else \
		echo "✓ .env file found"; \
	fi
	@if [ ! -d $(VENV) ]; then \
		echo "WARNING: Virtual environment not found. Run 'make setup-env'"; \
	else \
		echo "✓ Virtual environment found"; \
	fi

# List available experiment configurations
list-configs:
	@echo "Available experiment configurations:"
	PYTHONPATH=src $(PYTHON) src/main.py --list

# Run specific experiment configuration(s)
run-config:
	@echo "Usage: make run-config CONFIG=config_name"
	@echo "   or: make run-config CONFIG='config1 config2'"
	@if [ -z "$(CONFIG)" ]; then \
		echo "Error: CONFIG parameter required"; \
		echo "Available configs:"; \
		PYTHONPATH=src $(PYTHON) src/main.py --list; \
	else \
		echo "Running experiment(s): $(CONFIG)"; \
		PYTHONPATH=src $(PYTHON) src/main.py --config $(CONFIG); \
	fi

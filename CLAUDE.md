# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Costly is a Python library for estimating costs and running times of LLM workflows before execution. It uses the `@costly()` decorator on functions that make LLM API calls, enabling simulation mode to predict costs without actually calling the APIs.

## Commands

```bash
# Run tests (excluding slow tests, the default)
poetry run pytest -s

# Run slow tests (require actual API calls)
poetry run pytest -s -m "slow"

# Run a single test file
poetry run pytest -s tests/test_decorator.py

# Run a specific test
poetry run pytest -s tests/test_decorator.py::test_name

# Install dependencies
poetry install
```

Note: Tests require a `.env` file with `OPENAI_API_KEY` for slow tests that make real API calls.

## Architecture

### Core Components

- **`costly/decorator.py`**: The `@costly()` decorator that wraps LLM-calling functions. Handles both sync and async functions. Intercepts `cost_log`, `simulate`, and `description` kwargs to enable cost tracking and simulation.

- **`costly/costlog.py`**: `Costlog` class that accumulates cost data. Supports two modes:
  - `memory`: Stores items in-memory list
  - `jsonl`: Persists to `.jsonl` files with separate totals files

- **`costly/estimators/llm_api_estimation.py`**: `LLM_API_Estimation` class for calculating costs from token counts. Uses litellm's pricing data. Key methods:
  - `get_cost_real()`: Calculate actual costs from real API responses
  - `get_cost_simulating()`: Estimate costs for simulation mode
  - `messages_to_input_tokens()`: Token counting for chat messages

- **`costly/simulators/llm_simulator_faker.py`**: `LLM_Simulator_Faker` class that generates fake LLM responses using Faker. Supports Pydantic models, basic types, and generics.

- **`costly/updatable_default.py`**: Utility decorators for propagating `description` through nested function calls for cost breakdown tracking.

### Usage Pattern

Functions decorated with `@costly()` automatically accept `cost_log`, `simulate`, and `description` kwargs:
- Pass `simulate=True` to get fake responses and estimated costs
- Pass a `Costlog` instance to track costs across multiple calls
- Return `CostlyResponse(output=..., cost_info={...})` to provide exact token counts from API responses

### Parameter Mapping

The decorator supports mapping function parameters to standard names (`model`, `messages`, `input_string`, `input_tokens`) via lambdas or string references:
```python
@costly(messages=lambda kwargs: kwargs["history"], model="model_name")
def my_func(history, model_name): ...
```

# costly
Estimate costs and running times of complex LLM workflows/experiments/pipelines in advance before spending money, via simulations. Just put `@costly()` on the load-bearing function; make sure all functions that call it pass `**kwargs` to it and call your complex function with `simulate=True` and some `cost_log: Costlog` object. See [examples.ipynb](examples.ipynb) for more details.

## Installation

```bash
pip install costly
```

## Usage

See [examples.ipynb](examples.ipynb) for a full walkthrough.

## Testing

```bash
poetry run pytest -s
```
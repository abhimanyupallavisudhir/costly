# costly
Estimate costs of complex LLM workflows in advance before spending money, via simulations. Just put `@costly()` on the load-bearing function (provided it has the right type signature); make sure all functions that call it pass `**kwargs` to it and call your complex function with `simulate=True` and some `cost_log: Costlog` object. See [examples.ipynb](examples.ipynb) for more details.

## Installation

```bash
pip install costly
```

## Usage

See [examples.ipynb](examples.ipynb) for a quick walkthrough.

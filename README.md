# costly
Estimate costs and running times of complex LLM workflows/experiments/pipelines in advance before spending money, via simulations. Just put `@costly()` on the load-bearing function; make sure all functions that call it pass `**kwargs` to it and call your complex function with `simulate=True` and some `cost_log: Costlog` object. See [examples.ipynb](examples.ipynb) for more details.

https://github.com/abhimanyupallavisudhir/costly

## Installation

```bash
pip install costly
```

## Usage

See [examples.ipynb](examples.ipynb) for a full walkthrough; some examples below.

```python

from costly import Costlog, costly, CostlyResponse
from costly.estimators.llm_api_estimation import LLM_API_Estimation as estimator


@costly()
def chatgpt(input_string: str, model: str) -> str:
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": input_string}]
    )
    output_string = response.choices[0].message.content
    return output_string


@costly(
    input_tokens=lambda kwargs: LLM_API_Estimation.messages_to_input_tokens(
        kwargs["messages"], kwargs["model"]
    ),
)
def chatgpt_messages(messages: list[dict[str, str]], model: str) -> str:
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(model=model, messages=messages)
    output_string = response.choices[0].message.content
    return output_string


@costly()
def chatgpt(input_string: str, model: str) -> str:
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": input_string},
        ],
    )

    return CostlyResponse(
        output=response.choices[0].message.content,
        cost_info={
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        },
    ) # in usage, this will still just return the output, not the whole CostlyResponse object

```

## Testing

```bash
poetry run pytest -s
```

## TODO

- [ ] Make it work with async
- [ ] Support for locally run LLMs -- ideally need a cost & time estimator that takes into account your machine details, GPU pricing etc.
- [ ] Decide and document what the best way to "propagate" `description` (for breakdown purposes) through function calls is. Have the user manually write `def f(...): ... g(description = kwargs.get("description") + ["f"]`? Add a `@description("blabla")` decorator? Add a `@description` decorator that automatically appends the function name and arguments into `description`?
- [ ] Better solution for token counting for Chat messages (search `HACK` in the repo)

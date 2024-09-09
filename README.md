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
poetry run pytest -s -m "not slow"
poetry run pytest -s -m "slow"

Tests for instructor currently fail.
```

## TODO

- [x] Make it work with async
- [x] Decide and document what the best way to "propagate" `description` (for breakdown purposes) through function calls is. Have the user manually write `def f(...): ... g(description = kwargs.get("description") + ["f"]`? Add a `@description("blabla")` decorator? Add a `@description` decorator that automatically appends the function name and arguments into `description`?
- [x] Better solution for token counting for Chat messages (search `HACK` in the repo)
- [x] make instructor tests pass https://community.openai.com/t/how-to-calculate-the-tokens-when-using-function-call/266573/11
- [ ] Support for locally run LLMs -- ideally need a cost & time estimator that takes into account your machine details, GPU pricing etc.
- [ ] support more models

Instructor tests don't really pass but I can kinda live with this. Lmk if anyone has a good way to count tokens from messages that includes tool calling (I'm using [this](https://community.openai.com/t/how-to-calculate-the-tokens-when-using-function-call/266573/11)).
```
FAILED tests/test_estimators/test_llm_api_estimation.py::test_estimate_contains_exact_instructor[PERSONINFO_gpt-4o] - AssertionError: ['Input tokens estimate 65 not within 20pc of truth 83']
FAILED tests/test_estimators/test_llm_api_estimation.py::test_estimate_contains_exact_instructor[PERSONINFO_gpt-4o-mini] - AssertionError: ['Input tokens estimate 65 not within 20pc of truth 83']
FAILED tests/test_estimators/test_llm_api_estimation.py::test_estimate_contains_exact_instructor[PERSONINFO_gpt-4-turbo] - AssertionError: ['Input tokens estimate 65 not within 20pc of truth 85']
FAILED tests/test_estimators/test_llm_api_estimation.py::test_estimate_contains_exact_instructor[PERSONINFO_gpt-3.5-turbo] - AssertionError: ['Input tokens estimate 65 not within 20pc of truth 85']
FAILED tests/test_estimators/test_llm_api_estimation.py::test_estimate_contains_exact_instructor[FOOMODEL_gpt-4o] - AssertionError: ['Input tokens estimate 77 not within 20pc of truth 108']
FAILED tests/test_estimators/test_llm_api_estimation.py::test_estimate_contains_exact_instructor[FOOMODEL_gpt-4o-mini] - AssertionError: ['Input tokens estimate 77 not within 20pc of truth 108']
FAILED tests/test_estimators/test_llm_api_estimation.py::test_estimate_contains_exact_instructor[FOOMODEL_gpt-4-turbo] - AssertionError: ['Input tokens estimate 77 not within 20pc of truth 113']
FAILED tests/test_estimators/test_llm_api_estimation.py::test_estimate_contains_exact_instructor[FOOMODEL_gpt-3.5-turbo] - AssertionError: ['Input tokens estimate 77 not within 20pc of truth 113']
FAILED tests/test_estimators/test_llm_api_estimation.py::test_estimate_contains_exact_instructor[BARMODEL_gpt-4o] - AssertionError: ['Input tokens estimate 70 not within 20pc of truth 168']
FAILED tests/test_estimators/test_llm_api_estimation.py::test_estimate_contains_exact_instructor[BARMODEL_gpt-4o-mini] - AssertionError: ['Input tokens estimate 70 not within 20pc of truth 168']
FAILED tests/test_estimators/test_llm_api_estimation.py::test_estimate_contains_exact_instructor[BARMODEL_gpt-4-turbo] - AssertionError: ['Input tokens estimate 70 not within 20pc of truth 178']
FAILED tests/test_estimators/test_llm_api_estimation.py::test_estimate_contains_exact_instructor[BARMODEL_gpt-4] - AssertionError: ['Input tokens estimate 70 not within 20pc of truth 126']
FAILED tests/test_estimators/test_llm_api_estimation.py::test_estimate_contains_exact_instructor[BARMODEL_gpt-3.5-turbo] - AssertionError: ['Input tokens estimate 70 not within 20pc of truth 178']
```

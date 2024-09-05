"""
Tests to make sure the estimates for simulation actually match the actual cost.
"""

import pytest
from openai import OpenAI
import instructor
from pydantic import BaseModel
from costly.costlog import Costlog
from costly.estimators.llm_api_estimation import LLM_API_Estimation
from tests.example_functions import (
    chatgpt,
    chatgpt_prompt,
    chatgpt_instructor,
    CLIENT,
    PERSONINFO,
    FOOMODEL,
    BARMODEL,
)

PROMPTS = [
    "Explain the Riemann Hypothesis in full mathematical detail.",
    "Write a short story.",
    "Write the bubblesort algorithm in Python.",
]
MESSAGESS = [
    [{"content": prompt, "role": "user"}] for prompt in PROMPTS
]
MODELS = LLM_API_Estimation.PRICES.keys() # TODO implement tests for non-OpenAI models
MODELS_OPENAI = [model for model in MODELS if model.startswith("gpt")]
PARAMSS = [
    {"messages": messages, "model": model} for model in MODELS_OPENAI for messages in MESSAGESS
]
PARAMSS_PROMPT = [
    {"prompt": prompt, "model": model} for model in MODELS_OPENAI for prompt in PROMPTS
]
PROMPTS_INSTRUCTOR = [
    "Give me an example of a PERSONINFO object.",
    "Give me an example of a FOOMODEL object.",
    "Give me an example of a BARMODEL object.",
]
MESSAGESS_INSTRUCTOR = [
    [{"content": prompt, "role": "user"}] for prompt in PROMPTS_INSTRUCTOR
]
PARAMSS_INSTRUCTOR_ = [
    {"messages": MESSAGESS_INSTRUCTOR[0], "response_model": PERSONINFO},
    {"messages": MESSAGESS_INSTRUCTOR[1], "response_model": FOOMODEL},
    {"messages": MESSAGESS_INSTRUCTOR[2], "response_model": BARMODEL},
]
PARAMSS_INSTRUCTOR = [
    {**params, "model": model, "client": instructor.from_openai(OpenAI())}
    for params in PARAMSS_INSTRUCTOR_
    for model in MODELS_OPENAI
]


def check_cost_estimates(real_cost, sim_cost):
    print("real_cost", real_cost)
    print("sim_cost", sim_cost)
    assertions = [
        {
            "check": 0.8 * real_cost["input_tokens"]
            <= sim_cost["input_tokens"]
            <= 1.2 * real_cost["input_tokens"],
            "message": f"Input tokens estimate {sim_cost['input_tokens']} not within 20pc of truth {real_cost['input_tokens']}",
        },
        {
            "check": sim_cost["output_tokens_min"] <= real_cost["output_tokens_min"],
            "message": f"Output tokens estimate minimum {sim_cost['output_tokens_min']} exceeds truth {real_cost['output_tokens_min']}",
        },
        {
            "check": real_cost["output_tokens_max"] <= sim_cost["output_tokens_max"],
            "message": f"Output tokens estimate maximum {sim_cost['output_tokens_max']} is less than truth {real_cost['output_tokens_max']}",
        },
        {
            "check": sim_cost["time_min"] <= real_cost["time_min"],
            "message": f"Time estimate minimum {sim_cost['time_min']} exceeds truth {real_cost['time_min']}",
        },
        {
            "check": real_cost["time_max"] <= sim_cost["time_max"],
            "message": f"Time estimate maximum {sim_cost['time_max']} is less than truth {real_cost['time_max']}",
        },
        {
            "check": sim_cost["cost_min"] <= real_cost["cost_min"],
            "message": f"Cost estimate minimum {sim_cost['cost_min']} exceeds truth {real_cost['cost_min']}",
        },
        {
            "check": real_cost["cost_max"] <= sim_cost["cost_max"],
            "message": f"Cost estimate maximum {sim_cost['cost_max']} is less than truth {real_cost['cost_max']}",
        },
    ]
    failures = [assertion for assertion in assertions if not assertion["check"]]
    assert not failures, [assertion["message"] for assertion in failures]


@pytest.mark.slow
@pytest.mark.parametrize("params", PARAMSS)
def test_estimate_contains_exact(params):
    costlog = Costlog()
    real = chatgpt(
        **params,
        simulate=False,
        cost_log=costlog,
    )
    sim = chatgpt(
        **params,
        simulate=True,
        cost_log=costlog,
    )
    real_cost = costlog.items[0]
    sim_cost = costlog.items[1]
    check_cost_estimates(real_cost, sim_cost)


@pytest.mark.slow
@pytest.mark.parametrize("params", PARAMSS_PROMPT)
def test_estimate_contains_exact_prompt(params):
    costlog = Costlog()
    real = chatgpt_prompt(
        **params,
        simulate=False,
        cost_log=costlog,
    )
    sim = chatgpt_prompt(**params, simulate=True, cost_log=costlog)
    real_cost = costlog.items[0]
    sim_cost = costlog.items[1]
    check_cost_estimates(real_cost, sim_cost)


@pytest.mark.slow
@pytest.mark.parametrize("params", PARAMSS_INSTRUCTOR)
def test_estimate_contains_exact_instructor(params):
    costlog = Costlog()
    real = chatgpt_instructor(
        **params,
        simulate=False,
        cost_log=costlog,
    )
    sim = chatgpt_instructor(
        **params,
        simulate=True,
        cost_log=costlog,
    )
    real_cost = costlog.items[0]
    sim_cost = costlog.items[1]
    check_cost_estimates(real_cost, sim_cost)

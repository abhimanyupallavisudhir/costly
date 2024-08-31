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
    chatgpt_messages,
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
MODELS = LLM_API_Estimation.PRICES.keys()
MODELS_OPENAI = [model for model in MODELS if model.startswith("gpt")]
PARAMSS = [
    {"input_string": prompt, "model": model} for model in MODELS for prompt in PROMPTS
]

PARAMSS_INSTRUCTOR_ = [
    {
        "input_string": "Give me an example of a PERSONINFO object.",
        "response_model": PERSONINFO,
    },
    {
        "input_string": "Give me an example of a FOOMODEL object.",
        "response_model": FOOMODEL,
    },
    {
        "input_string": "Give me an example of a BARMODEL object.",
        "response_model": BARMODEL,
    },
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
            "message": "Input tokens estimate not within 20pc of truth",
        },
        {
            "check": sim_cost["output_tokens_min"] <= real_cost["output_tokens_min"],
            "message": "Output tokens estimate minimum exceeds truth",
        },
        {
            "check": real_cost["output_tokens_max"] <= sim_cost["output_tokens_max"],
            "message": "Output tokens estimate maximum is less than truth",
        },
        {
            "check": sim_cost["time_min"] <= real_cost["time_min"],
            "message": "Time estimate minimum exceeds truth",
        },
        {
            "check": real_cost["time_max"] <= sim_cost["time_max"],
            "message": "Time estimate maximum is less than truth",
        },
        {
            "check": sim_cost["cost_min"] <= real_cost["cost_min"],
            "message": "Cost estimate minimum exceeds truth",
        },
        {
            "check": real_cost["cost_max"] <= sim_cost["cost_max"],
            "message": "Cost estimate maximum is less than truth",
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
@pytest.mark.parametrize("params", PARAMSS)
def test_estimate_contains_exact_messages(params):
    params["messages"] = LLM_API_Estimation._input_string_to_messages(
        params.pop("input_string")
    )
    costlog = Costlog()
    real = chatgpt_messages(
        **params,
        simulate=False,
        cost_log=costlog,
    )
    sim = chatgpt_messages(**params, simulate=True, cost_log=costlog)
    real_cost = costlog.items[0]
    sim_cost = costlog.items[1]
    check_cost_estimates(real_cost, sim_cost)


# @pytest.mark.slow
@pytest.mark.parametrize("params", PARAMSS_INSTRUCTOR)
def test_estimate_contains_exact_instructor(params):
    params["messages"] = LLM_API_Estimation._input_string_to_messages(
        params.pop("input_string")
    )
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

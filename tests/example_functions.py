import instructor
from typing import Any
from pydantic import BaseModel
from costly import costly, CostlyResponse
from costly.estimators.llm_api_estimation import LLM_API_Estimation
from costly.simulators.llm_simulator_faker import LLM_Simulator_Faker
from openai import OpenAI, AsyncOpenAI
from instructor import Instructor
import asyncio

class PERSONINFO(BaseModel):
    name: str
    age: int


class FOOMODEL(BaseModel):
    name: str
    age: int
    bmi: float
    metadata: dict[str, Any] | None = None


class BARMODEL(BaseModel):
    foo: FOOMODEL
    fookids: list[FOOMODEL]


CLIENT = instructor.from_openai(OpenAI())
CLIENT_ASYNC = instructor.from_openai(AsyncOpenAI())

@costly()
def chatgpt(messages: list[dict[str, str]], model: str) -> str:
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(model=model, messages=messages)
    return CostlyResponse(
        output=response.choices[0].message.content,
        cost_info={
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        },
    )


@costly(
    messages=(lambda kwargs: kwargs["history"]),
    model=(lambda kwargs: kwargs["model_name"]),
)
def chatgpt2(history: list[dict[str, str]], model_name: str) -> str:
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(model=model_name, messages=history)
    return CostlyResponse(
        output=response.choices[0].message.content,
        cost_info={
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        },
    )


@costly(messages="history", model="model_name")
def chatgpt3(history: list[dict[str, str]], model_name: str) -> str:
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(model=model_name, messages=history)
    return CostlyResponse(
        output=response.choices[0].message.content,
        cost_info={
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        },
    )


@costly(
    input_tokens=lambda kwargs: LLM_API_Estimation.prompt_to_input_tokens(**kwargs),
)
def chatgpt_prompt(
    prompt: str, model: str, system_prompt: str = "You are a helpful assistant."
) -> str:
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"content": prompt, "role": "user"},
            {"content": system_prompt, "role": "system"},
        ],
    )
    output_string = response.choices[0].message.content
    return output_string


@costly(
    input_tokens=lambda kwargs: LLM_API_Estimation.get_input_tokens_instructor(
        **kwargs
    ),
)
def chatgpt_instructor(
    messages: list[dict[str, str]],
    model: str,
    client: Instructor,
    response_model: BaseModel,
) -> str:
    response = client.chat.completions.create_with_completion(
        model=model,
        messages=messages,
        response_model=response_model,
    )
    output_string, cost_info = response
    return CostlyResponse(
        output=output_string,
        cost_info={
            "input_tokens": cost_info.usage.prompt_tokens,
            "output_tokens": cost_info.usage.completion_tokens,
        },
    )


@costly()
async def chatgpt_async(messages: list[dict[str, str]], model: str) -> str:
    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    response = await client.chat.completions.create(model=model, messages=messages)
    return CostlyResponse(
        output=response.choices[0].message.content,
        cost_info={
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        },
    )


@costly(
    messages=(lambda kwargs: kwargs["history"]),
    model=(lambda kwargs: kwargs["model_name"]),
)
async def chatgpt2_async(history: list[dict[str, str]], model_name: str) -> str:
    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    response = await client.chat.completions.create(model=model_name, messages=history)
    return CostlyResponse(
        output=response.choices[0].message.content,
        cost_info={
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        },
    )


@costly(
    input_tokens=lambda kwargs: LLM_API_Estimation.prompt_to_input_tokens(**kwargs),
)
async def chatgpt_prompt_async(
    prompt: str, model: str, system_prompt: str = "You are a helpful assistant."
) -> str:
    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"content": prompt, "role": "user"},
            {"content": system_prompt, "role": "system"},
        ],
    )
    output_string = response.choices[0].message.content
    return output_string


@costly(
    input_tokens=lambda kwargs: LLM_API_Estimation.get_input_tokens_instructor(
        **kwargs
    ),
)
async def chatgpt_instructor_async(
    messages: list[dict[str, str]],
    model: str,
    client: Instructor,
    response_model: BaseModel,
) -> str:
    response = await client.chat.completions.create_with_completion(
        model=model,
        messages=messages,
        response_model=response_model,
    )
    output_string, cost_info = response
    return CostlyResponse(
        output=output_string,
        cost_info={
            "input_tokens": cost_info.usage.prompt_tokens,
            "output_tokens": cost_info.usage.completion_tokens,
        },
    )


@costly(
    simulator=LLM_Simulator_Faker.simulate_llm_probs,
)
def chatgpt_probs(messages: list[dict[str, str]], model: str, return_probs_for: list[str]) -> dict[str, float]:
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        logprobs=True,
        top_logprobs=len(return_probs_for)
    )
    
    logprobs = response.choices[0].logprobs.content[0].top_logprobs
    probs = {option: 0 for option in return_probs_for}
    
    for logprob in logprobs:
        if logprob.token in return_probs_for:
            probs[logprob.token] = 2 ** logprob.logprob  # Convert log probability to probability
    
    # Normalize probabilities
    total = sum(probs.values())
    normalized_probs = {k: v / total for k, v in probs.items()}
    
    return CostlyResponse(
        output=normalized_probs,
        cost_info={
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        },
    )

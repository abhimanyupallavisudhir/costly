import instructor
from typing import Any
from pydantic import BaseModel
from costly import costly, CostlyResponse
from costly.estimators.llm_api_estimation import LLM_API_Estimation
from openai import OpenAI
from instructor import Instructor

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

@costly()
def chatgpt(input_string: str, model: str) -> str:
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": input_string}]
    )
    return CostlyResponse(
        output=response.choices[0].message.content,
        cost_info={
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        },
    )



@costly(
    input_string=(lambda kwargs: kwargs["prompt"]),
    model=(lambda kwargs: kwargs["model_name"]),
)
def chatgpt2(prompt: str, model_name: str) -> str:
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model=model_name, messages=[{"role": "user", "content": prompt}]
    )
    return CostlyResponse(
        output=response.choices[0].message.content,
        cost_info={
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        },
    )


@costly(input_string="prompt", model="model_name")
def chatgpt3(prompt: str, model_name: str) -> str:
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model=model_name, messages=[{"role": "user", "content": prompt}]
    )
    return CostlyResponse(
        output=response.choices[0].message.content,
        cost_info={
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        },
    )

@costly(
    input_string=lambda kwargs: LLM_API_Estimation.messages_to_input_string(
        kwargs["messages"]
    ),
)
def chatgpt_messages(messages: list[dict[str, str]], model: str) -> str:
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
    input_string=lambda kwargs: LLM_API_Estimation.get_raw_prompt_instructor(**kwargs),
)
def chatgpt_instructor(
    messages: str | list[dict[str, str]],
    model: str,
    client: Instructor,
    response_model: BaseModel,
) -> str:
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
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
            "output_tokens": cost_info.usage.completion_tokens
        }
    )
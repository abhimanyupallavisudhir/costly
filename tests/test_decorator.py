import pytest
import instructor
from pydantic import BaseModel
from costly.decorator import costly
from costly.estimators.llm_api_estimation import LLM_API_Estimation
from openai import OpenAI
from instructor import Instructor


@costly()
def chatgpt(input_string: str, model: str) -> str:
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": input_string}]
    )
    output_string = response.choices[0].message.content
    return output_string


@costly(input_string="prompt", model="model_name")
def chatgpt2(prompt: str, model_name: str) -> str:
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model=model_name, messages=[{"role": "user", "content": prompt}]
    )
    output_string = response.choices[0].message.content
    return output_string


@costly(
    input_string=lambda kwargs: LLM_API_Estimation.messages_to_input_string(
        kwargs.pop("messages")
    ),
)
def chatgpt_messages(messages: list[dict[str, str]], model: str) -> str:
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(model=model, messages=messages)
    output_string = response.choices[0].message.content
    return output_string


client = instructor.from_openai(OpenAI())


@costly(
    input_string=lambda kwargs: LLM_API_Estimation.get_raw_prompt_instructor(
        messages=kwargs.pop("input_string"),
        client=kwargs.pop("client"),
        model=kwargs.get("model"),
        response_model=kwargs.get("response_model"),
    ),
)
def chatgpt_instructor(
    input_string: str, model: str, client: Instructor, response_model: BaseModel
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": input_string}],
        response_model=response_model,
    )
    return response

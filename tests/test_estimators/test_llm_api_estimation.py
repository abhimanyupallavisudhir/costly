import pytest
from pydantic import BaseModel
from costly.decorator import costly

@costly()
def chatgpt(input_string: str, model: str) -> str:
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": input_string}]
    )
    output_string = response.choices[0].message.content
    return output_string

@costly()
def chatgpt_messages(messages: list[dict[str, str]], model: str) -> str:
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content

@costly()
def instructor(input_string: str, model: str, response_model: BaseModel) -> str:
    import instructor
    from openai import OpenAI
    client = instructor.from_openai(OpenAI())
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": input_string}],
        response_model=response_model,
    )
    return response


import instructor
from pydantic import BaseModel
from costly import costly
from costly.estimators.llm_api_estimation import LLM_API_Estimation
from openai import OpenAI
from instructor import Instructor

class PERSONINFO(BaseModel):
    name: str
    age: int

CLIENT = instructor.from_openai(OpenAI())

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
    input_string=(lambda kwargs: kwargs["prompt"]),
    model=(lambda kwargs: kwargs["model_name"]),
)
def chatgpt2(prompt: str, model_name: str) -> str:
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model=model_name, messages=[{"role": "user", "content": prompt}]
    )
    output_string = response.choices[0].message.content
    return output_string


@costly(input_string="prompt", model="model_name")
def chatgpt3(prompt: str, model_name: str) -> str:
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model=model_name, messages=[{"role": "user", "content": prompt}]
    )
    output_string = response.choices[0].message.content
    return output_string


@costly(
    input_string=lambda kwargs: LLM_API_Estimation.messages_to_input_string(
        kwargs["messages"]
    ),
)
def chatgpt_messages(messages: list[dict[str, str]], model: str) -> str:
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(model=model, messages=messages)
    output_string = response.choices[0].message.content
    return output_string


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
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_model=response_model,
    )
    return response

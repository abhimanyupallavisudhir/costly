import pytest
import instructor
from pydantic import BaseModel
from costly import costly, Costlog
from costly.estimators.llm_api_estimation import LLM_API_Estimation
from openai import OpenAI
from instructor import Instructor

class PersonInfo(BaseModel):
    name: str
    age: int

client = instructor.from_openai(OpenAI())

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


def test_chatgpt():
    costlog = Costlog()
    x = chatgpt(
        input_string="Write the Lorem ipsum text",
        model="gpt-4o-mini",
        cost_log=costlog,
        simulate=False,
        description=["chatgpt call"],
    )
    y = chatgpt(
        input_string="Write the Lorem ipsum text",
        model="gpt-4o-mini",
        cost_log=costlog,
        simulate=True,
        description=["chatgpt call"],
    )
    assert isinstance(x, str)
    assert isinstance(y, str)
    assert len(costlog.items) == 2
    assert costlog.totals["calls"] == 2

def test_chatgpt2():
    costlog = Costlog()
    x = chatgpt2(
        prompt="Write the Lorem ipsum text",
        model_name="gpt-4o-mini",
        cost_log=costlog,
        simulate=False,
        description=["chatgpt call"],
    )
    y = chatgpt2(
        prompt="Write the Lorem ipsum text",
        model_name="gpt-4o-mini",
        cost_log=costlog,
        simulate=True,
        description=["chatgpt call"],
    )
    assert isinstance(x, str)
    assert isinstance(y, str)
    assert len(costlog.items) == 2
    assert costlog.totals["calls"] == 2

def test_chatgpt3():
    costlog = Costlog()
    x = chatgpt3(
        prompt="Write the Lorem ipsum text",
        model_name="gpt-4o-mini",
        cost_log=costlog,
        simulate=False,
        description=["chatgpt call"],
    )
    y = chatgpt3(
        prompt="Write the Lorem ipsum text",
        model_name="gpt-4o-mini",
        cost_log=costlog,
        simulate=True,
        description=["chatgpt call"],
    )
    assert isinstance(x, str)
    assert isinstance(y, str)
    assert len(costlog.items) == 2
    assert costlog.totals["calls"] == 2

def test_chatgpt_messages():
    costlog = Costlog()
    x = chatgpt_messages(
        messages=[{"role": "user", "content": "Write the Lorem ipsum text"}],
        model="gpt-4o-mini",
        cost_log=costlog,
        simulate=False,
        description=["chatgpt call"],
    )
    y = chatgpt_messages(
        messages=[{"role": "user", "content": "Write the Lorem ipsum text"}],
        model="gpt-4o-mini",
        cost_log=costlog,
        simulate=True,
        description=["chatgpt call"],
    )
    assert isinstance(x, str)
    assert isinstance(y, str)
    assert len(costlog.items) == 2
    assert costlog.totals["calls"] == 2

def test_chatgpt_instructor():
    costlog = Costlog()
    x = chatgpt_instructor(
        messages="Write the Lorem ipsum text",
        model="gpt-4o-mini",
        client=client,
        response_model=PersonInfo,
        cost_log=costlog,
        simulate=False,
        description=["chatgpt call"],
    )
    y = chatgpt_instructor(
        messages="Write the Lorem ipsum text",
        model="gpt-4o-mini",
        client=client,
        response_model=PersonInfo,
        cost_log=costlog,
        simulate=True,
        description=["chatgpt call"],
    )
    assert isinstance(x, PersonInfo)
    assert isinstance(y, PersonInfo)
    assert len(costlog.items) == 2
    assert costlog.totals["calls"] == 2
    

import pytest
import instructor
from pydantic import BaseModel
from costly import costly, Costlog
from costly.estimators.llm_api_estimation import LLM_API_Estimation
from openai import OpenAI
from instructor import Instructor
from tests.example_functions import (
    chatgpt,
    chatgpt2,
    chatgpt3,
    chatgpt_prompt,
    chatgpt_instructor,
    CLIENT,
    PERSONINFO,
)


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
    x = chatgpt_prompt(
        messages=[{"role": "user", "content": "Write the Lorem ipsum text"}],
        model="gpt-4o-mini",
        cost_log=costlog,
        simulate=False,
        description=["chatgpt call"],
    )
    y = chatgpt_prompt(
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
        client=CLIENT,
        response_model=PERSONINFO,
        cost_log=costlog,
        simulate=False,
        description=["chatgpt call"],
    )
    y = chatgpt_instructor(
        messages="Write the Lorem ipsum text",
        model="gpt-4o-mini",
        client=CLIENT,
        response_model=PERSONINFO,
        cost_log=costlog,
        simulate=True,
        description=["chatgpt call"],
    )
    assert isinstance(x, PERSONINFO)
    assert isinstance(y, PERSONINFO)
    assert len(costlog.items) == 2
    assert costlog.totals["calls"] == 2

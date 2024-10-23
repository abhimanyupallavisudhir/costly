import pytest
import asyncio
import instructor
from pydantic import BaseModel
from costly import costly, Costlog
from costly.estimators.llm_api_estimation import LLM_API_Estimation
from openai import OpenAI, AsyncOpenAI
from instructor import Instructor
from tests.example_functions import (
    chatgpt,
    chatgpt2,
    chatgpt3,
    chatgpt_prompt,
    chatgpt_instructor,
    chatgpt_async,
    chatgpt2_async,
    chatgpt_prompt_async,
    chatgpt_instructor_async,
    CLIENT,
    CLIENT_ASYNC,
    PERSONINFO,
)


def test_chatgpt():
    costlog = Costlog()
    x = chatgpt(
        messages=[{"role": "user", "content": "Write the Lorem ipsum text"}],
        model="gpt-4o-mini",
        cost_log=costlog,
        simulate=False,
        description=["chatgpt call"],
    )
    y = chatgpt(
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


def test_chatgpt2():
    costlog = Costlog()
    x = chatgpt2(
        history=[{"role": "user", "content": "Write the Lorem ipsum text"}],
        model_name="gpt-4o-mini",
        cost_log=costlog,
        simulate=False,
        description=["chatgpt call"],
    )
    y = chatgpt2(
        history=[{"role": "user", "content": "Write the Lorem ipsum text"}],
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
        history=[{"role": "user", "content": "Write the Lorem ipsum text"}],
        model_name="gpt-4o-mini",
        cost_log=costlog,
        simulate=False,
        description=["chatgpt call"],
    )
    y = chatgpt3(
        history=[{"role": "user", "content": "Write the Lorem ipsum text"}],
        model_name="gpt-4o-mini",
        cost_log=costlog,
        simulate=True,
        description=["chatgpt call"],
    )
    assert isinstance(x, str)
    assert isinstance(y, str)
    assert len(costlog.items) == 2
    assert costlog.totals["calls"] == 2


def test_chatgpt_prompt():
    costlog = Costlog()
    x = chatgpt_prompt(
        prompt="Write the Lorem ipsum text",
        model="gpt-4o-mini",
        cost_log=costlog,
        simulate=False,
        description=["chatgpt call"],
    )
    y = chatgpt_prompt(
        prompt="Write the Lorem ipsum text",
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
        messages=[{"role": "user", "content": "Write the Lorem ipsum text"}],
        model="gpt-4o-mini",
        client=CLIENT,
        response_model=PERSONINFO,
        cost_log=costlog,
        simulate=False,
        description=["chatgpt call"],
    )
    y = chatgpt_instructor(
        messages=[{"role": "user", "content": "Write the Lorem ipsum text"}],
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


@pytest.mark.asyncio
async def test_chatgpt_async():
    costlog = Costlog()
    x = await chatgpt_async(
        messages=[{"role": "user", "content": "Write the Lorem ipsum text"}],
        model="gpt-4o-mini",
        cost_log=costlog,
        simulate=False,
        description=["chatgpt call"],
    )
    y = await chatgpt_async(
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


@pytest.mark.asyncio
async def test_chatgpt2_async():
    costlog = Costlog()
    x = await chatgpt2_async(
        history=[{"role": "user", "content": "Write the Lorem ipsum text"}],
        model_name="gpt-4o-mini",
        cost_log=costlog,
        simulate=False,
        description=["chatgpt call"],
    )
    y = await chatgpt2_async(
        history=[{"role": "user", "content": "Write the Lorem ipsum text"}],
        model_name="gpt-4o-mini",
        cost_log=costlog,
        simulate=True,
        description=["chatgpt call"],
    )
    assert isinstance(x, str)
    assert isinstance(y, str)
    assert len(costlog.items) == 2
    assert costlog.totals["calls"] == 2


@pytest.mark.asyncio
async def test_chatgpt_prompt_async():
    costlog = Costlog()
    x = await chatgpt_prompt_async(
        prompt="Write the Lorem ipsum text",
        model="gpt-4o-mini",
        cost_log=costlog,
        simulate=False,
        description=["chatgpt call"],
    )
    y = await chatgpt_prompt_async(
        prompt="Write the Lorem ipsum text",
        model="gpt-4o-mini",
        cost_log=costlog,
        simulate=True,
        description=["chatgpt call"],
    )
    assert isinstance(x, str)
    assert isinstance(y, str)
    assert len(costlog.items) == 2
    assert costlog.totals["calls"] == 2


@pytest.mark.asyncio
async def test_chatgpt_instructor_async():
    costlog = Costlog()
    x = await chatgpt_instructor_async(
        messages=[{"role": "user", "content": "Write the Lorem ipsum text"}],
        model="gpt-4o-mini",
        client=CLIENT_ASYNC,
        response_model=PERSONINFO,
        cost_log=costlog,
        simulate=False,
        description=["chatgpt call"],
    )
    y = await chatgpt_instructor_async(
        messages=[{"role": "user", "content": "Write the Lorem ipsum text"}],
        model="gpt-4o-mini",
        client=CLIENT_ASYNC,
        response_model=PERSONINFO,
        cost_log=costlog,
        simulate=True,
        description=["chatgpt call"],
    )
    assert isinstance(x, PERSONINFO)
    assert isinstance(y, PERSONINFO)
    assert len(costlog.items) == 2
    assert costlog.totals["calls"] == 2


@pytest.mark.asyncio
async def test_chatgpt_async_parallel():
    costlog = Costlog()
    messages = [
        [{"role": "user", "content": "Write a short story"}],
        [{"role": "user", "content": "Explain quantum computing"}],
        [{"role": "user", "content": "Describe the water cycle"}],
    ]

    results = await asyncio.gather(
        *[
            chatgpt_async(
                messages=msg, model="gpt-4o-mini", cost_log=costlog, simulate=False
            )
            for msg in messages
        ]
    )

    assert len(results) == 3
    assert all(isinstance(result, str) for result in results)
    assert len(costlog.items) == 3
    assert costlog.totals["calls"] == 3


@pytest.mark.asyncio
async def test_chatgpt2_async_parallel():
    costlog = Costlog()
    histories = [
        [{"role": "user", "content": "Write a haiku"}],
        [{"role": "user", "content": "Explain photosynthesis"}],
        [{"role": "user", "content": "Describe the solar system"}],
    ]

    results = await asyncio.gather(
        *[
            chatgpt2_async(
                history=hist, model_name="gpt-4o-mini", cost_log=costlog, simulate=False
            )
            for hist in histories
        ]
    )

    assert len(results) == 3
    assert all(isinstance(result, str) for result in results)
    assert len(costlog.items) == 3
    assert costlog.totals["calls"] == 3


@pytest.mark.asyncio
async def test_chatgpt_prompt_async_parallel():
    costlog = Costlog()
    prompts = [
        "Write a limerick",
        "Explain the theory of relativity",
        "Describe the process of photosynthesis",
    ]

    results = await asyncio.gather(
        *[
            chatgpt_prompt_async(
                prompt=prompt, model="gpt-4o-mini", cost_log=costlog, simulate=False
            )
            for prompt in prompts
        ]
    )

    assert len(results) == 3
    assert all(isinstance(result, str) for result in results)
    assert len(costlog.items) == 3
    assert costlog.totals["calls"] == 3


@pytest.mark.asyncio
async def test_chatgpt_instructor_async_parallel():
    costlog = Costlog()
    messages_list = [
        [{"role": "user", "content": "Generate a person named Alice"}],
        [{"role": "user", "content": "Generate a person named Bob"}],
        [{"role": "user", "content": "Generate a person named Charlie"}],
    ]

    results = await asyncio.gather(
        *[
            chatgpt_instructor_async(
                messages=msgs,
                model="gpt-4o-mini",
                client=CLIENT_ASYNC,
                response_model=PERSONINFO,
                cost_log=costlog,
                simulate=False,
            )
            for msgs in messages_list
        ]
    )

    assert len(results) == 3
    assert all(isinstance(result, PERSONINFO) for result in results)
    assert len(costlog.items) == 3
    assert costlog.totals["calls"] == 3


@pytest.mark.asyncio
async def test_mixed_async_parallel():
    costlog = Costlog()

    results = await asyncio.gather(
        chatgpt_async(
            messages=[{"role": "user", "content": "Write a short story"}],
            model="gpt-4o-mini",
            cost_log=costlog,
            simulate=False,
        ),
        chatgpt2_async(
            history=[{"role": "user", "content": "Explain quantum computing"}],
            model_name="gpt-4o-mini",
            cost_log=costlog,
            simulate=False,
        ),
        chatgpt_prompt_async(
            prompt="Describe the water cycle",
            model="gpt-4o-mini",
            cost_log=costlog,
            simulate=False,
        ),
        chatgpt_instructor_async(
            messages=[{"role": "user", "content": "Generate a person named David"}],
            model="gpt-4o-mini",
            client=CLIENT_ASYNC,
            response_model=PERSONINFO,
            cost_log=costlog,
            simulate=False,
        ),
    )

    assert len(results) == 4
    assert isinstance(results[0], str)
    assert isinstance(results[1], str)
    assert isinstance(results[2], str)
    assert isinstance(results[3], PERSONINFO)
    assert len(costlog.items) == 4
    assert costlog.totals["calls"] == 4

from costly import Costlog, CostlyResponse, costly
from dotenv import load_dotenv

load_dotenv()  # remove when done

cl = Costlog()
nosim = False
sim = True


@costly()
def chatgpt(
    messages: list[dict[str, str]],
    model: str,
    cost_log: Costlog = cl,
    simulate: bool = nosim,
) -> str:
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


@costly()
def chatgpt_simulated(
    messages: list[dict[str, str]],
    model: str,
    cost_log: Costlog = cl,
    simulate: bool = sim,
) -> str:
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


def test_global():
    x = chatgpt(
        messages=[{"content": "Write the Lorem ipsum text", "role": "user"}],
        model="gpt-4o-mini",
    )
    y = chatgpt_simulated(
        messages=[{"content": "Write the Lorem ipsum text", "role": "user"}],
        model="gpt-4o-mini",
    )
    assert len(cl.items) == 2
    assert cl.items[0]["simulated"] == False
    assert cl.items[1]["simulated"] == True

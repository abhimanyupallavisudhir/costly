from costly import Costlog, CostlyResponse, costly

cost_log = Costlog()
simulate = False


@costly()
def chatgpt(
    messages: list[dict[str, str]],
    model: str,
    cost_log: Costlog = cost_log,
    simulate: bool = simulate,
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
    global simulate
    simulate = True
    y = chatgpt(
        messages=[{"content": "Write the Lorem ipsum text", "role": "user"}],
        model="gpt-4o-mini",
    )
    print(cost_log.items)


test_global()

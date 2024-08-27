import tiktoken
import logging
from io import StringIO
from instructor import Instructor
from pydantic import BaseModel
import json
import warnings
from unittest.mock import patch

"""Library of functions for creating cost items from LLM API calls"""

PRICES = {
    "gpt-4o": {
        "input_tokens": 5.0e-6,
        "output_tokens": 15.0e-6,
        "time": 18e-3,
    },
    "gpt-4o-mini": {
        "input_tokens": 0.15e-6,
        "output_tokens": 0.6e-6,
        "time": 9e-3,
    },
    "gpt-4-turbo": {
        "input_tokens": 10.0e-6,
        "output_tokens": 30.0e-6,
        "time": 36e-3,
    },
    "gpt-4": {
        "input_tokens": 30.0e-6,
        "output_tokens": 60.0e-6,
        "time": 36e-3,
    },
    "gpt-3.5-turbo": {
        "input_tokens": 0.50e-6,
        "output_tokens": 1.50e-6,
        "time": 36e-3,
    },
    "claude-3-5-sonnet": {
        "input_tokens": 3.0e-6,
        "output_tokens": 15.0e-6,
        "time": 18e-3,
    },
    "claude-3-opus": {
        "input_tokens": 15.0e-6,
        "output_tokens": 75.0e-6,
        "time": 18e-3,
    },
    "claude-3-haiku": {
        "input_tokens": 0.25e-6,
        "output_tokens": 1.25e-6,
        "time": 9e-3,
    },
}
"""
input_tokens: dollar per input token
output_tokens: dollar per output token
time: seconds per output token
"""


"""
Auxillary functions:
- get_model(model, supported_models=PRICES.keys()) -> model
- get_prices(model, price_dict=PRICES) -> prices
- tokenize(text, model) -> input_tokens
- tokenize_rough(text, model) -> input_tokens
- output_tokens_estimate(input_tokens, model) -> output_tokens_min, output_tokens_max
- get_raw_prompt(messages, client:Instructor, response_model:BaseModel) -> str
- process_raw_prompt(raw_prompt) -> str
"""


def get_model(model: str, supported_models: list = None):
    """Get model in supported_models with the longest prefix matching model"""
    if supported_models is None:
        supported_models = PRICES.keys()

    matching_keys = [key for key in supported_models if model.startswith(key)]
    if not matching_keys:
        raise ValueError(f"No matching model found for {model}")

    return max(matching_keys, key=len)


def get_prices(model: str, price_dict: dict = None):
    """Get prices for a model"""
    if price_dict is None:
        price_dict = PRICES

    return price_dict[get_model(model, price_dict.keys())]


def tokenize(input_string: str, model: str) -> int:
    """Tokenize text"""
    supported_models = [
        "gpt-4",
        "gpt-3.5-turbo",
        "text-embedding-ada-002",
        "text-embedding-3-small",
        "text-embedding-3-large",
    ]
    try:
        encoding = tiktoken.encoding_for_model(get_model(model, supported_models))
    except:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(input_string))


def tokenize_rough(input_string: str, model: str = None) -> int:
    "For a quick estimate, just divide by 4.5"
    return len(input_string) // 4.5


def output_tokens_estimate(
    input_string: str = None, input_tokens: int = None, model: str = None
) -> tuple[int, int]:
    return [0, 2048]


def get_raw_prompt_instructor(
    messages: list[dict[str, str]],
    client: Instructor,
    model: str,
    response_model: BaseModel,
):
    log_stream = StringIO()
    logger = logging.getLogger("instructor")
    logger.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler(log_stream)
    logger.addHandler(stream_handler)

    with patch("httpx.Client.send") as mock_send:
        mock_send.return_value = None  # prevent actual API calls

        try:
            bla = client.chat.completions.create(
                messages=messages,
                response_model=response_model,
                model=model,
            )
        except Exception:
            # there WILL be an error
            # there's nothing you can do about it
            pass  # cope and seethe

    log_contents = log_stream.getvalue()
    for line in log_contents.splitlines():
        if "Instructor Request" in line:
            line = line.split("Instructor Request: ")[1]
            return process_raw_prompt(line)

    warnings.warn(
        "No raw prompt found in logs. Maybe anthropic "
        "isn't supported or something idk"
    )
    return ""


def process_raw_prompt(input_string: str) -> str:
    try:
        # Step 1: Split at 'new_kwargs='
        split_parts = input_string.split("new_kwargs=", 1)
        if len(split_parts) < 2:
            warnings.warn(
                "Failed to split the string at 'new_kwargs='. Returning the original string."
            )
            return input_string

        json_string = split_parts[1].strip()

        # Step 2: Attempt to load the remaining part as JSON
        try:
            translation_table = str.maketrans({"'": '"', '"': "'"})
            json_string_trans = json_string.translate(translation_table)
            json_data = json.loads(json_string_trans)
        except json.JSONDecodeError:
            warnings.warn(
                "Failed to decode JSON with the translated string. "
                "Will try the original string."
            )
            try:
                json_data = json.loads(json_string)
            except json.JSONDecodeError:
                warnings.warn(
                    "Failed to decode JSON with the original string. "
                    "Returning the original string."
                )
                return input_string

        # Step 3: Try to get actual useful stuff in life
        if isinstance(json_data, dict):
            try:
                messages, tools, tool_choice = (
                    json_data.get("messages", []),
                    json_data.get("tools", []),
                    json_data.get("tool_choice", []),
                )
                str_messages = "".join([m["content"] for m in messages])
                funcs = [t["function"] for t in tools if "function" in t]
                func_names = [f.get("name", "") for f in funcs]
                func_descs = [f.get("description", "") for f in funcs]
                func_paras = [
                    f.get("parameters", {}).get("properties", {}).values()
                    for f in funcs
                ]  # list of lists of dicts
                func_para_values = [[p.values() for p in ps] for ps in func_paras]
                str_func_names = "".join(func_names)
                str_func_descs = "".join(func_descs)
                str_func_para_values = "".join(
                    ["".join([str(i) for i in v]) for v in func_para_values]
                )
                return (
                    str_messages
                    + str_func_names
                    + str_func_descs
                    + str_func_para_values
                )

            except KeyError:
                warnings.warn("Failed to extract relevant keys from decoded JSON.")
                return json_data
        else:
            warnings.warn(
                "Decoded JSON is not a dictionary. Returning the original JSON."
            )
            return json_data

        return json_data

    except Exception as e:
        warnings.warn(
            f"An unexpected error occurred: {e}. Returning the original string."
        )
        return input_string


"""
Cases:
- Simulating
    - From input_tokens, output_tokens
    - From input_tokens, output_string
    - From input_tokens
    - From input_string
- Real
    - From input_tokens, output_tokens, timer

"""


def get_costitem_simulating_from_input_tokens_output_tokens(
    input_tokens: int,
    output_tokens_min: int,
    output_tokens_max: int,
    model: str,
    **kwargs,
) -> dict[str, float]:
    prices = get_prices(model)
    cost_input_tokens = input_tokens * prices["input_tokens"]
    cost_output_tokens_min = output_tokens_min * prices["output_tokens"]
    cost_output_tokens_max = output_tokens_max * prices["output_tokens"]
    time_min = output_tokens_min * prices["time"]
    time_max = output_tokens_max * prices["time"]
    return {
        "cost_min": cost_input_tokens + cost_output_tokens_min,
        "cost_max": cost_input_tokens + cost_output_tokens_max,
        "time_min": time_min,
        "time_max": time_max,
        "input_tokens": input_tokens,
        "output_tokens_min": output_tokens_min,
        "output_tokens_max": output_tokens_max,
        "calls": 1,
        "model": model,
        "simulated": True,
        **kwargs,
    }
    
def get_tokens(
    model: str,
    input_tokens: int = None,
    output_tokens_min: int = None,
    output_tokens_max: int = None,
    input_string: str = None,
    output_string: str = None,
) -> tuple[int, int, int]:
    if input_tokens is None:
        try:
            assert input_string is not None
            input_tokens = tokenize(input_string, model)
        except:
            raise ValueError(f"Failed to tokenize input_string {input_string}")
    if output_tokens_min is None or output_tokens_max is None:
        try:
            assert output_string is not None
            output_tokens_min, output_tokens_max = tokenize(output_string, model)
        except:
            try:
                output_tokens_min, output_tokens_max = output_tokens_estimate(
                    input_string=input_string, input_tokens=input_tokens, model=model
                )
            except:
                raise ValueError(
                    f"Failed to estimate output tokens for input_string "
                    f"{input_string} and input_tokens {input_tokens}"
                )
    return input_tokens, output_tokens_min, output_tokens_max


def get_cost_item_simulating(
    model: str,
    input_tokens: int = None,
    output_tokens_min: int = None,
    output_tokens_max: int = None,
    input_string: str = None,
    output_string: str = None,
    **kwargs,
) -> dict[str, float]:
    input_tokens, output_tokens_min, output_tokens_max = get_tokens(
        model=model,
        input_tokens=input_tokens,
        output_tokens_min=output_tokens_min,
        output_tokens_max=output_tokens_max,
        input_string=input_string,
        output_string=output_string,
    )
    return get_costitem_simulating_from_input_tokens_output_tokens(
        input_tokens=input_tokens,
        output_tokens_min=output_tokens_min,
        output_tokens_max=output_tokens_max,
        model=model,
        input_string=input_string,
        output_string=output_string,
        **kwargs,
    )


def get_costitem_real_from_input_tokens_output_tokens_timer(
    input_tokens: int, output_tokens: int, timer: float, model: str, **kwargs
) -> dict[str, float]:
    prices = get_prices(model)
    cost_input_tokens = input_tokens * prices["input_tokens"]
    cost_output_tokens = output_tokens * prices["output_tokens"]
    time = timer
    return {
        "cost_min": cost_input_tokens + cost_output_tokens,
        "cost_max": cost_input_tokens + cost_output_tokens,
        "time_min": time,
        "time_max": time,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "calls": 1,
        "model": model,
        "simulated": False,
        **kwargs,
    }

def get_costitem_real(
    model: str,
    input_tokens: int = None,
    output_tokens_min: int = None,
    output_tokens_max: int = None,
    input_string: str = None,
    output_string: str = None,
    timer: float = None,
    **kwargs,
) -> dict[str, float]:
    assert timer is not None
    input_tokens, output_tokens_min, output_tokens_max = get_tokens(
        model=model,
        input_tokens=input_tokens,
        output_tokens_min=output_tokens_min,
        output_tokens_max=output_tokens_max,
        input_string=input_string,
        output_string=output_string,
    )
    return get_costitem_real_from_input_tokens_output_tokens_timer(
        input_tokens=input_tokens,
        output_tokens_min=output_tokens_min,
        output_tokens_max=output_tokens_max,
        timer=timer,
        model=model,
        input_string=input_string,
        output_string=output_string,
        **kwargs,
    )

    

import tiktoken
import logging
from io import StringIO
from pydantic import BaseModel
import ast
import warnings
from unittest.mock import patch


class LLM_API_Estimation:
    """Library of functions for creating cost items from LLM API calls.
    You may subclass this and override any of the attributes or methods.

    Attributes:
    - PRICES: dictionary of prices for each model

    Methods to produce cost items:
    - get_cost_simulating(...) -> dict[str, float]
    - get_cost_real(...) -> dict[str, float]

    Auxillary methods:
    - get_model(model, supported_models=PRICES.keys()) -> model (get the model with the longest prefix matching model)
    - get_prices(model, price_dict=PRICES) -> prices
    - tokenize(text, model) -> input_tokens (tokenize text with tiktoken)
    - output_tokens_estimate(input_tokens, model) -> output_tokens_min, output_tokens_max
      (estimate output tokens for simulating)

    Methods to help convert things to input_tokens or input_string:
    - messages_to_input_tokens(messages: list[dict[str, str]]) -> int
    - prompt_to_input_tokens(prompt: str, system_prompt: str = None, model: str = None) -> int
    - get_raw_input_string_instructor(messages: str | list[dict[str, str]], response_model:BaseModel) -> str

    Private methods:
    - _get_cost_simulating_from_input_tokens_output_tokens(...) -> dict[str, float]
    - _get_cost_real_from_input_tokens_output_tokens_timer(...) -> dict[str, float]
    - _get_tokens(...) -> tuple[int, int, int]
    - _process_raw_prompt(...) -> str
    - _tokenize_rough(text, model) -> input_tokens

    Subclassing tips:
    - PRICES: to change the prices/times for each model, or support new models or model names e.g.
      PRICES = LLM_API_Estimation.PRICES | {"my_model": LLM_API_Estimation.PRICES["gpt-4o"]}
    - tokenize: e.g.
      tokenize=_tokenize_rough
    - output_tokens_estimate: to change the way you estimate output tokens for simulating
    - get_model: e.g. if you want to just match models literally --
      get_model=lambda model, supported_models: model
    - get_raw_input_string_instructor: e.g. if instructor adds a better way to see the raw prompt
    - _process_raw_prompt: e.g. if instructor adds a better way to see the raw prompt
    - _get_cost_simulating_from_input_tokens_output_tokens,
      _get_cost_real_from_input_tokens_output_tokens_timer,
      get_cost_simulating,
      get_cost_real
      if you're going to change these (e.g. to use this library for some purpose other than
      LLM API calls), maybe just define a new class
    """

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

    @classmethod
    def get_model(cls, model: str, supported_models: list = None):
        """Get model in supported_models with the longest prefix matching model"""
        if supported_models is None:
            supported_models = cls.PRICES.keys()

        matching_keys = [key for key in supported_models if model.startswith(key)]
        if not matching_keys:
            raise ValueError(f"No matching model found for {model}")

        return max(matching_keys, key=len)

    @classmethod
    def get_prices(cls, model: str, price_dict: dict = None):
        """Get prices for a model"""
        if price_dict is None:
            price_dict = cls.PRICES

        return price_dict[cls.get_model(model, price_dict.keys())]

    @classmethod
    def tokenize(cls, input_string: str, model: str) -> int:
        """Tokenize text"""
        supported_models = [
            "gpt-4",
            "gpt-3.5-turbo",
            "text-embedding-ada-002",
            "text-embedding-3-small",
            "text-embedding-3-large",
        ]
        try:
            encoding = tiktoken.encoding_for_model(
                cls.get_model(model, supported_models)
            )
        except:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(input_string))

    @classmethod
    def _tokenize_rough(cls, input_string: str, model: str = None) -> int:
        "For a quick estimate, just divide by 4.5"
        return len(input_string) // 4.5

    @classmethod
    def output_tokens_estimate(
        cls,
        input_string: str = None,
        messages: list[dict[str, str]] = None,
        input_tokens: int = None,
        model: str = None,
    ) -> tuple[int, int]:
        return [0, 2048]

    @classmethod
    def messages_to_input_tokens(
        cls,
        messages: list[dict[str, str]], model: str = None
    ) -> int:
        """Return the number of tokens used by a list of messages.

        From: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        """
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            warnings.warn(
                "messages_to_input_tokens: model not found. Using cl100k_base encoding."
            )
            encoding = tiktoken.get_encoding("cl100k_base")
        if model in {
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k-0613",
        }:
            tokens_per_message = 3
            tokens_per_name = 1
        elif model == "gpt-3.5-turbo-0301":
            tokens_per_message = (
                4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            )
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif "gpt-3.5-turbo" in model:
            warnings.warn(
                "messages_to_input_tokens: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613."
            )
            return cls.messages_to_input_tokens(
                messages, model="gpt-3.5-turbo-0613"
            )
        elif "gpt-4" in model:
            warnings.warn(
                "messages_to_input_tokens: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613."
            )
            return cls.messages_to_input_tokens(
                messages, model="gpt-4-0613"
            )
        else:
            warnings.warn(
                f"messages_to_input_tokens: model {model} not found. Returning num tokens assuming gpt-4-0613."
            )
            return cls.messages_to_input_tokens(
                messages, model="gpt-4-0613"
            )
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    @classmethod
    def prompt_to_messages(
        cls,
        prompt: str,
        system_prompt: str = None,
    ) -> list[dict[str, str]]:
        messages = []
        if system_prompt is not None:
            messages.append({"content": system_prompt, "role": "system"})
        messages.append({"content": prompt, "role": "user"})
        return messages

    @classmethod
    def prompt_to_input_tokens(
        cls,
        prompt: str,
        system_prompt: str = None,
        model: str = None,
    ) -> int:
        return cls.messages_to_input_tokens(
            cls.prompt_to_messages(prompt, system_prompt), model
        )

    @classmethod
    def get_input_tokens_instructor(
        cls,
        messages: str | list[dict[str, str]],
        model: str,
        response_model: BaseModel,
        **kwargs,  # just let people pass in whatever they want
    ):
        """
        From: https://community.openai.com/t/how-to-calculate-the-tokens-when-using-function-call/266573/11
        """
        instructor_messages = cls._get_raw_messages_instructor(
            messages=messages,
            model=model,
            response_model=response_model,
            process=True,
            **kwargs,
        )
        messages = instructor_messages["messages"]
        tools = instructor_messages["tools"]
        functions = [tool["function"] for tool in tools if tool["type"] == "function"]

        functions_tokens = 0
        functions_tokens += 12
        for function in functions:
            function_tokens = cls.tokenize(
                function["name"], model
            ) + cls.tokenize(function["description"], model)
            if "parameters" in function:
                parameters = function["parameters"]
                if "properties" in parameters:
                    function_tokens += 11
                    properties = parameters["properties"]
                    for properties_key, properties_value in properties.items():
                        function_tokens += cls.tokenize(
                            properties_key, model
                        )
                        for field, field_value in properties_value.items():
                            if field == "type" or field == "description":
                                function_tokens += (
                                    cls.tokenize(field_value, model) + 2
                                )
                            elif field == "enum":
                                function_tokens -= 3
                                for o in field_value:
                                    function_tokens += (
                                        cls.tokenize(o, model) + 3
                                    )
                            else:
                                warnings.warn(
                                    f"Field {field} not found in tokenization function"
                                )
            functions_tokens += function_tokens

        messages_tokens = cls.messages_to_input_tokens(messages, model)
        return messages_tokens + functions_tokens

    @classmethod
    def _get_raw_messages_instructor(
        cls,
        messages: str | list[dict[str, str]],
        model: str,
        response_model: BaseModel,
        process=True,
        **kwargs,  # just let people pass in whatever they want
    ) -> str | dict:
        
        import instructor
        from openai import OpenAI
        client = instructor.from_openai(OpenAI(api_key='MOCK'))
        
        if isinstance(messages, str):
            messages = [{"content": messages, "role": "user"}]
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
                if process:
                    return cls._process_raw_prompt(line)
                return line

        warnings.warn(
            "No raw prompt found in logs. Maybe anthropic "
            "isn't supported or something idk"
        )
        return ""        

    @classmethod
    def _process_raw_prompt(cls, input_string: str) -> dict:
        # Step 1: Split at 'new_kwargs='
        split_parts = input_string.split("new_kwargs=", 1)
        if len(split_parts) < 2:
            warnings.warn(
                "Failed to split the string at 'new_kwargs='. Returning the original string."
            )
            return input_string

        dict_string = split_parts[1].strip()

        # Step 2: Attempt to load the remaining part as a dict
        dict_data = ast.literal_eval(dict_string)

        if not isinstance(dict_data, dict):
            warnings.warn(
                f"Decoded object is not a dictionary, but a {type(dict_data)}."
            )

        return dict_data

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

    @classmethod
    def _get_cost_simulating_from_input_tokens_output_tokens(
        cls,
        input_tokens: int,
        output_tokens_min: int,
        output_tokens_max: int,
        model: str,
        **kwargs,
    ) -> dict[str, float]:
        prices = cls.get_prices(model)
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

    @classmethod
    def _get_tokens(
        cls,
        model: str,
        input_tokens: int = None,
        output_tokens_min: int = None,
        output_tokens_max: int = None,
        input_string: str = None,
        messages: list[dict[str, str]] = None,
        output_string: str = None,
    ) -> tuple[int, int, int]:
        if input_tokens is None:
            try:
                assert input_string is not None or messages is not None
                if input_string is not None:
                    input_tokens = cls.tokenize(input_string, model)
                else:
                    input_tokens = cls.messages_to_input_tokens(
                        messages, model
                    )
            except:
                raise ValueError(
                    f"Failed to tokenize input_string {input_string} or messages {messages}"
                )
        if output_tokens_min is None or output_tokens_max is None:
            try:
                assert output_string is not None
                output_tokens = cls.tokenize(output_string, model)
                output_tokens_min, output_tokens_max = output_tokens, output_tokens
            except:
                try:
                    output_tokens_min, output_tokens_max = (
                        cls.output_tokens_estimate(
                            input_string=input_string,
                            messages=messages,
                            input_tokens=input_tokens,
                            model=model,
                        )
                    )
                except:
                    raise ValueError(
                        f"Failed to estimate output tokens for input_string "
                        f"{input_string} and input_tokens {input_tokens}"
                    )
        return input_tokens, output_tokens_min, output_tokens_max

    @classmethod
    def get_cost_simulating(
        cls,
        model: str,
        input_tokens: int = None,
        output_tokens_min: int = None,
        output_tokens_max: int = None,
        input_string: str = None,
        messages: list[dict[str, str]] = None,
        output_string: str = None,
        **kwargs,
    ) -> dict[str, float]:
        input_tokens, output_tokens_min, output_tokens_max = (
            cls._get_tokens(
                model=model,
                input_tokens=input_tokens,
                output_tokens_min=output_tokens_min,
                output_tokens_max=output_tokens_max,
                input_string=input_string,
                messages=messages,
                output_string=output_string,
            )
        )
        return cls._get_cost_simulating_from_input_tokens_output_tokens(
            input_tokens=input_tokens,
            output_tokens_min=output_tokens_min,
            output_tokens_max=output_tokens_max,
            model=model,
            input_string=input_string,
            messages=messages,
            output_string=output_string,
            **kwargs,
        )

    @classmethod
    def _get_cost_real_from_input_tokens_output_tokens_timer(
        cls,
        input_tokens: int, output_tokens: int, timer: float, model: str, **kwargs
    ) -> dict[str, float]:
        prices = cls.get_prices(model)
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
            "output_tokens_min": output_tokens,
            "output_tokens_max": output_tokens,
            "calls": 1,
            "model": model,
            "simulated": False,
            **kwargs,
        }

    @classmethod
    def get_cost_real(
        cls,
        model: str,
        input_tokens: int = None,
        output_tokens: int = None,
        input_string: str = None,
        messages: list[dict[str, str]] = None,
        output_string: str = None,
        timer: float = None,
        **kwargs,
    ) -> dict[str, float]:
        assert timer is not None
        input_tokens, output_tokens, _ = cls._get_tokens(
            model=model,
            input_tokens=input_tokens,
            output_tokens_min=output_tokens,
            output_tokens_max=output_tokens,
            input_string=input_string,
            messages=messages,
            output_string=output_string,
        )

        return cls._get_cost_real_from_input_tokens_output_tokens_timer(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            timer=timer,
            model=model,
            input_string=input_string,
            messages=messages,
            output_string=output_string,
            **kwargs,
        )

import types
import typing
import random
from typing import Any
from faker import Faker
from pydantic import BaseModel
from costly.costlog import Costlog
from costly.estimators.llm_api_estimation import LLM_API_Estimation


class LLM_Simulator_Faker:
    """
    Main things you would likely subclass:
    - FAKER_PARAMS: default is:
        ```python
        {
            "str": {
                "max_nb_chars": 600 * 4.5,
            },
            "int": {
                "min": 0,
                "max": 100,
            },
            "float": {
                "a": -20,
                "b": 20,
            },
        }
        ```
        Override this to make it generate different lengths of text, etc.
    - _fake_custom: to add highest-priority/overriding behaviour for your own types
    - tryfuncs: to index any new faking methods you define or change priorities
    - redirects: to change type redirects like list -> list[str], object -> str, etc.

    Example:

    ```python
    class MySimulator(LLM_Simulator_Faker):

        tryfuncs = super().tryfuncs + [
            {"func": _fake_pandas_df, "priority": 100},
        ]

        def _fake_pandas_df(self, t: type):
            if t.__module__ == "pandas" and t.__name__ == "DataFrame":
                return pd.DataFrame(...)
            else:
                raise ValueError(f"Unsupported type: {t}")

        def _fake_custom(self, t: type):
            if t == MyType:
                ...
            else:
                raise ValueError(f"Unsupported type: {t}")
    ```

    Attributes:
        FAKER: Faker instance
        tryfuncs: list of tryfuncs with priorities
        redirects: dict of types to redirect to

    Methods:
        simulate_llm_call: simulate an LLM call
        fake: fake a value of a given type
        _fake_basic: fake a basic value
        _fake_paramgeneric: fake a value of a generic type
        _fake_redirect: fake a value of a redirected type
        _fake_pydantic: fake a value of a pydantic type
        _fake_datetime: fake a value of a datetime type
        _fake_uuid: fake a value of a uuid type
        _fake_typing: fake a value of a typing type
        _fake_custom: fake a value of a custom type
    """

    FAKER = Faker()

    FAKER_PARAMS = {
        "str": {
            "max_nb_chars": 600 * 4.5,
        },
        "int": {
            "min": 0,
            "max": 100,
        },
        "float": {
            "a": -20,
            "b": 20,
        },
        "size_set": {
            "min": 0,
            "max": 10,
        },
        "size_dict": {
            "min": 0,
            "max": 10,
        },
        "size_list": {
            "min": 0,
            "max": 10,
        },
    }

    @classmethod
    def simulate_llm_call(
        cls,
        input_string: str = None,  # must supply at least one of input_string, input_tokens, messages
        input_tokens: int = None,
        messages: list[dict[str, str]] = None,
        model: str = None,
        response_model: type = str,
        cost_log: Costlog = None,
        description: list[str] = None,
    ) -> str | Any:
        """
        Simulate an LLM call.
        """
        response_model = response_model or str
        response = cls.fake(response_model)
        if cost_log is not None:
            assert model is not None, "model is required for tracking costs"
            with cost_log.new_item() as (item, _):
                cost_item = LLM_API_Estimation.get_cost_simulating(
                    input_string=input_string,
                    input_tokens=input_tokens,
                    model=model,
                    messages=messages,
                    description=description,
                    # output_string=response, # not needed
                )
                item.update(cost_item)
        return response

    @classmethod
    def simulate_llm_probs(
        cls,
        return_probs_for: list[str],
        input_string: str = None,  # must supply at least one of input_string, input_tokens, messages
        input_tokens: int = None,
        messages: list[dict[str, str]] = None,
        model: str = None,
        cost_log: Costlog = None,
        description: list[str] = None,
    ) -> dict[str, float]:
        """
        Simulate a dict with keys as return_probs_for and values as probabilities.
        """
        probs_ = {key: random.random() for key in return_probs_for}
        total = sum(probs_.values())
        probs = {key: value / total for key, value in probs_.items()}
        if cost_log is not None:
            assert model is not None, "model is required for tracking costs"
            with cost_log.new_item() as (item, _):
                cost_item = LLM_API_Estimation.get_cost_simulating(
                    input_string=input_string,
                    input_tokens=input_tokens,
                    output_tokens_min=0,
                    output_tokens_max=max(len(key) for key in return_probs_for),
                    model=model,
                    messages=messages,
                    description=description,
                )
                item.update(cost_item)
        return probs

    @classmethod
    def fake(cls, t: type):
        try_funcs_in_order = sorted(
            cls.tryfuncs, key=lambda x: x["priority"], reverse=True
        )
        exceptions = []
        for try_func in try_funcs_in_order:
            try:
                return try_func["func"](cls, t)
            except Exception as e:
                exceptions.append({"try_func": try_func, "exception": e})
        raise ValueError(
            f"Unsupported type: {t}\n"
            + "\n".join([str(e) for e in exceptions])
        )

    @classmethod
    def _fake_basic(cls, t: type):
        if t == type(None):
            return None
        elif isinstance(t, types.UnionType):
            t = cls.FAKER.random_element(elements=t.__args__)
            return cls.fake(t)
        elif t == str:
            return cls.FAKER.text(**cls.FAKER_PARAMS["str"])
        elif t == int:
            return cls.FAKER.random_int(**cls.FAKER_PARAMS["int"])
        elif t == float:
            return random.uniform(**cls.FAKER_PARAMS["float"])
        elif t == bool:
            return cls.FAKER.random_element(elements=[True, False])
        else:
            raise ValueError(f"Unsupported type: {t}")

    @classmethod
    def _fake_paramgeneric(cls, t: type):
        origin, args = t.__origin__, t.__args__
        if origin == list:
            return [
                cls.fake(args[0])
                for _ in range(cls.FAKER.random_int(**cls.FAKER_PARAMS["size_list"]))
            ]
        elif origin == dict:
            return {
                cls.fake(args[0]): cls.fake(args[1])
                for _ in range(cls.FAKER.random_int(**cls.FAKER_PARAMS["size_dict"]))
            }
        elif origin == tuple:
            return tuple(cls.fake(args[i]) for i in range(len(args)))
        elif origin == set:
            return {
                cls.fake(args[0])
                for _ in range(cls.FAKER.random_int(**cls.FAKER_PARAMS["size_set"]))
            }
        else:
            raise ValueError(f"Unsupported type: {t}")

    @classmethod
    def _fake_redirect(cls, t: type):
        if t in cls.redirects:
            return cls.fake(cls.redirects[t])
        else:
            raise ValueError(f"Unsupported type: {t}")

    @classmethod
    def _fake_pydantic(cls, t: type):
        assert issubclass(t, BaseModel)
        factory_dict = {}
        for name, field in t.model_fields.items():
            factory_dict[name] = cls.fake(field.annotation)
        return t(**factory_dict)

    @classmethod
    def _fake_datetime(cls, t: type):
        assert t.__module__ == "datetime"
        if t.__name__ == "datetime":
            return cls.FAKER.date_time_this_decade()
        elif t.__name__ == "date":
            return cls.FAKER.date_this_decade()
        elif t.__name__ == "time":
            return cls.FAKER.time_object()
        else:
            raise ValueError(f"Unsupported type: {t}")

    @classmethod
    def _fake_uuid(cls, t: type):
        assert t.__module__ == "uuid" and t.__name__ == "UUID"
        return cls.FAKER.uuid4(cast_to=None)

    @classmethod
    def _fake_typing(cls, t: type):
        assert t.__module__ == "typing"
        if t.__name__ == "Any":
            t = object
        elif t.__name__ == "Union":
            t = cls.FAKER.random_element(elements=t.__args__)
        elif t.__name__ == "Optional":
            t = cls.FAKER.random_element(elements=[type(None), t.__args__[0]])
        elif typing.get_origin(t) is not None:
            t = typing.get_origin(t)
        else:
            raise ValueError(f"Unsupported type: {t}")
        return cls.fake(t)

    @classmethod
    def _fake_custom(cls, t: type):
        raise ValueError("_fake_custom must be subclassed to be used")

    tryfuncs = [
        {"func": lambda cls, t: cls._fake_custom(t), "priority": 1000},
        {"func": lambda cls, t: cls._fake_paramgeneric(t), "priority": 500},
        {"func": lambda cls, t: cls._fake_pydantic(t), "priority": 100},
        {"func": lambda cls, t: cls._fake_basic(t), "priority": 100},
        {"func": lambda cls, t: cls._fake_datetime(t), "priority": 100},
        {"func": lambda cls, t: cls._fake_uuid(t), "priority": 100},
        {"func": lambda cls, t: cls._fake_typing(t), "priority": 100},
        {"func": lambda cls, t: cls._fake_redirect(t), "priority": -100},
    ]
    """
    Priority of tryfuncs:
    - _fake_custom: 1000
    - _fake_paramgeneric: 500
    - _fake_pydantic: 100
    - _fake_basic: 100
    - _fake_datetime: 100
    - _fake_uuid: 100
    - _fake_typing: 0
    - _fake_redirect: -100
    
    Important that:
    - _fake_custom is highest, as it should be used to override
      all other behaviour
    - _fake_redirect is lowest, to avoid unwanted redirects e.g. for parametric
      generics
    - _fake_typing is also pretty low, so that _fake_paramgeneric
      can do its job when possible
    """

    redirects = {
        list: list[str],
        tuple: tuple[str, str],
        set: set[str],
        dict: dict[str, str],
        object: str,
    }

# earlier method based on Polyfactory
# no longer in use
# 
# # Function to dynamically create a Polyfactory factory class for any Pydantic model
# def create_factory_for_model(model: Type[BaseModel]) -> Type[ModelFactory]:
#     type_based_overrides = {
#         ForecastingQuestion: lambda: pick_random_fq(
#             get_data_path() / "fq" / "real" / "test_formatted.jsonl"
#         ),
#         ForecastingQuestion_stripped: lambda: pick_random_fq(
#             get_data_path() / "fq" / "real" / "test_formatted.jsonl", strip=True
#         ),
#         Prob: lambda: Prob(prob=random.uniform(0.01, 0.99)),
#         Prob_cot: lambda: Prob_cot(
#             chain_of_thought="I'm thinking about this with a lot of attention and have come to the conclusion",
#             prob=random.uniform(0.01, 0.99),
#         ),
#         float: lambda: random.uniform(0.01, 9.99),
#         # float: lambda: random.uniform(0.01, 0.99),
#     }

#     # Define the dynamic factory class
#     class DynamicFactory(ModelFactory[model]):
#         __model__ = model

#         # @classmethod
#         # def get_field_value(
#         #     cls, field_meta: PydanticFieldMeta, *args: Any, **kwargs: Any
#         # ):

#         #     # if not field_meta.name:
#         #     #     return super().get_field_value(field_meta, *args, **kwargs)

#         #     # field_name = field_meta.name
#         #     # field = model.model_fields[field_name]
#         #     # if field.annotation in type_based_overrides:
#         #     #     return type_based_overrides[field.annotation]()
#         #     # return super().get_field_value(field_meta, *args, **kwargs)

#         #     if field_meta.annotation in type_based_overrides:
#         #         # Return the pre-generated instance to prevent recursion
#         #         return type_based_overrides[field_meta.annotation]()

#         #     # For other fields, use the default generation logic
#         #     return super().get_field_value(field_meta, *args, **kwargs)

#         @classmethod
#         def build(cls, *args: Any, **kwargs: Any) -> BaseModel:
#             # If the model has a type-based override, return it directly
#             if cls.__model__ in type_based_overrides:
#                 return type_based_overrides[cls.__model__]()

#             # Otherwise, proceed with default behavior
#             return super().build(*args, **kwargs)

#     return DynamicFactory

#     return DynamicFactory
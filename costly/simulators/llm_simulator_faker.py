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
    Main things you would like to subclass:
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

    @staticmethod
    def simulate_llm_call(
        input_string: str,
        model: str = None,
        response_model: type = str,
        cost_log: Costlog = None,
        description: list[str] = None,
    ) -> str | Any:
        """
        Simulate an LLM call.
        """
        response_model = response_model or str
        response = LLM_Simulator_Faker.fake(response_model)
        if cost_log is not None:
            assert model is not None, "model is required for tracking costs"
            with cost_log.new_item() as (item, _):
                cost_item = LLM_API_Estimation.get_cost_simulating(
                    input_string=input_string,
                    model=model,
                    description=description,
                    # output_string=response, # not needed
                )
                item.update(cost_item)
        return response

    @staticmethod
    def fake(t: type):
        # sort tryfuncs by priority
        try_funcs_in_order = sorted(
            LLM_Simulator_Faker.tryfuncs, key=lambda x: x["priority"], reverse=True
        )
        for try_func in try_funcs_in_order:
            try:
                return try_func["func"](t)
            except:
                pass
        raise ValueError(f"Unsupported type: {t}")

    @staticmethod
    def _fake_basic(t: type):
        if t == type(None):
            return None
        elif isinstance(t, types.UnionType):
            t = LLM_Simulator_Faker.FAKER.random_element(elements=t.__args__)
            return LLM_Simulator_Faker.fake(t)
        elif t == str:
            return LLM_Simulator_Faker.FAKER.text(max_nb_chars=int(600 * 4.5))
        elif t == int:
            return LLM_Simulator_Faker.FAKER.random_int(min=0, max=100)
        elif t == float:
            return random.random() * 40 - 20
        elif t == bool:
            return LLM_Simulator_Faker.FAKER.random_element(elements=[True, False])
        else:
            raise ValueError(f"Unsupported type: {t}")

    @staticmethod
    def _fake_paramgeneric(t: type):
        origin, args = t.__origin__, t.__args__
        if origin == list:
            return [
                LLM_Simulator_Faker.fake(args[0])
                for _ in range(LLM_Simulator_Faker.FAKER.random_int(min=0, max=10))
            ]
        elif origin == dict:
            return {
                LLM_Simulator_Faker.fake(args[0]): LLM_Simulator_Faker.fake(args[1])
                for _ in range(LLM_Simulator_Faker.FAKER.random_int(min=0, max=10))
            }
        elif origin == tuple:
            return tuple(LLM_Simulator_Faker.fake(args[i]) for i in range(len(args)))
        else:
            raise ValueError(f"Unsupported type: {t}")

    @staticmethod
    def _fake_redirect(t: type):
        if t in LLM_Simulator_Faker.redirects:
            return LLM_Simulator_Faker.fake(LLM_Simulator_Faker.redirects[t])
        else:
            raise ValueError(f"Unsupported type: {t}")

    @staticmethod
    def _fake_pydantic(t: type):
        assert issubclass(t, BaseModel)
        factory_dict = {}
        for name, field in t.model_fields.items():
            factory_dict[name] = LLM_Simulator_Faker.fake(field.type_)
        return t(**factory_dict)

    @staticmethod
    def _fake_datetime(t: type):
        assert t.__module__ == "datetime"
        if t.__name__ == "datetime":
            return LLM_Simulator_Faker.FAKER.date_time_this_decade()
        elif t.__name__ == "date":
            return LLM_Simulator_Faker.FAKER.date_this_decade()
        elif t.__name__ == "time":
            return LLM_Simulator_Faker.FAKER.time_object()
        else:
            raise ValueError(f"Unsupported type: {t}")

    @staticmethod
    def _fake_uuid(t: type):
        assert t.__module__ == "uuid" and t.__name__ == "UUID"
        return LLM_Simulator_Faker.FAKER.uuid4(cast_to=None)

    @staticmethod
    def _fake_typing(t: type):
        assert t.__module__ == "typing"
        if t.__name__ == "Any":
            t = object
        elif t.__name__ == "Union":
            t = LLM_Simulator_Faker.FAKER.random_element(elements=t.__args__)
        elif t.__name__ == "Optional":
            # t = LLM_Simulator_Faker.FAKER.random_element(elements=t.__args__)
            # If you don't want to ever return None, uncomment the above line
            t = LLM_Simulator_Faker.FAKER.random_element(elements=[type(None), t.__args__[0]])
        elif typing.get_origin(t) is not None:
            t = typing.get_origin(t)
        else:
            raise ValueError(f"Unsupported type: {t}")
        return LLM_Simulator_Faker.fake(t)

    @staticmethod
    def _fake_custom(t: type):
        raise ValueError("_fake_custom must be subclassed to be used")

    tryfuncs = [
        {"func": _fake_custom, "priority": 1000},
        {"func": _fake_paramgeneric, "priority": 500},
        {"func": _fake_pydantic, "priority": 100},
        {"func": _fake_basic, "priority": 100},
        {"func": _fake_datetime, "priority": 100},
        {"func": _fake_uuid, "priority": 100},
        {"func": _fake_typing, "priority": 100},
        {"func": _fake_redirect, "priority": -100},
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

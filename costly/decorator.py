import asyncio
import warnings
from functools import wraps
from dataclasses import dataclass
from copy import deepcopy
from typing import Callable, Any
from costly.utils import CostlyWarning
from costly.costlog import Costlog
from costly.simulators.llm_simulator_faker import LLM_Simulator_Faker
from costly.estimators.llm_api_estimation import LLM_API_Estimation
from inspect import signature, Parameter, iscoroutinefunction


@dataclass
class CostlyResponse:
    output: Any
    cost_info: dict[str, Any]


def costly(
    simulator: Callable = LLM_Simulator_Faker.simulate_llm_call,
    estimator: Callable = LLM_API_Estimation.get_cost_real,
    **param_mappings: dict[str, Callable],
):
    def decorator(func: Callable) -> Callable:
        sig = signature(func)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Get default value for cost_log if it's defined in the function
            # This is necessary for global cost_logs, due to Python's very weird
            # behaviour: although it allows you to update a global cost_log from
            # inside a function (because it's mutable), it doesn't allow you to
            # do so if it's passed as a *default argument*, only if it's explicitly
            # passed (which we can't do because the whole point of allowing it as a
            # global variable is to avoid passing it everywhere in your code explicitly).
            # This is very confusing to me because I thought the standard quirk of Python
            # was that it made defaults mutable, but like this is the only scenario when
            # it would be remotely useful, and it's not?
            # https://chatgpt.com/share/67005ec7-c7fc-8005-b61f-a3e2992519b1
            if "cost_log" not in kwargs:
                cost_log_param = sig.parameters.get("cost_log")
                if cost_log_param and cost_log_param.default is not Parameter.empty:
                    cost_log = cost_log_param.default
                else:
                    cost_log = None
            else:
                cost_log = kwargs.pop("cost_log")

            simulate = kwargs.pop("simulate", False)
            description = kwargs.pop("description", None)

            # Create a dictionary with default values
            options = {
                k: v.default
                for k, v in sig.parameters.items()
                if v.default is not Parameter.empty
            }

            # Update default kwargs with provided kwargs
            options.update(kwargs)

            # apply param_mappings
            costly_kwargs = options | {
                key: mapping(options) if callable(mapping) else options.get(mapping)
                for key, mapping in param_mappings.items()
            }

            if simulate:
                simulator_kwargs = {
                    k: v
                    for k, v in costly_kwargs.items()
                    if k in signature(simulator).parameters
                } | {"cost_log": cost_log, "description": description}
                return simulator(**simulator_kwargs)

            if cost_log is not None:
                async with cost_log.new_item_async() as (item, timer):
                    output = await func(*args, **kwargs)  # await the coroutine
                    cost_info = {}
                    if isinstance(output, CostlyResponse):
                        output, cost_info = output.output, output.cost_info
                    estimator_kwargs = (
                        {
                            k: v
                            for k, v in costly_kwargs.items()
                            if k in signature(estimator).parameters
                        }
                        | {
                            "output_string": output,
                            "description": description,
                            "timer": timer(),
                        }
                        | cost_info
                    )
                    cost_item = estimator(**estimator_kwargs)
                    item.update(cost_item)
            else:
                warnings.warn(
                    f"`cost_log` is None for the function:\n"
                    f"{func.__name__}\n"
                    f"with args:\n"
                    f"{args}\n"
                    f"and kwargs:\n"
                    f"{kwargs}\n"
                    "Maybe cost_log is not being passed through in some part of your logic?",
                    CostlyWarning,
                )
                output = await func(*args, **kwargs)
                if isinstance(output, CostlyResponse):
                    output, cost_info = output.output, output.cost_info
            return output

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if "cost_log" not in kwargs:
                cost_log_param = sig.parameters.get("cost_log")
                if cost_log_param and cost_log_param.default is not Parameter.empty:
                    cost_log = cost_log_param.default
                else:
                    cost_log = None
            else:
                cost_log = kwargs.pop("cost_log")

            simulate = kwargs.pop("simulate", False)
            description = kwargs.pop("description", None)

            # Get the function's signature
            sig = signature(func)

            # Create a dictionary with default values
            options = {
                k: v.default
                for k, v in sig.parameters.items()
                if v.default is not Parameter.empty
            }

            # Update default kwargs with provided kwargs
            options.update(kwargs)

            # apply param_mappings
            costly_kwargs = options | {
                key: mapping(options) if callable(mapping) else options.get(mapping)
                for key, mapping in param_mappings.items()
            }

            if simulate:
                simulator_kwargs = {
                    k: v
                    for k, v in costly_kwargs.items()
                    if k in signature(simulator).parameters
                } | {"cost_log": cost_log, "description": description}
                return simulator(**simulator_kwargs)

            if cost_log is not None:
                with cost_log.new_item() as (item, timer):
                    output = func(*args, **kwargs)  # call function normally
                    cost_info = {}
                    if isinstance(output, CostlyResponse):
                        output, cost_info = output.output, output.cost_info
                    estimator_kwargs = (
                        {
                            k: v
                            for k, v in costly_kwargs.items()
                            if k in signature(estimator).parameters
                        }
                        | {
                            "output_string": output,
                            "description": description,
                            "timer": timer(),
                        }
                        | cost_info
                    )
                    cost_item = estimator(**estimator_kwargs)
                    item.update(cost_item)
            else:
                warnings.warn(
                    f"`cost_log` is None for the function:\n"
                    f"{func.__name__}\n"
                    f"with args:\n"
                    f"{args}\n"
                    f"and kwargs:\n"
                    f"{kwargs}\n"
                    "Maybe cost_log is not being passed through in some part of your logic?",
                    CostlyWarning,
                )
                output = func(*args, **kwargs)
                if isinstance(output, CostlyResponse):
                    output, cost_info = output.output, output.cost_info
            return output

        return async_wrapper if iscoroutinefunction(func) else sync_wrapper

    return decorator

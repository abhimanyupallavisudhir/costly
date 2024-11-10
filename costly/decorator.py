import asyncio
from functools import wraps
from dataclasses import dataclass
from copy import deepcopy
from typing import Callable, Any
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
    disable_costly: bool = False,
    **param_mappings: dict[str, Callable],
):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if disable_costly:
                output = await func(*args, **kwargs)
                if isinstance(output, CostlyResponse):
                    output, cost_info = output.output, output.cost_info
                return output

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

            cost_log = options.pop("cost_log", None)
            simulate = options.pop("simulate", False)
            description = options.pop("description", None)

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

            if cost_log is None:
                output = await func(*args, **options)
                if isinstance(output, CostlyResponse):
                    output, cost_info = output.output, output.cost_info
            else:
                async with cost_log.new_item_async() as (item, timer):
                    output = await func(*args, **options)  # await the coroutine
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
            return output

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if disable_costly:
                output = func(*args, **kwargs)
                if isinstance(output, CostlyResponse):
                    output, cost_info = output.output, output.cost_info
                return output

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

            cost_log = options.pop("cost_log", None)
            simulate = options.pop("simulate", False)
            description = options.pop("description", None)

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

            if cost_log is None:
                output = func(*args, **options)
                if isinstance(output, CostlyResponse):
                    output, cost_info = output.output, output.cost_info
            else:
                with cost_log.new_item() as (item, timer):
                    output = func(*args, **options)  # call function normally
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
            return output

        return async_wrapper if iscoroutinefunction(func) else sync_wrapper

    return decorator

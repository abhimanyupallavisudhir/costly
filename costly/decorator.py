from functools import wraps
from dataclasses import dataclass
from copy import deepcopy
from typing import Callable, Any
from costly.costlog import Costlog
from costly.simulators.llm_simulator_faker import LLM_Simulator_Faker
from costly.estimators.llm_api_estimation import LLM_API_Estimation
from inspect import signature


@dataclass
class CostlyResponse:
    output: Any
    cost_info: dict[str, Any]


def costly(
    simulator: Callable = LLM_Simulator_Faker.simulate_llm_call,
    estimator: Callable = LLM_API_Estimation.get_cost_real,
    **param_mappings: dict[str, Callable]
):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            cost_log = kwargs.pop("cost_log", None)
            simulate = kwargs.pop("simulate", False)
            description = kwargs.pop("description", None)

            # Apply parameter mappings
            mapped_kwargs = kwargs | {
                key: mapping(kwargs) if callable(mapping) else kwargs.get(mapping)
                for key, mapping in param_mappings.items()
            }

            if simulate:
                simulator_params = signature(simulator).parameters
                simulator_kwargs = {
                    k: v for k, v in mapped_kwargs.items() if k in simulator_params
                } | {"cost_log": cost_log, "description": description}
                return simulator(**simulator_kwargs)

            if cost_log is not None:
                with cost_log.new_item() as (item, timer):
                    output = func(*args, **kwargs)
                    cost_info = {}
                    if isinstance(output, CostlyResponse):
                        output, cost_info = output.output, output.cost_info
                    estimator_params = signature(estimator).parameters
                    estimator_kwargs = (
                        {
                            k: v
                            for k, v in mapped_kwargs.items()
                            if k in estimator_params
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
                output = func(*args, **kwargs)
                if isinstance(output, CostlyResponse):
                    output, cost_info = output.output, output.cost_info
            return output

        return wrapper

    return decorator

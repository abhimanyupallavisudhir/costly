from functools import wraps
from copy import deepcopy
from typing import Callable
from costly.costlog import Costlog
from costly.simulators.llm_simulator_faker import LLM_Simulator_Faker
from costly.estimators.llm_api_estimation import LLM_API_Estimation
from inspect import signature


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

            # copy kwargs to be sent to original function because these
            # will be popped and replaced with input_string and model
            fun_kwargs = kwargs.copy()

            # Apply parameter mappings
            mapped_kwargs = kwargs | {
                key: mapping(kwargs) if callable(mapping) else kwargs.get(mapping)
                for key, mapping in param_mappings.items()
            }

            if simulate:
                simulator_params = signature(simulator).parameters
                simulator_kwargs = {
                    k: v for k, v in mapped_kwargs.items() if k in simulator_params
                }
                return simulator(
                    **simulator_kwargs, cost_log=cost_log, description=description
                )

            if cost_log is not None:
                with cost_log.new_item() as (item, timer):
                    output = func(*args, **fun_kwargs)
                    estimator_params = signature(estimator).parameters
                    estimator_kwargs = {
                        k: v for k, v in mapped_kwargs.items() if k in estimator_params
                    }
                    cost_item = estimator(
                        **estimator_kwargs,
                        output_string=output,
                        description=description,
                        timer=timer()
                    )
                    item.update(cost_item)
            else:
                output = func(*args, **fun_kwargs)

            return output

        return wrapper

    return decorator

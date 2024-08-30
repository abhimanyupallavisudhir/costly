from functools import wraps
from typing import Callable
from costly.costlog import Costlog
from costly.simulators.llm_simulator_faker import LLM_Simulator_Faker
from costly.estimators.llm_api_estimation import LLM_API_Estimation


def costly(
    simulator: Callable = LLM_Simulator_Faker.simulate_llm_call,
    estimator: Callable = LLM_API_Estimation.get_cost_real
):
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            cost_log = kwargs.pop('cost_log', None)
            simulate = kwargs.pop('simulate', False)
            description = kwargs.pop('description', None)

            if simulate:
                return simulator(
                    **kwargs,
                    cost_log=cost_log,
                    description=description
                )

            if cost_log is not None:
                with cost_log.new_item() as (item, timer):
                    output = func(*args, **kwargs)
                    cost_item = estimator(
                        **kwargs,
                        output_string=output,
                        description=description,
                        timer=timer()
                    )
                    item.update(cost_item)
            else:
                output = func(*args, **kwargs)

            return output

        return wrapper

    return decorator

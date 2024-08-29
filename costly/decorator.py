from functools import wraps
from costly.costlog import Costlog
from costly.simulators.llm_simulator_faker import LLM_Simulator_Faker
from costly.estimators.llm_api_estimation import LLM_API_Estimation


def costly(func):
    @wraps(func)
    def wrapper(
        input_string: str,
        model: str,
        cost_log: Costlog = None,
        simulate: bool = False,
        description: list[str] = None,
        *args,
        **kwargs
    ):
        if simulate:
            return LLM_Simulator_Faker.simulate_llm_call(
                input_string=input_string,
                model=model,
                response_model=str,
                cost_log=cost_log,
                description=description,
            )

        if cost_log is not None:
            with cost_log.new_item() as (item, timer):
                output_string = func(input_string, model, *args, **kwargs)
                cost_item = LLM_API_Estimation.get_cost_real(
                    model=model,
                    input_string=input_string,
                    output_string=output_string,
                    description=description,
                    timer=timer(),
                )
                item.update(cost_item)
        else:
            output_string = func(input_string, model, *args, **kwargs)

        return output_string

    return wrapper

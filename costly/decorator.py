from functools import wraps
from dataclasses import dataclass
from typing import Callable, Any
from costly.simulators.llm_simulator_faker import LLM_Simulator_Faker
from costly.estimators.llm_api_estimation import LLM_API_Estimation
from inspect import signature, Parameter, iscoroutinefunction


# Control parameters consumed by `costly` itself. They may be passed to any
# decorated function regardless of whether the function declares them.
_CONTROL_PARAMS = ("cost_log", "simulate", "description")


@dataclass
class CostlyResponse:
    output: Any
    cost_info: dict[str, Any]


def costly(
    simulator: Callable = LLM_Simulator_Faker.simulate_llm_call,
    estimator: Callable = LLM_API_Estimation.get_cost_real,
    disable_costly: bool = False,
    fast: bool = False,
    **param_mappings: dict[str, Callable],
):
    # Compute the accepted parameter names of the simulator/estimator once, since
    # they are fixed for a given decorator application.
    simulator_params = set(signature(simulator).parameters)
    estimator_params = set(signature(estimator).parameters)

    def decorator(func: Callable) -> Callable:
        # Cache signature-derived metadata once at decoration time instead of
        # recomputing it on every call.
        sig = signature(func)
        func_params = sig.parameters
        var_keyword_name = next(
            (
                name
                for name, p in func_params.items()
                if p.kind == Parameter.VAR_KEYWORD
            ),
            None,
        )

        def prepare(args: tuple, kwargs: dict):
            """Bind call arguments and split out costly's control parameters.

            Control parameters (cost_log/simulate/description) may be supplied
            even when the decorated function does not declare them, so they are
            removed before binding against the real signature.
            """
            kwargs = dict(kwargs)
            popped = {
                name: kwargs.pop(name)
                for name in _CONTROL_PARAMS
                if name not in func_params and name in kwargs
            }

            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            options = bound_args.arguments

            cost_log = popped.get("cost_log", options.pop("cost_log", None))
            simulate = popped.get("simulate", options.pop("simulate", False))
            description = popped.get("description", options.pop("description", None))

            # If the function accepts **kwargs, flatten those extra keyword
            # arguments back into the main options mapping.
            if var_keyword_name is not None:
                options.update(options.pop(var_keyword_name, {}))

            # apply param_mappings
            costly_kwargs = options | {
                key: mapping(options) if callable(mapping) else options.get(mapping)
                for key, mapping in param_mappings.items()
            }
            return options, costly_kwargs, cost_log, simulate, description

        def simulator_call(costly_kwargs, cost_log, description):
            simulator_kwargs = {
                k: v for k, v in costly_kwargs.items() if k in simulator_params
            } | {"cost_log": cost_log, "description": description, "fast": fast}
            return simulator(**simulator_kwargs)

        def build_estimator_kwargs(costly_kwargs, output, description, timer, cost_info):
            return (
                {k: v for k, v in costly_kwargs.items() if k in estimator_params}
                | {
                    "output_string": output,
                    "description": description,
                    "timer": timer,
                    "fast": fast,
                }
                | cost_info
            )

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if disable_costly:
                output = await func(*args, **kwargs)
                if isinstance(output, CostlyResponse):
                    output = output.output
                return output

            options, costly_kwargs, cost_log, simulate, description = prepare(
                args, kwargs
            )

            if simulate:
                return simulator_call(costly_kwargs, cost_log, description)

            if cost_log is None:
                output = await func(**options)
                if isinstance(output, CostlyResponse):
                    output = output.output
            else:
                async with cost_log.new_item_async() as (item, timer):
                    output = await func(**options)  # await the coroutine
                    cost_info = {}
                    if isinstance(output, CostlyResponse):
                        output, cost_info = output.output, output.cost_info
                    cost_item = estimator(
                        **build_estimator_kwargs(
                            costly_kwargs, output, description, timer(), cost_info
                        )
                    )
                    item.update(cost_item)
            return output

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if disable_costly:
                output = func(*args, **kwargs)
                if isinstance(output, CostlyResponse):
                    output = output.output
                return output

            options, costly_kwargs, cost_log, simulate, description = prepare(
                args, kwargs
            )

            if simulate:
                return simulator_call(costly_kwargs, cost_log, description)

            if cost_log is None:
                output = func(**options)
                if isinstance(output, CostlyResponse):
                    output = output.output
            else:
                with cost_log.new_item() as (item, timer):
                    output = func(**options)  # call function normally
                    cost_info = {}
                    if isinstance(output, CostlyResponse):
                        output, cost_info = output.output, output.cost_info
                    cost_item = estimator(
                        **build_estimator_kwargs(
                            costly_kwargs, output, description, timer(), cost_info
                        )
                    )
                    item.update(cost_item)
            return output

        return async_wrapper if iscoroutinefunction(func) else sync_wrapper

    return decorator

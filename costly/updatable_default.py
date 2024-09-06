from typing import Callable, Any
from functools import wraps
from dataclasses import dataclass


@dataclass
class DependentDefault:
    dep: Callable


def updatable_default(updator: Callable = lambda x, y: x | y, **defaults):
    """
    Ever wished that instead of *overriding* a function's default
    parameters with those supplied in your call, you could just
    "update" it with that supplied in your call? Something like
    f(y += 1)?

    You can do this with `updatable_default`. Instead of defining the default
    parameter of the function in the standard way, do it like this:

    @updatable_default(x = {"a": 1})
    def f(x):
        return x

    f() # returns {"a": 1}
    f(x = {"b": 2}) # returns {"a": 1, "b": 2}
    f(x = {"a": 3}) # returns {"a": 3}

    @updatable_default(updator = lambda x, y: x + y, y = 8)
    def f(y):
        return y

    f(3) # returns 11

    def update_recursive(source, overrides):
        for key, value in overrides.items():
            if isinstance(value, dict) and key in source and isinstance(source[key], dict):
                update_recursive(source[key], value)
            elif isinstance(value, list) and key in source and isinstance(source[key], list):
                source[key] = source[key] + value
            else:
                source[key] = value
        return source


    @updatable_default(
        post={
            "content": "Hello, World!",
            "OP": {
                "name": "John",
                "age": 30,
                "metadata": {
                    "account_created": "2024-01-01",
                    "banned": False,
                    "votes_cast": 1,
                },
            },
            "metadata": {"score": -2, "flags": ["spam"]},
        },
        updator=update_recursive,
    )
    def f(post):
        return post


    f(
        post={
            "OP": {
                "metadata": {"account_deleted": "2024-01-02", "banned": True},
            },
            "metadata": {"flags": ["deleted"]},
        },
    )

    ### returns:
    {
        "content": "Hello, World!",
        "OP": {
            "name": "John",
            "age": 30,
            "metadata": {
                "account_created": "2024-01-01",
                "banned": True,
                "votes_cast": 1,
                "account_deleted": "2024-01-02",
            },
        },
        "metadata": {"score": -2, "flags": ["spam", "deleted"]},
    }

    Stuff this can be useful for:

    - tracing
    - cumulative configurations
    - "biasing" or "damping" some kind of computation

    The default updator is `|`.

    You can also have a default be dependent on func, args and kwargs by
    setting it to an object DependentDefault(dep=lambda ...)
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for arg_name, default_value in defaults.items():
                if isinstance(default_value, DependentDefault):
                    default_value = default_value.dep(func, *args, **kwargs)
                if arg_name in kwargs:
                    kwargs[arg_name] = updator(default_value, kwargs[arg_name])
                # else:
                #     import inspect

                #     sig = inspect.signature(func)
                #     param_position = list(sig.parameters.keys()).index(arg_name)
                #     if param_position < len(args):
                #         args = list(args)
                #         args[param_position] = updator(
                #             default_value, args[param_position]
                #         )
                #         args = tuple(args)
                else:  # if arg_name not provided, add it and set it to default
                    kwargs[arg_name] = default_value
            return func(*args, **kwargs)

        return wrapper

    return decorator


def invoice(description: str = None):
    """
    Example usage:

    @invoice()
    def manufacture_cups(**kwargs):
        manufacture_steel(**kwargs)
        educate_potters(**kwargs)
        return

    @invoice("educating potters")
    def educate_potters(**kwargs):
        capture_enemies(**kwargs)
        return

    @invoice("manufacturing steel")
    def manufacture_steel(**kwargs):
        mine_iron(**kwargs)
        return

    @invoice("mining iron")
    def mine_iron(**kwargs):
        print(kwargs["description"])
        return

    @invoice("capturing enemies")
    def capture_enemies(**kwargs):
        print(kwargs["description"])
        return

    manufacture_cups()
    ### prints:
    [{'func': 'manufacture_cups', 'args': [], 'kwargs': {}}, 'manufacturing steel', 'mining iron']
    [{'func': 'manufacture_cups', 'args': [], 'kwargs': {}}, 'educating potters', 'capturing enemies']

    If description is not provided, it will default to the function's name
    and arguments like above.

    """
    if description is None:
        description = DependentDefault(
            dep=lambda func, *args, **kwargs: {
                "func": func.__name__,
                "args": list(args),
                "kwargs": kwargs,
            }
        )
    return updatable_default(
        description=description,
        updator=lambda default, new: (
            new + [default] if isinstance(new, list) else [new] + [default]
        ),
    )
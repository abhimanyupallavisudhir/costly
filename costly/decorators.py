from functools import wraps

def propagate(update_func, arg_name, *extra_args):
    """Decorator to update a passed argument."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if arg_name in kwargs:
                kwargs[arg_name] = update_func(kwargs[arg_name], *extra_args)
            else:
                # Find the position of the argument in the function signature
                import inspect
                sig = inspect.signature(func)
                param_keys = list(sig.parameters.keys())
                if arg_name in param_keys:
                    arg_position = param_keys.index(arg_name)
                else:
                    arg_position = None
                
                if arg_position is not None and arg_position < len(args):
                    args = list(args)
                    args[arg_position] = update_func(args[arg_position], *extra_args)
                    args = tuple(args)
                else:
                    # If the argument is not provided, we do not modify the args
                    pass
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def _update_description(description: list[str] | None, value: str) -> list[str]:
    if description is None:
        description = []
    return description + [value]

costable = lambda v: propagate(_update_description, 'description', v)
"""Decorator to pass a 'description' of a function."""


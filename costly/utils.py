from types import NoneType, GenericAlias, UnionType
from typing import Any, Optional, Union, get_origin, get_args, Sequence, Mapping

def isinstance_better(v, t: type) -> bool:

    origin = get_origin(t)
    args = get_args(t)

    if origin in (Union, UnionType):
        return any(isinstance_better(v, arg) for arg in get_args(t))

    if origin is None:
        return isinstance(v, t)
    if not isinstance(v, origin):
        return False
    if isinstance(v, Sequence) and not isinstance(v, (str, bytes)):
        return all(isinstance_better(item, args[0]) for item in v)
    elif isinstance(v, Mapping):
        return all(
            isinstance_better(k, args[0]) and isinstance_better(v, args[1])
            for k, v in v.items()
        )
    return None
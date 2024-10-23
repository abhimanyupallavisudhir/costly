from types import NoneType, GenericAlias, UnionType
from typing import Any, Optional, Union, get_origin, get_args, Sequence, Mapping
import json
from pydantic import BaseModel


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


def json_serializable(value):
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


def make_json_serializable(value):
    if isinstance(value, dict):
        return {k: make_json_serializable(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [make_json_serializable(v) for v in value]
    elif isinstance(value, tuple):
        return tuple(make_json_serializable(v) for v in value)
    elif isinstance(value, set):
        return set(make_json_serializable(v) for v in value)
    elif isinstance(value, BaseModel):
        return make_json_serializable(value.model_dump(mode="json"))
    elif not json_serializable(value):
        return str(value)
    return value

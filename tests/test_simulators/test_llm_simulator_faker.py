import pytest
from types import NoneType
from typing import Optional, Union, Any
from pydantic import BaseModel
from costly.simulators.llm_simulator_faker import LLM_Simulator_Faker
from costly.utils import isinstance_better
from tests.example_functions import FOOMODEL, BARMODEL

TYPES = {
    int,
    str,
    float,
    list,
    dict,
    tuple,
    set,
    list[str],
    list[int],
    list[float],
    dict[str, int],
    dict[str, float],
    dict[str, Optional[list]] | str | None,
    FOOMODEL,
    list[FOOMODEL],
    BARMODEL,
    NoneType,
    Union[int, str],
    float | str,
}


@pytest.mark.parametrize("t", TYPES)
def test_fake_type(t: type):
    x = LLM_Simulator_Faker.fake(t)
    assert isinstance_better(x, t), f"Expected {t}, got {type(x)}"

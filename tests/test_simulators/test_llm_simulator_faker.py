import pytest
from types import NoneType
from typing import Optional, Union, Any
from pydantic import BaseModel
from costly.simulators.llm_simulator_faker import LLM_Simulator_Faker
from costly.utils import isinstance_better
from costly.costlog import Costlog
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


def test_simulate_llm_probs():
    return_probs_for = ["option1", "option2", "option3"]
    costlog = Costlog()
    
    probs = LLM_Simulator_Faker.simulate_llm_probs(
        return_probs_for=return_probs_for,
        input_string="Test input",
        model="gpt-3.5-turbo",
        cost_log=costlog
    )
    
    assert isinstance(probs, dict)
    assert set(probs.keys()) == set(return_probs_for)
    assert sum(probs.values()) == pytest.approx(1.0)
    
    for value in probs.values():
        assert 0 <= value <= 1
    
    assert len(costlog.items) == 1
    cost_item = costlog.items[0]
    assert "input_tokens" in cost_item
    assert "output_tokens_min" in cost_item
    assert "output_tokens_max" in cost_item
    assert cost_item["output_tokens_min"] == 0
    assert cost_item["output_tokens_max"] == max(len(key) for key in return_probs_for)

import pytest
import time
import jsonlines
from costly.costlog import Costlog
from pathlib import Path

def test_costlog_new_item():
    costlog = Costlog(mode="memory", totals_keys={"time"})
    with costlog.new_item() as (item, timer):
        time.sleep(0.35)
        item.update({"Hi": "Hello", "time": timer()})
    assert costlog.items == [{"Hi": "Hello", "time": pytest.approx(0.35, abs=0.01)}]
    assert costlog.totals["time"] == pytest.approx(0.35, abs=0.01)


def test_costlog_new_item_multiple():
    costlog = Costlog(mode="memory", totals_keys={"time"})
    with costlog.new_item() as (item, timer):
        time.sleep(0.08)
        item.update({"Hi": "Hello", "time": timer()})
    with costlog.new_item() as (item, timer):
        time.sleep(0.12)
        item.update({"Hi": "Hello", "time": timer()})
    assert costlog.items == [
        {"Hi": "Hello", "time": pytest.approx(0.08, abs=0.01)},
        {"Hi": "Hello", "time": pytest.approx(0.12, abs=0.01)},
    ]
    assert costlog.totals["time"] == pytest.approx(0.20, abs=0.01)

def test_costlog_jsonl():
    costlog = Costlog(mode="jsonl", totals_keys={"time"}, path=Path("tests", "test_costlog.jsonl"))
    with costlog.new_item() as (item, timer):
        time.sleep(0.08)
        item.update({"Hi": "Hello", "time": timer()})
    with costlog.new_item() as (item, timer):
        time.sleep(0.12)
        item.update({"Hi": "Hello", "time": timer()})
    assert costlog.totals["time"] == pytest.approx(0.20, abs=0.01)
    with open(costlog.path, "r") as f:
        items = [item for item in jsonlines.Reader(f)]
    assert items == [
        {"Hi": "Hello", "time": pytest.approx(0.08, abs=0.01)},
        {"Hi": "Hello", "time": pytest.approx(0.12, abs=0.01)},
    ]    
        
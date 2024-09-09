import pytest
import time
import asyncio
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
    costlog = Costlog(mode="jsonl", totals_keys={"time"}, path=Path("tests", "test_costlog.jsonl"), overwrite=True)
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

@pytest.mark.asyncio
async def test_costlog_new_item_async():
    costlog = Costlog(mode="memory", totals_keys={"time"})
    async with costlog.new_item_async() as (item, timer):
        await asyncio.sleep(0.35)
        item.update({"Hi": "Hello", "time": timer()})
    assert costlog.items == [{"Hi": "Hello", "time": pytest.approx(0.35, abs=0.01)}]
    assert costlog.totals["time"] == pytest.approx(0.35, abs=0.01)

@pytest.mark.asyncio
async def test_costlog_new_item_async_multiple():
    costlog = Costlog(mode="memory", totals_keys={"time"})
    async with costlog.new_item_async() as (item, timer):
        await asyncio.sleep(0.08)
        item.update({"Hi": "Hello", "time": timer()})
    async with costlog.new_item_async() as (item, timer):
        await asyncio.sleep(0.12)
        item.update({"Hi": "Hello", "time": timer()})
    assert costlog.items == [
        {"Hi": "Hello", "time": pytest.approx(0.08, abs=0.01)},
        {"Hi": "Hello", "time": pytest.approx(0.12, abs=0.01)},
    ]
    assert costlog.totals["time"] == pytest.approx(0.20, abs=0.01)

@pytest.mark.asyncio
async def test_costlog_jsonl_async():
    costlog = Costlog(mode="jsonl", totals_keys={"time"}, path=Path("tests", "test_costlog_async.jsonl"), overwrite=True)
    async with costlog.new_item_async() as (item, timer):
        await asyncio.sleep(0.08)
        item.update({"Hi": "Hello", "time": timer()})
    async with costlog.new_item_async() as (item, timer):
        await asyncio.sleep(0.12)
        item.update({"Hi": "Hello", "time": timer()})
    assert costlog.totals["time"] == pytest.approx(0.20, abs=0.01)
    with open(costlog.path, "r") as f:
        items = [item for item in jsonlines.Reader(f)]
    assert items == [
        {"Hi": "Hello", "time": pytest.approx(0.08, abs=0.01)},
        {"Hi": "Hello", "time": pytest.approx(0.12, abs=0.01)},
    ]

@pytest.mark.asyncio
async def test_costlog_async_parallel():
    costlog = Costlog(mode="memory", totals_keys={"time"})
    
    async def task(sleep_time):
        async with costlog.new_item_async() as (item, timer):
            await asyncio.sleep(sleep_time)
            item.update({"time": timer()})
    
    await asyncio.gather(
        task(0.1),
        task(0.2),
        task(0.3)
    )
    
    assert len(costlog.items) == 3
    assert costlog.items[0]["time"] == pytest.approx(0.1, abs=0.01)
    assert costlog.items[1]["time"] == pytest.approx(0.2, abs=0.01)
    assert costlog.items[2]["time"] == pytest.approx(0.3, abs=0.01)
    assert costlog.totals["time"] == pytest.approx(0.6, abs=0.01)

@pytest.mark.asyncio
async def test_costlog_async_mixed():
    costlog = Costlog(mode="memory", totals_keys={"time"})
    
    with costlog.new_item() as (item, timer):
        time.sleep(0.1)
        item.update({"type": "sync", "time": timer()})
    
    async with costlog.new_item_async() as (item, timer):
        await asyncio.sleep(0.2)
        item.update({"type": "async", "time": timer()})
    
    assert len(costlog.items) == 2
    assert costlog.items[0] == {"type": "sync", "time": pytest.approx(0.1, abs=0.01)}
    assert costlog.items[1] == {"type": "async", "time": pytest.approx(0.2, abs=0.01)}
    assert costlog.totals["time"] == pytest.approx(0.3, abs=0.01)


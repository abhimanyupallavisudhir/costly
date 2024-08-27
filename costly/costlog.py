import os
import datetime
import jsonlines
import time
from dataclasses import dataclass
from typing import Literal
from contextlib import contextmanager
from pathlib import Path


class Costlog:

    def __init__(
        self,
        path: Path = None,
        mode: Literal["jsonl", "memory"] = "jsonl",
        totals_keys: set[str] = None,
    ):
        match mode:
            case "jsonl":
                if path is None:
                    self.path = Path(
                        os.getcwd(),
                        ".costly",
                        f"costlog_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jsonl",
                    )
                else:
                    self.path = path
                if not self.path.exists():
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    with open(self.path, "w") as f:
                        jsonlines.Writer(f)
            case "memory":
                self.items = []
            case _:
                raise ValueError(f"Invalid mode: {mode}")
        self.mode = mode
        if totals_keys is None:
            totals_keys = {"cost_min", "cost_max", "time_min", "time_max", "calls"}
        self.totals_keys = totals_keys
        self.totals = {key: 0.0 for key in totals_keys}

    def append(self, **kwargs):
        match self.mode:
            case "jsonl":
                with jsonlines.open(self.path, "a") as f:
                    f.write(kwargs)
            case "memory":
                self.items.append(kwargs)
        for key in self.totals:
            self.totals[key] += kwargs.get(key, 0.0)

    @contextmanager
    def new_item(self):
        item = {}
        t1 = time.perf_counter()
        yield item, lambda: time.perf_counter() - t1
        self.append(**item)

    def totals_from_items(self):
        match self.mode:
            case "jsonl":
                with jsonlines.open(self.path) as f:
                    items = list(f)
            case "memory":
                items = self.items
            case _:
                raise ValueError(f"Invalid mode: {self.mode}")
        return {key: sum(item[key] for item in items) for key in self.totals_keys}

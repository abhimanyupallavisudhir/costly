import os
import datetime
import json
import jsonlines
import time
import asyncio
from dataclasses import dataclass
from typing import Literal
from contextlib import contextmanager, asynccontextmanager
from pathlib import Path
from costly.utils import make_json_serializable


class Costlog(dict):

    def __init__(
        self,
        path: Path | str = None,
        mode: Literal["jsonl", "memory"] = "memory",
        totals_keys: set[str] = None,
        discard_extras: bool = True,
        overwrite: bool = None,
    ):
        """
        Arguments:
            path: Path to the costlog file, if in jsonl mode. If None, the costlog will be stored in
                the current working directory in a folder called ".costly".
            mode: "jsonl" or "memory". "jsonl" will store the costlog in a jsonl file,
                "memory" will store the costlog in memory which may not be suitable for large logs.
            totals_keys: The keys to store in the totals.
            discard_extras: If True, extra keys in the costlog items will be discarded.
            overwrite: If the costlog file already exists, overwrite it. If None, the user will be asked. Only relevant if mode is "jsonl".
        """
        match mode:
            case "jsonl":
                if path is None:
                    self.path = Path(
                        os.getcwd(),
                        ".costly",
                        f"costlog_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jsonl",
                    )
                else:
                    self.path = Path(path)
                if self.path.exists():
                    if overwrite is None:
                        overwrite = input(
                            f"Costlog file {self.path} already exists. Overwrite? (y/n) "
                        )
                    if overwrite not in ["y", True]:
                        raise FileExistsError(
                            f"Costlog file {self.path} already exists."
                        )
                    else:
                        self.path.unlink()
                else:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    with open(self.path, "w") as f:
                        jsonlines.Writer(f)
                self.totals_path = self.path.with_suffix(".totals.json")
                self.totals_by_model_path = self.path.with_suffix(
                    ".totals_by_model.json"
                )
            case "memory":
                self.items = []
            case _:
                raise ValueError(f"Invalid mode: {mode}")
        self.mode = mode
        if totals_keys is None:
            totals_keys = {
                "cost_min",
                "cost_max",
                "time_min",
                "time_max",
                "calls",
                "input_tokens",
                "output_tokens_min",
                "output_tokens_max",
            }
        self.totals_keys = totals_keys
        self.acceptable_keys = totals_keys | {"model", "simulated", "cost", "time"}
        self.discard_extras = discard_extras
        self.totals = {key: 0.0 for key in totals_keys}
        self.totals_by_model = {}
        super().__init__(self.totals)

    def append(self, **kwargs):
        match self.mode:
            case "jsonl":
                with jsonlines.open(self.path, "a") as f:
                    f.write(make_json_serializable(kwargs))
            case "memory":
                self.items.append(kwargs)
        for key in self.totals:
            self.totals[key] += kwargs.get(key, 0.0)
            if kwargs.get("model", "__UNKNOWN__") not in self.totals_by_model:
                self.totals_by_model[kwargs.get("model", "__UNKNOWN__")] = {
                    key: 0.0 for key in self.totals_keys
                }
            self.totals_by_model[kwargs.get("model", "__UNKNOWN__")][key] += kwargs.get(
                key, 0.0
            )
        if self.mode == "jsonl":
            with open(self.totals_path, "w") as f:
                json.dump(self.totals, f, indent=4)
            with open(self.totals_by_model_path, "w") as f:
                json.dump(self.totals_by_model, f, indent=4)

    @contextmanager
    def new_item(self):
        item = {}
        t1 = time.perf_counter()
        yield item, lambda: time.perf_counter() - t1
        if self.discard_extras:
            item = {key: item[key] for key in item if key in self.acceptable_keys}
        self.append(**item)

    @asynccontextmanager
    async def new_item_async(self):
        item = {}
        t1 = time.perf_counter()
        yield item, lambda: time.perf_counter() - t1
        if self.discard_extras:
            item = {key: item[key] for key in item if key in self.acceptable_keys}
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

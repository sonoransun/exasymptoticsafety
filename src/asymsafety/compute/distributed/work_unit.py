"""Work unit definition for distributed and crowd computing.

A work unit packages a computation task (fixed point search, flow
integration, etc.) in a JSON-serializable format that can be sent
to any worker via HTTP, message queue, or BOINC.
"""

from __future__ import annotations
import json
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any

import numpy as np


class TaskType(str, Enum):
    FIND_FIXED_POINT = "find_fixed_point"
    INTEGRATE_FLOW = "integrate_flow"
    EVALUATE_BATCH = "evaluate_batch"


@dataclass
class WorkUnit:
    """A single unit of computation."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_type: TaskType = TaskType.FIND_FIXED_POINT
    system_data: dict = field(default_factory=dict)  # serialized BetaFunctionSystem
    parameters: dict = field(default_factory=dict)    # task-specific parameters
    status: str = "pending"  # pending, running, complete, failed
    result: Any = None
    error: str | None = None

    def to_json(self) -> str:
        d = asdict(self)
        d["task_type"] = self.task_type.value
        if isinstance(d.get("result"), np.ndarray):
            d["result"] = {"__ndarray__": True, "data": d["result"].tolist()}
        return json.dumps(d)

    @classmethod
    def from_json(cls, json_str: str) -> "WorkUnit":
        d = json.loads(json_str)
        d["task_type"] = TaskType(d["task_type"])
        if isinstance(d.get("result"), dict) and d["result"].get("__ndarray__"):
            d["result"] = np.array(d["result"]["data"])
        return cls(**d)


def create_fp_work_unit(system_data: dict, guess: dict[str, float], tol: float = 1e-10) -> WorkUnit:
    """Create a work unit for a single fixed point search."""
    return WorkUnit(
        task_type=TaskType.FIND_FIXED_POINT,
        system_data=system_data,
        parameters={"guess": guess, "tol": tol},
    )


def create_flow_work_unit(system_data: dict, initial_conditions: dict[str, float],
                           t_span: tuple[float, float] = (-10, 10)) -> WorkUnit:
    """Create a work unit for flow integration."""
    return WorkUnit(
        task_type=TaskType.INTEGRATE_FLOW,
        system_data=system_data,
        parameters={"initial_conditions": initial_conditions, "t_span": list(t_span)},
    )


def create_batch_work_unit(system_data: dict, points: np.ndarray) -> WorkUnit:
    """Create a work unit for batch beta evaluation."""
    return WorkUnit(
        task_type=TaskType.EVALUATE_BATCH,
        system_data=system_data,
        parameters={"points": points.tolist()},
    )

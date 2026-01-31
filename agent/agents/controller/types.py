from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from typing_extensions import TypedDict

class Task(TypedDict):
    task_id: str
    kind: Literal["primary", "recall"]
    agent: str
    desc: str
    inputs: Dict[str, Any]
    status: Literal["todo", "running", "done", "failed"]

class Plan(TypedDict):
    goal: str
    tasks_primary: List[Task]
    tasks_recall: List[Task]
    stop_rule: str

class ApprovalPayload(TypedDict):
    stage: Literal["plan", "answer"]
    bundle: Dict[str, Any]  # 你前端展示的1/2/3信息，这里后端先生成结构

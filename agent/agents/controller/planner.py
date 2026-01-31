from __future__ import annotations
from typing import Any, Dict
from .types import Plan, Task

class Planner:
    """之后接入 LLM 规划，也可以规则化。这里先给最小可跑通 stub。"""

    def draft_plan(self, state: Dict[str, Any]) -> Plan:
        q = state.get("query", "")
        # TODO: 实现：意图识别/路径规划/拆任务
        t1: Task = {
            "task_id": "t_search",
            "kind": "primary",
            "agent": "Search_Agent",
            "desc": f"检索与问题相关的海洋新闻：{q}",
            "inputs": {"query": q},
            "status": "todo",
        }
        plan: Plan = {
            "goal": f"回答：{q}",
            "tasks_primary": [t1],
            "tasks_recall": [
                {
                    "task_id": "r_verify",
                    "kind": "recall",
                    "agent": "User_Recall",
                    "desc": "回顾：是否需要限定海域/时间范围/可信来源？",
                    "inputs": {},
                    "status": "todo",
                }
            ],
            "stop_rule": "当 primary 任务全部完成且用户批准最终答案后停止",
        }
        return plan

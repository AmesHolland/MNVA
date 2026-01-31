from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal
from typing_extensions import TypedDict, Annotated
from operator import add

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

def dict_merge(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """浅合并：右侧覆盖左侧。用于 tasks/findings 等字典累积。"""
    merged = dict(left or {})
    merged.update(right or {})
    return merged

class OceanState(TypedDict, total=False):
    # --- 会话标识（建议每轮都带着） ---
    user_id: str
    thread_id: str
    turn_id: int
    run_id: str

    # --- 对话消息（自动 append） ---
    messages: Annotated[List[AnyMessage], add_messages]

    # --- 当前输入与约束 ---
    query: str
    intent: str
    constraints: Dict[str, Any]   # time_range/geo/entities/topics/sources/lang/...

    # --- 规划与执行控制 ---
    plan: Dict[str, Any]          # {goal, tasks_primary, tasks_recall, stop_rule, ...}
    plan_version: int
    approval: Dict[str, Any]      # {stage, decision, edits, ...} 最近一次审批恢复值

    tasks: Annotated[Dict[str, Any], dict_merge]   # task_id -> task dict（方便局部更新）
    current_task_id: Optional[str]
    done: bool
    stop_reason: Optional[str]

    # --- RAG / 新闻数据（尽量轻量）---
    news_list: List[Dict[str, Any]]   # meta list（不要放全文）
    news_selected_ids: List[str]

    # --- 中间发现与证据 ---
    findings: Annotated[Dict[str, Any], dict_merge]       # topics/entities/flows/turning_points...
    evidence_index: Annotated[Dict[str, Any], dict_merge] # claim_id -> [news_id...]

    # --- 可视化参数：建议放 spec + data_key 指针 ---
    viz_buffer: Annotated[List[Dict[str, Any]], add]      # list append
    viz_key: Optional[str]                                # 最终落地后的 key

    # --- 观测与错误 ---
    tool_trace: Annotated[List[Dict[str, Any]], add]
    errors: Annotated[List[Dict[str, Any]], add]

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt

from agent.state import OceanState
from agent.config.app_config import AppConfig
from agent.agents.registry import build_agent_registry
from agent.agents.controller.planner import Planner

def _task_map_from_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    tasks = {}
    for t in plan.get("tasks_primary", []):
        tasks[t["task_id"]] = t
    for t in plan.get("tasks_recall", []):
        tasks[t["task_id"]] = t
    return tasks

def _next_todo_primary(tasks: Dict[str, Any]) -> Optional[str]:
    # 只执行 primary；recall 任务用于展示/提醒（后续你也可以把它做成 interrupt）
    for tid, t in tasks.items():
        if t.get("kind") == "primary" and t.get("status") == "todo":
            return tid
    return None

class OrchestratorGraphBuilder:
    def __init__(self, app_config: AppConfig):
        self.cfg = app_config
        self.registry = build_agent_registry()
        self.planner = Planner()

    # ---------------- Nodes ----------------

    def ingest_input(self, state: OceanState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """把 query 写入 messages（如果外部没有显式传 messages）。"""
        q = state.get("query", "")
        if not state.get("messages"):
            return {"messages": [HumanMessage(content=q)]}
        return {}

    def intent_router(self, state: OceanState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """后续换成真正的意图识别。这里先 stub。"""
        q = (state.get("query") or "").lower()
        intent = "mixed"
        intent = "search"
        # if "时间" in q or "trend" in q:
        #     intent = "time_evolution"
        # if "实体" in q or "ship" in q:
        #     intent = "entity_tracking"
        return { "intent": intent }

    def plan_draft(self, state: OceanState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        plan = self.planner.draft_plan(state)
        tasks_map = _task_map_from_plan(plan)
        return {
            "plan": plan,
            "plan_version": int(state.get("plan_version", 0)) + 1,
            "tasks": tasks_map,
            "done": False,
            "stop_reason": None,
            "messages": [AIMessage(content=f"[planner] drafted plan v{int(state.get('plan_version',0))+1}")],
        }

    def hitl_plan_approval(self, state: OceanState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """动态 interrupt：把 plan bundle 抛给外部；resume 值会作为 interrupt() 返回值。"""
        plan = state.get("plan", {})
        tasks = state.get("tasks", {})
        bundle = {
            "stage": "plan",
            "plan": plan,
            "tasks_primary": [t for t in tasks.values() if t.get("kind") == "primary"],
            "tasks_recall": [t for t in tasks.values() if t.get("kind") == "recall"],
            "logic_chain": plan.get("logic_chain", []),
            "viz_recommendations": plan.get("viz_recommendations", []),
        }
        decision = interrupt({"stage": "plan", "bundle": bundle})
        # decision 期望格式例子：
        # {"approved": true} 或 {"approved": false, "edits": {...}} 或 {"approved": false, "feedback": "..."}
        return {"approval": {"stage": "plan", **(decision or {})}}

    def route_after_plan(self, state: OceanState) -> str:
        ap = state.get("approval", {})
        if ap.get("stage") == "plan" and ap.get("approved") is True:
            return "execute_next_task"
        return "apply_plan_feedback"

    def apply_plan_feedback(self, state: OceanState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """根据用户修改意见更新 constraints / query / 规划输入，然后回到 plan_draft。"""
        ap = state.get("approval", {})
        edits = ap.get("edits") or {}
        feedback = ap.get("feedback")
        updates: Dict[str, Any] = {}

        if isinstance(edits, dict) and edits:
            # 你可以约定 edits 里允许更新 constraints/query 等
            if "query" in edits:
                updates["query"] = edits["query"]
            if "constraints" in edits:
                updates["constraints"] = {**(state.get("constraints") or {}), **edits["constraints"]}

        if feedback:
            updates["messages"] = [AIMessage(content=f"[plan_feedback] {feedback}")]
        return updates

    def execute_next_task(self, state: OceanState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        tasks = state.get("tasks") or {}
        tid = _next_todo_primary(tasks)
        if tid is None:
            return {"done": True, "stop_reason": "all_primary_tasks_done"}

        task = dict(tasks[tid])
        task["status"] = "running"

        # 更新 task 状态（dict_merge reducer 会合并）
        updates: Dict[str, Any] = {"current_task_id": tid, "tasks": {tid: task}}

        agent_name = task.get("agent")
        agent = self.registry.get(agent_name)
        if agent is None:
            task["status"] = "failed"
            return {
                **updates,
                "tasks": {tid: task},
                "errors": [{"task_id": tid, "error": f"Unknown agent: {agent_name}"}],
            }

        # 调用子 Agent（你在子 Agent 内部实现 RAG/tool 调用）
        child_out = agent.invoke(state, config=config)  # 约定：返回部分 state 更新
        print(child_out)
        # 任务完成
        task["status"] = "done"
        updates["tasks"] = {tid: task}

        # 合并子 agent 输出
        merged = dict(child_out or {})
        merged.update(updates)
        return merged

    def should_continue(self, state: OceanState) -> str:
        if state.get("done") is True:
            return "draft_answer"
        # safety: 限制最大 step，避免死循环
        # （你也可以用 state 计数器 reducer 来实现更严谨）
        return "execute_next_task"

    def draft_answer(self, state: OceanState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """最终文本生成：这里先 stub。你之后会用 evidence/findings/news_candidates 生成可验证回答。"""
        q = state.get("query", "")
        reason = state.get("stop_reason")
        return {
            "messages": [AIMessage(content=f"[draft_answer stub] q={q}, stop_reason={reason}")],
            "answer": f"[stub answer] {q}",
        }

    def hitl_answer_approval(self, state: OceanState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        answer = state.get("answer", "")
        bundle = {
            "stage": "answer",
            "answer": answer,
            "evidence_index": state.get("evidence_index", {}),
            "news_candidates": state.get("news_candidates", [])[:10],
            "viz_buffer": state.get("viz_buffer", []),
        }
        decision = interrupt({"stage": "answer", "bundle": bundle})
        return {"approval": {"stage": "answer", **(decision or {})}}

    def route_after_answer(self, state: OceanState) -> str:
        ap = state.get("approval", {})
        if ap.get("stage") == "answer" and ap.get("approved") is True:
            return "finalize"
        return "apply_answer_feedback"

    def apply_answer_feedback(self, state: OceanState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """用户不满意最终答案：可以选择 (1) 只重写答案 (2) 追加任务再跑。
        这里给你一个默认策略：
          - 若 ap.edits.answer 直接覆盖 answer，然后再走一次审批
          - 若 ap.add_tasks 存进 tasks（todo），回到 execute_next_task
        """
        ap = state.get("approval", {})
        edits = ap.get("edits") or {}
        add_tasks = ap.get("add_tasks") or []

        updates: Dict[str, Any] = {}
        if isinstance(edits, dict) and "answer" in edits:
            updates["answer"] = edits["answer"]
            updates["messages"] = [AIMessage(content="[apply_answer_feedback] answer edited")]
            return updates  # 直接回到 hitl_answer_approval

        if add_tasks:
            # 把新任务塞进 tasks（注意 task_id 不能重复）
            tasks_patch = {}
            for t in add_tasks:
                if "task_id" not in t:
                    t = dict(t)
                    t["task_id"] = f"t_{uuid4().hex[:8]}"
                t.setdefault("kind", "primary")
                t.setdefault("status", "todo")
                tasks_patch[t["task_id"]] = t
            updates["tasks"] = tasks_patch
            updates["done"] = False
            updates["stop_reason"] = None
            updates["messages"] = [AIMessage(content=f"[apply_answer_feedback] added {len(tasks_patch)} tasks")]
            return updates

        # fallback：回到规划
        return {"messages": [AIMessage(content="[apply_answer_feedback] fallback -> replan")]}

    def finalize(self, state: OceanState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """把 viz_buffer 落地（这里先用一个简单 key；你可换 Redis/DB）。"""
        # 生产里：写入 Redis/DB，返回 viz_key
        viz_key = f"viz:{uuid4().hex}"
        return {
            "viz_key": viz_key,
            "messages": [AIMessage(content=f"[finalize] viz_key={viz_key}")],
        }

    # ---------------- Build graph ----------------

    def build(self):
        g = StateGraph(OceanState)

        g.add_node("ingest_input", self.ingest_input)
        g.add_node("intent_router", self.intent_router)

        g.add_node("plan_draft", self.plan_draft)
        g.add_node("hitl_plan_approval", self.hitl_plan_approval)
        g.add_node("apply_plan_feedback", self.apply_plan_feedback)

        g.add_node("execute_next_task", self.execute_next_task)
        g.add_node("draft_answer", self.draft_answer)
        g.add_node("hitl_answer_approval", self.hitl_answer_approval)
        g.add_node("apply_answer_feedback", self.apply_answer_feedback)
        g.add_node("finalize", self.finalize)

        g.add_edge(START, "ingest_input")
        g.add_edge("ingest_input", "intent_router")
        g.add_edge("intent_router", "plan_draft")
        g.add_edge("plan_draft", "hitl_plan_approval")

        g.add_conditional_edges("hitl_plan_approval", self.route_after_plan, {
            "execute_next_task": "execute_next_task",
            "apply_plan_feedback": "apply_plan_feedback",
        })
        g.add_edge("apply_plan_feedback", "plan_draft")

        g.add_conditional_edges("execute_next_task", self.should_continue, {
            "execute_next_task": "execute_next_task",
            "draft_answer": "draft_answer",
        })

        g.add_edge("draft_answer", "hitl_answer_approval")
        g.add_conditional_edges("hitl_answer_approval", self.route_after_answer, {
            "finalize": "finalize",
            "apply_answer_feedback": "apply_answer_feedback",
        })

        # apply_answer_feedback：如果只改 answer，就回到审批；如果加任务，就回到执行；否则回到 plan
        def _route_feedback(state: OceanState) -> str:
            ap = state.get("approval", {})
            edits = ap.get("edits") or {}
            add_tasks = ap.get("add_tasks") or []
            if isinstance(edits, dict) and "answer" in edits:
                return "hitl_answer_approval"
            if add_tasks:
                return "execute_next_task"
            return "plan_draft"

        g.add_conditional_edges("apply_answer_feedback", _route_feedback, {
            "hitl_answer_approval": "hitl_answer_approval",
            "execute_next_task": "execute_next_task",
            "plan_draft": "plan_draft",
        })

        g.add_edge("finalize", END)
        return g

from __future__ import annotations

from typing import Any, Dict, Optional

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from agent.config.app_config import AppConfig
from agent.memory.checkpoint import build_checkpointer_sqlite
from agent.memory.store import build_store_inmemory, UserMemory
from agent.agents.controller.graph import OrchestratorGraphBuilder

class MainAgent:
    """对外暴露：run_turn / resume_turn / get_thread_state"""

    def __init__(self, app_config: Optional[AppConfig] = None):
        self.cfg = app_config or AppConfig()

        # 1) checkpointer：线程级短期记忆/可恢复
        self.checkpointer = build_checkpointer_sqlite(self.cfg.checkpoint_db_path)

        # 2) store：跨线程长期记忆（可选）
        self.store = build_store_inmemory()
        self.user_memory = UserMemory(self.store)

        # 3) graph
        builder = OrchestratorGraphBuilder(self.cfg).build()
        self.graph = builder.compile(
            checkpointer=self.checkpointer,
            store=self.store,
        )

    def _config(self, thread_id: str, user_id: Optional[str] = None) -> RunnableConfig:
        # thread_id 是 checkpointer 的主键；没有它无法持久化/恢复 :contentReference[oaicite:7]{index=7}
        configurable: Dict[str, Any] = {"thread_id": thread_id}
        if user_id:
            configurable["user_id"] = user_id
        return {"configurable": configurable}

    def run_turn(
        self,
        *,
        user_id: str,
        thread_id: str,
        query: str,
        constraints: Optional[Dict[str, Any]] = None,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        init_state: Dict[str, Any] = {
            "user_id": user_id,
            "thread_id": thread_id,
            "query": query,
            "constraints": constraints or {},
            "done": False,
            "tool_trace": [],
            "errors": [],
            "viz_buffer": [],
        }
        if extra_state:
            init_state.update(extra_state)

        return self.graph.invoke(init_state, config=self._config(thread_id, user_id))

    def resume_turn(
        self,
        *,
        user_id: str,
        thread_id: str,
        resume_payload: Any,
    ) -> Dict[str, Any]:
        # resume_payload 会成为 interrupt() 的返回值 :contentReference[oaicite:8]{index=8}
        return self.graph.invoke(Command(resume=resume_payload), config=self._config(thread_id, user_id))

    def get_thread_state(self, *, user_id: str, thread_id: str):
        # 返回 StateSnapshot（含 values/next/tasks 等）
        return self.graph.get_state(self._config(thread_id, user_id))

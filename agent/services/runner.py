from __future__ import annotations
from typing import Any, Dict, Optional
from agent.agents.controller.main_agent import MainAgent

class AgentRunner:
    def __init__(self):
        self.agent = MainAgent()

    def start_turn(self, user_id: str, thread_id: str, query: str, constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        result = self.agent.run_turn(user_id=user_id, thread_id=thread_id, query=query, constraints=constraints)
        return self._normalize(result)

    def resume_turn(self, user_id: str, thread_id: str, resume_payload: Any) -> Dict[str, Any]:
        result = self.agent.resume_turn(user_id=user_id, thread_id=thread_id, resume_payload=resume_payload)
        return self._normalize(result)

    def get_state(self, user_id: str, thread_id: str) -> Any:
        return self.agent.get_thread_state(user_id=user_id, thread_id=thread_id)

    def _normalize(self, result: Dict[str, Any]) -> Dict[str, Any]:
        # 统一返回格式：如果中断，__interrupt__ 会出现在 result 里
        if "__interrupt__" in result:
            return {"status": "NEED_APPROVAL", "interrupts": result["__interrupt__"], "state": result}
        return {"status": "DONE", "state": result}

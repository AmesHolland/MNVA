from __future__ import annotations
from typing import Any, Dict, Optional, Tuple

def build_store_inmemory():
    """
    LangGraph 的长期记忆 store（跨 thread），默认 InMemory。
    后续会替换为 PostgresStore / RedisStore。
    """
    from langgraph.store.memory import InMemoryStore
    return InMemoryStore()

# 下面给你一个“应用层 wrapper”，用于按 user_id 管理长期信息
class UserMemory:
    def __init__(self, store):
        self.store = store

    def _ns(self, user_id: str) -> Tuple[str, str]:
        # namespace 分层：("users", user_id)
        return ("users", user_id)

    def put_profile(self, user_id: str, key: str, value: Dict[str, Any]) -> None:
        self.store.put(self._ns(user_id), key, value)

    def get_profile(self, user_id: str, key: str) -> Optional[Dict[str, Any]]:
        item = self.store.get(self._ns(user_id), key)
        return None if item is None else item.value

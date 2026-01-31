from __future__ import annotations
from typing import Optional
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import InMemorySaver
def build_checkpointer_sqlite(db_path: str):
    """
    推荐安装：pip install langgraph-checkpoint-sqlite
    import 路径：from langgraph.checkpoint.sqlite import SqliteSaver
    见 PyPI 示例。:contentReference[oaicite:5]{index=5}
    """
    # try:
    #     return SqliteSaver.from_conn_string(db_path)
    # except Exception:
    #     # fallback：内存（开发用）
    #     return InMemorySaver()
    return InMemorySaver()

from __future__ import annotations
from dataclasses import dataclass

@dataclass
class AppConfig:
    # checkpoint
    checkpoint_db_path: str = "./agent/memory/checkpoints.sqlite"

    # 对话裁剪/摘要（你可以之后接入 LLM summary）
    max_messages_in_state: int = 10

    # 规划执行
    max_task_steps: int = 6

    # 可视化参数落地（这里先给本地内存/DB key，后续你可换 Redis）
    viz_namespace: str = "viz"

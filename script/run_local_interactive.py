from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, Optional, Tuple

from langgraph.types import Command

# 你的主入口
from agent.agents.controller.main_agent import MainAgent


# ====== 小工具：打印/序列化 ======
def _truncate(s: str, n: int = 800) -> str:
    if s is None:
        return ""
    s = str(s)
    return s if len(s) <= n else s[:n] + f"...(truncated,{len(s)} chars)"


def _safe_json(obj: Any) -> str:
    """尽量把复杂对象转成可读 JSON（失败就 str）。"""
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    except Exception:
        return str(obj)


def _extract_interrupt_payload(interrupt_list: Any) -> Any:
    """result['__interrupt__'] 通常是 Interrupt 对象列表，这里尽量抽取 .value。"""
    if interrupt_list is None:
        return None
    out = []
    try:
        for it in interrupt_list:
            # Interrupt(value=...) 或 dict 之类
            v = getattr(it, "value", it)
            out.append(v)
        return out
    except Exception:
        return interrupt_list


def _parse_stream_item(item: Any) -> Tuple[Optional[Any], str, Any]:
    """
    兼容不同 stream 返回签名：
    - 单 mode: chunk
    - 多 mode: (mode, chunk)
    - 有些版本/场景: (metadata, mode, chunk)  (尤其 async 示例里常见)
    返回: (metadata|None, mode, chunk)
    """
    if isinstance(item, tuple):
        if len(item) == 2:
            mode, chunk = item
            return None, str(mode), chunk
        if len(item) == 3:
            meta, mode, chunk = item
            return meta, str(mode), chunk
    # 单 mode
    return None, "updates", item


def _print_updates(chunk: Dict[str, Any], logf=None):
    # updates chunk 通常形如 {node_name: {delta...}}，也可能包含 "__interrupt__"
    if "__interrupt__" in chunk:
        payload = _extract_interrupt_payload(chunk.get("__interrupt__"))
        msg = f"\n[INTERRUPT DETECTED]\n{_safe_json(payload)}\n"
        print(msg)
        if logf:
            logf.write(msg + "\n")
        return

    # 普通 node 更新
    try:
        node = list(chunk.keys())[0]
        delta = chunk[node]
    except Exception:
        node = "?"
        delta = chunk

    msg = f"\n[UPDATES] node={node}\n{_safe_json(delta)}\n"
    print(msg)
    if logf:
        logf.write(msg + "\n")


def _print_messages(chunk: Any, logf=None):
    # messages chunk：官方说明是 (LLM token/message chunk, metadata) 二元组:contentReference[oaicite:5]{index=5}
    try:
        msg_obj, meta = chunk
        content = getattr(msg_obj, "content", None)
        # 有些 chunk 是 token 片段，有些是完整消息
        line = f"[MESSAGES] {_truncate(content) if content is not None else _truncate(msg_obj)}"
        print(line)
        if logf:
            logf.write(line + "\n")
    except Exception:
        line = f"[MESSAGES] { _truncate(chunk) }"
        print(line)
        if logf:
            logf.write(line + "\n")


def _print_custom(chunk: Any, logf=None):
    # custom：你在节点/工具里用 get_stream_writer() 发出来的任意对象:contentReference[oaicite:6]{index=6}
    line = f"[CUSTOM] { _safe_json(chunk) }"
    print(line)
    if logf:
        logf.write(line + "\n")


def _print_debug(chunk: Any, logf=None):
    # debug：尽可能多的信息（很吵）:contentReference[oaicite:7]{index=7}
    line = f"[DEBUG] { _truncate(_safe_json(chunk), 2000) }"
    print(line)
    if logf:
        logf.write(line + "\n")


# ====== 运行器：一次跑到中断 or 结束 ======
def run_until_pause_or_end(
    *,
    graph,
    input_obj: Any,
    config: Dict[str, Any],
    stream_modes,
    log_path: Optional[str] = None,
) -> Dict[str, Any]:
    logf = open(log_path, "a", encoding="utf-8") if log_path else None
    interrupt_payload = None

    try:
        # streaming 官方推荐：用 updates + messages，同时 subgraphs=True 便于检测 interrupt:contentReference[oaicite:8]{index=8}
        for item in graph.stream(
            input_obj,
            config=config,
            stream_mode=stream_modes,
            subgraphs=True,
        ):
            meta, mode, chunk = _parse_stream_item(item)

            if mode == "updates":
                # 检测 interrupt：chunk 里可能直接出现 "__interrupt__":contentReference[oaicite:9]{index=9}
                if isinstance(chunk, dict) and "__interrupt__" in chunk:
                    interrupt_payload = _extract_interrupt_payload(chunk.get("__interrupt__"))
                    _print_updates(chunk, logf=logf)
                    break
                _print_updates(chunk, logf=logf)

            elif mode == "messages":
                _print_messages(chunk, logf=logf)

            elif mode == "custom":
                _print_custom(chunk, logf=logf)

            elif mode == "debug":
                _print_debug(chunk, logf=logf)

            else:
                # 兜底：未知 mode
                line = f"[{mode}] {_truncate(_safe_json(chunk), 1200)}"
                print(line)
                if logf:
                    logf.write(line + "\n")

    finally:
        if logf:
            logf.flush()
            logf.close()

    # 取最新快照（无论中断还是结束，都能看到当前 values）
    snapshot = graph.get_state(config)
    values = getattr(snapshot, "values", None) or {}
    nxt = getattr(snapshot, "next", None)

    return {
        "interrupted": interrupt_payload is not None,
        "interrupt_payload": interrupt_payload,
        "next": nxt,
        "values": values,
    }


def print_state_brief(values: Dict[str, Any]):
    keys = [
        "query",
        "intent",
        "plan_version",
        "current_task_id",
        "done",
        "stop_reason",
        "viz_key",
        "answer",
    ]
    brief = {k: values.get(k) for k in keys if k in values}
    print("\n[STATE BRIEF]")
    print(_safe_json(brief))

    # 你最关心的中间变量（按需加）
    for k in ["tool_trace", "errors", "findings", "evidence_index"]:
        if k in values:
            print(f"\n[{k}]")
            print(_truncate(_safe_json(values.get(k)), 2000))


def prompt_resume_payload(interrupt_payload: Any) -> Any:
    """
    你可以直接输入 JSON，或用快捷命令：
    - approve
    - reject
    - feedback:...
    """
    print("\n=== 需要你决定下一步（resume payload）===")
    print("快捷输入：approve | reject | feedback:xxxx | 或直接输入 JSON")
    raw = input("resume> ").strip()

    if raw == "approve":
        return {"approved": True}
    if raw == "reject":
        return {"approved": False}
    if raw.startswith("feedback:"):
        return {"approved": False, "feedback": raw[len("feedback:") :].strip()}

    # 尝试按 JSON 解析
    try:
        return json.loads(raw)
    except Exception:
        # 兜底：当作字符串
        return raw


def main():
    print("=== LangGraph Ocean News Interactive Runner ===")

    user_id = os.environ.get("USER_ID") or input("user_id (default=u1)> ").strip() or "u1"
    thread_id = os.environ.get("THREAD_ID") or input("thread_id (default=u1:conv1)> ").strip() or f"{user_id}:conv1"

    # 你可以把日志写入文件，方便回放
    log_path = os.environ.get("RUN_LOG") or ""
    if log_path:
        print(f"[log] writing to {log_path}")

    # streaming modes 可调：
    # - updates：每步 state delta
    # - messages：LLM token/消息流（节点里真调用了 LLM 才会有）:contentReference[oaicite:10]{index=10}
    # - custom：节点/工具主动 writer(...) 输出:contentReference[oaicite:11]{index=11}
    # - debug：超级详细:contentReference[oaicite:12]{index=12}
    default_modes = ["updates", "messages"]
    if os.environ.get("STREAM_CUSTOM") == "1":
        default_modes.append("custom")
    if os.environ.get("STREAM_DEBUG") == "1":
        default_modes.append("debug")

    agent = MainAgent()
    graph = agent.graph

    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}

    print("\n命令：")
    print("  new    - 输入新问题并运行到中断/结束")
    print("  resume - 对上一次中断提供输入并继续运行")
    print("  state  - 打印当前 thread 的 state 快照（brief + 中间变量）")
    print("  quit   - 退出\n")

    last_interrupt = None

    while True:
        cmd = input("ocean> ").strip().lower()

        if cmd in {"quit", "exit", "q"}:
            print("bye.")
            return

        if cmd == "state":
            snap = graph.get_state(config)
            values = getattr(snap, "values", None) or {}
            print_state_brief(values)
            print(f"\n[next] {getattr(snap, 'next', None)}")
            continue

        if cmd == "new":
            query = input("query> ").strip()
            # 你也可以在这里让用户输入 constraints（先空着）
            init_state = {
                "user_id": user_id,
                "thread_id": thread_id,
                "query": query,
                "constraints": {},
                "done": False,
                "tool_trace": [],
                "errors": [],
                "viz_buffer": [],
            }

            result = run_until_pause_or_end(
                graph=graph,
                input_obj=init_state,
                config=config,
                stream_modes=default_modes,
                log_path=log_path or None,
            )
            print_state_brief(result["values"])
            last_interrupt = result["interrupt_payload"]

            if result["interrupted"]:
                print("\n[paused] graph waiting for your resume payload.")
            else:
                print("\n[done] graph reached END.")
            continue

        if cmd == "resume":
            if not last_interrupt:
                print("没有检测到上一次中断（last_interrupt 为空）。你可以先 new 跑一轮，或用 state 看 next。")
                continue

            payload = prompt_resume_payload(last_interrupt)
            input_obj = Command(resume=payload)

            result = run_until_pause_or_end(
                graph=graph,
                input_obj=input_obj,
                config=config,
                stream_modes=default_modes,
                log_path=log_path or None,
            )
            print_state_brief(result["values"])
            last_interrupt = result["interrupt_payload"]

            if result["interrupted"]:
                print("\n[paused] graph waiting for your resume payload.")
            else:
                print("\n[done] graph reached END.")
            continue

        print("未知命令。可用：new | resume | state | quit")


if __name__ == "__main__":
    main()

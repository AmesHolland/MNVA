from __future__ import annotations
from agent.services.runner import AgentRunner

if __name__ == "__main__":
    runner = AgentRunner()

    user_id = "u1"
    thread_id = "u1:conv1"

    r1 = runner.start_turn(user_id, thread_id, "请帮我分析最近南海相关海洋安全新闻趋势")
    print("R1:", r1["status"])
    if r1["status"] == "NEED_APPROVAL":
        # 模拟用户同意计划
        r2 = runner.resume_turn(user_id, thread_id, {"approved": True})
        print("R2:", r2["status"])
        if r2["status"] == "NEED_APPROVAL":
            # 模拟用户同意最终答案
            r3 = runner.resume_turn(user_id, thread_id, {"approved": True})
            print("R3:", r3["status"])
            print("answer:", r3["state"].get("answer"))
            print("viz_key:", r3["state"].get("viz_key"))

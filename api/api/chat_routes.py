# api/chat_routes.py
import json
from flask import Blueprint, request, Response, stream_with_context, jsonify
from langchain_core.messages import HumanMessage, AIMessage

from agent.agents.controller.graph import create_visual_analytics_assistant

chat_bp = Blueprint('chat', __name__)

compiled_graph = create_visual_analytics_assistant()


def generate_sse_stream(graph_app, inputs, config):
    """
    通用生成器，用于将 LangGraph 的运行过程转换为 SSE 数据流。
    每次产出 (yield) 的格式必须严格遵循 SSE 规范: "event: [事件名]\ndata: [JSON数据]\n\n"
    """
    try:
        # 【新增】：在漫长的 LLM 思考开始前，立刻向前端发送一个占位事件，稳固 HTTP 连接
        yield f"event: node_progress\ndata: {json.dumps({'node': 'System Initialing...'}, ensure_ascii=False)}\n\n"
        # ====== 【新增 1】：记录本次 stream 实际执行了哪些节点 ======
        executed_nodes = []

        # 接下来才是你原本的图流转逻辑
        for chunk in graph_app.stream(inputs, config=config, stream_mode="updates"):
            for node_name, state_update in chunk.items():
                executed_nodes.append(node_name)  # 记录节点名
                yield f"event: node_progress\ndata: {json.dumps({'node': node_name}, ensure_ascii=False)}\n\n"

        # 图运行暂停（或结束）后，检查当前状态
        state_snapshot = graph_app.get_state(config)
        next_node = state_snapshot.next

        if next_node and "check" in next_node:
            # 命中 interrupt_before，要求前端审批
            current_plan = state_snapshot.values.get("plan", {})
            payload = {
                "status": "waiting_for_approval",
                "plan": current_plan
            }
            yield f"event: interrupt\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
        else:
            # ====== 【修改部分开始】 ======
            # 图完全执行结束
            final_results = state_snapshot.values.get("analysis_results", {})
            final_report = state_snapshot.values.get("final_report", "")

            # 获取最后一条 AI 消息的内容（这包含了 simple_chat 的直接回答）
            messages_history = state_snapshot.values.get("messages", [])
            last_ai_message = messages_history[-1].content if (
                        messages_history and messages_history[-1].type == 'ai') else ""

            # ====== 【新增 2】：判断本轮是否真正执行了整合节点 ======
            # 如果本轮经过了 "integrate" 节点，说明是慢分支产生了新图表
            # 如果没经过（比如只走了 simple_chat），那就是 False
            is_new_visual = "integrate" in executed_nodes

            # 组装 Payload，增加 direct_answer 字段
            payload = {
                "status": "completed",
                "results": final_results,
                "direct_answer": final_report if isinstance(final_report, str) else last_ai_message,
                "is_new_visual_result": is_new_visual  # <-- 明确的 Flag 传给前端
            }
            yield f"event: completed\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

    except Exception as e:
        yield f"event: error\ndata: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"


@chat_bp.route('/chat', methods=['POST'])
def chat():
    """阶段 1：接收查询，启动分析流程并流式输出"""
    data = request.json

    # 满足要求1：多用户与多对话支持
    # 前端在发起请求时，需生成并传递这两项。组合成 thread_id 确保全局唯一
    user_id = data.get('user_id', 'default_user')
    session_id = data.get('session_id')  # 对应某个具体的聊天窗口
    if not session_id:
        return jsonify({"error": "Missing session_id"}), 400

    thread_id = f"{user_id}_{session_id}"
    topic = data.get('query')

    # 配置 LangGraph 记忆的上下文
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = initial_state = {
        "messages": [HumanMessage(content=f"请对以下主题进行深入研究：{topic}")],
        "research_topic": topic,
        "research_questions": [],
        "intent": [],
        "plan": {},  # 等待 planner 填充
        "user_feedback": "",  # 新增：用于接收 CLI 输入的反馈
        "findings": [],
        "task_results": {},
        "final_report": "",
        "draft_sections": {},
        "current_phase": "",
        "iteration_count": 0,
        "research_list": []
    }

    # 满足要求2：流式返回
    # 注意：mimetype 必须是 text/event-stream
    return Response(
        stream_with_context(generate_sse_stream(compiled_graph, initial_state, config)),
        mimetype='text/event-stream'
    )


@chat_bp.route('/feedback', methods=['POST'])
def feedback():
    """阶段 2：接收用户的审批意见，唤醒图并继续流式输出"""
    data = request.json

    user_id = data.get('user_id', 'default_user')
    session_id = data.get('session_id')
    user_feedback = data.get('feedback')  # "approve" 或 具体的修改意见

    if not session_id or not user_feedback:
        return jsonify({"error": "Missing session_id or feedback"}), 400

    thread_id = f"{user_id}_{session_id}"
    config = {"configurable": {"thread_id": thread_id}}

    # 1. 验证该图是否真的停在 check 节点（防止前端恶意重放请求）
    state_snapshot = compiled_graph.get_state(config)
    if not state_snapshot.next or "check" not in state_snapshot.next:
        return jsonify({"error": "Current thread is not waiting for approval."}), 400

    # 2. 将用户的反馈注入状态
    compiled_graph.update_state(config, {"user_feedback": user_feedback})

    # 3. 再次流式唤醒图 (传入 None 表示从中断处继续)
    return Response(
        stream_with_context(generate_sse_stream(compiled_graph, None, config)),
        mimetype='text/event-stream'
    )


# def generate_sse_stream(graph_app, inputs, config):
#     """
#     通用生成器：支持 Token 级逐字输出 + 节点进度推送
#     SSE 事件类型：
#     - node_progress: 节点执行进度（如 "planner" 节点开始/结束）
#     - token: LLM 逐字生成的 Token
#     - interrupt: 等待用户审批
#     - completed: 流程完成
#     - error: 异常信息
#     """
#     try:
#         # ❶ 第一步：先主动推送「启动执行」的节点进度（可选，提升体验）
#         yield f"event: node_progress\ndata: {json.dumps({'node': 'start', 'msg': '开始执行分析流程'}, ensure_ascii=False)}\n\n"
#
#         # ❷ 核心修改：stream_mode 改为 "messages"，并处理 Token 流
#         # stream_mode="messages" 会返回 LLM 生成的增量消息块（Token 级）
#         full_content = ""  # 缓存 AI 生成的完整内容
#         for chunk in graph_app.stream(inputs, config=config, stream_mode="messages"):
#             # 解析 messages 模式的 chunk（增量消息块）
#             if isinstance(chunk, AIMessage):
#                 # 提取增量 Token（chunk.content 是当前新增的内容）
#                 delta = chunk.content[len(full_content):]  # 只取新增的 Token
#                 if delta:  # 避免空内容推送
#                     full_content = chunk.content  # 更新完整内容
#                     # 推送 Token 事件（前端可逐字拼接）
#                     yield f"event: token\ndata: {json.dumps({'content': delta}, ensure_ascii=False)}\n\n"
#
#             # 兼容：如果你的图仍返回节点更新（部分版本 LangGraph 兼容），保留节点进度推送
#             elif isinstance(chunk, dict) and any(k in chunk for k in graph_app.nodes):
#                 for node_name, _ in chunk.items():
#                     yield f"event: node_progress\ndata: {json.dumps({'node': node_name}, ensure_ascii=False)}\n\n"
#
#         # ❸ 图暂停/结束后的状态处理（和原有逻辑一致）
#         state_snapshot = graph_app.get_state(config)
#         next_node = state_snapshot.next
#
#         if next_node and "check" in next_node:
#             # 命中审批节点，推送中断事件
#             current_plan = state_snapshot.values.get("current_plan", {})
#             payload = {
#                 "status": "waiting_for_approval",
#                 "plan": current_plan,
#                 "generated_content": full_content  # 附带已生成的内容（可选）
#             }
#             yield f"event: interrupt\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
#         else:
#             # 流程完成，推送最终结果
#             final_results = state_snapshot.values.get("final_report", {})
#             payload = {
#                 "status": "completed",
#                 "results": final_results,
#                 "full_content": full_content  # 附带完整生成内容
#             }
#             yield f"event: completed\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
#
#     except Exception as e:
#         yield f"event: error\ndata: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

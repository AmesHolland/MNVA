# api/chat_routes.py
import json
from flask import Blueprint, request, Response, stream_with_context, jsonify
from langchain_core.messages import HumanMessage, AIMessage

from agent.agents.controller.graph import create_visual_analytics_assistant
from agent.config.llm_config import llm_qw_quick # 引入轻量级模型

chat_bp = Blueprint('chat', __name__)

compiled_graph = create_visual_analytics_assistant()


def generate_sse_stream(graph_app, inputs, config):
    """
    通用生成器，用于将 LangGraph 的运行过程转换为 SSE 数据流。
    每次产出 (yield) 的格式必须严格遵循 SSE规范: "event: [事件名]\ndata: [JSON数据]\n\n"
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
                
                # 【核心修改】：如果 data_profiling 节点执行完毕，立即提取预览数据并推送
                if node_name == "data_profiling":
                    try:
                        # state_update 的结构是节点的返回值：{"analysis_results": {"data_profile": {...}}}
                        analysis_res = state_update.get("analysis_results", {})
                        data_profile = analysis_res.get("data_profile", {})
                        # preview_data = data_profile.get("preview_map_data", [])
                        
                        if data_profile:
                            yield f"event: data_profile\ndata: {json.dumps(data_profile, ensure_ascii=False)}\n\n"
                    except Exception as e:
                        print(f"Error extracting preview data: {e}")

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
            task_history = state_snapshot.values.get("task_history", [])

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
                "is_new_visual_result": is_new_visual,  # <-- 明确的 Flag 传给前端
                "task_history": task_history # 返回历史任务列表
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
    intent_override = None # data.get('intent_override') # 获取前端传来的意图覆盖

    dataset_id = data.get('dataset_id', None)
    # 配置 LangGraph 记忆的上下文
    config = {"configurable": {"thread_id": thread_id}}
    
    # 检查是否存在历史状态
    state_snapshot = compiled_graph.get_state(config)
    
    if state_snapshot.values:
        # 如果有历史状态，追加消息，而不是重置
        # 注意：我们需要更新 research_topic 为最新的 query，以便后续节点使用
        update_dict = {
            "messages": [HumanMessage(content=f"{topic}")],
            "research_topic": topic,
            "user_feedback": "", # 重置反馈
            "current_phase": "intent", # 重置阶段
        }
        if intent_override:
             # 如果有覆盖指令，我们预先设置 intent
             update_dict["intent"] = intent_override
        else:
             # 【重要】如果没有覆盖指令，必须清空 intent，防止 intent_node 误读上一轮的沙盒 intent
             update_dict["intent"] = {}

        inputs = update_dict
    else:
        # 初始状态
        initial_state = {
            "messages": [HumanMessage(content=f"请对以下主题进行深入研究：{topic}")],
            "research_topic": topic,
            "research_questions": [],
            "intent": intent_override if intent_override else {}, # 初始化 intent
            "plan": {},  # 等待 planner 填充
            "user_feedback": "",  # 新增：用于接收 CLI 输入的反馈
            "findings": [],
            "task_results": {},
            "final_report": "",
            "draft_sections": {},
            "current_phase": "",
            "iteration_count": 0,
            "research_list": [],
            "dataset_id": dataset_id,
            "output_language": "English", # 或 "English", "Japanese", 'Simplified Chinese'
            "task_history": [] # 初始化任务历史
        }
        inputs = initial_state

    # 满足要求2：流式返回
    # 注意：mimetype 必须是 text/event-stream
    return Response(
        stream_with_context(generate_sse_stream(compiled_graph, inputs, config)),
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

@chat_bp.route('/geo_resolve', methods=['POST'])
def geo_resolve():
    """
    接收一组经纬度坐标，利用 LLM 识别其对应的地理区域名称。
    Input: { "coordinates": [[118.2, 15.1], [119.5, 14.8], ...] }
    Output: { "regions": ["黄岩岛", "菲律宾西海岸"] }
    """
    data = request.json
    coords = data.get('coordinates', [])
    
    if not coords or len(coords) == 0:
        return jsonify({"regions": []})

    # 采样：如果点太多，只取前 10 个和中间几个，避免 Token 爆炸
    sampled_coords = coords[:5] + coords[-5:] if len(coords) > 10 else coords
    
    prompt = f"""
    你是一个地理信息专家。请根据以下经纬度坐标样本，判断它们大致位于哪个具体的海洋区域、岛屿或国家附近。
    
    坐标样本 (经度, 纬度): {sampled_coords}
    
    请直接输出最核心的 1-3 个地理名称（例如：'南海', '钓鱼岛', '关岛'），不要输出任何解释性文字。
    返回格式必须是 JSON 列表，例如: ["Region A", "Region B"]
    并且结果要以英文返回
    """
    
    try:
        response = llm_qw_quick.invoke([HumanMessage(content=prompt)])
        # 简单的字符串清洗，尝试提取 JSON
        content = response.content.strip()
        # 如果 LLM 返回了 ```json ... ```，去掉它
        if "```" in content:
            content = content.split("```")[1].replace("json", "").strip()
        
        regions = json.loads(content)
        if not isinstance(regions, list):
            regions = [str(regions)]
            
        return jsonify({"regions": regions})
        
    except Exception as e:
        print(f"Geo resolve error: {e}")
        return jsonify({"regions": ["Unknown Region"]})

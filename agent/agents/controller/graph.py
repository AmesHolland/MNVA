from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from agent.agents.controller.nodes import intent_node, planning_node, check_node, analyzing_node, integrating_node, \
    route_after_check, simple_chat_node, data_profiling_node, data_retrieval_node, route_after_intent, spatiotemporal_scoping_anchor_node
from agent.agents.schemas import ResearchState


def create_visual_analytics_assistant():
    """创建并编译解耦后的研究助手系统图"""
    graph = StateGraph(ResearchState)

    # 注册节点
    graph.add_node("intent", intent_node)
    graph.add_node("simple_chat", simple_chat_node)  # 新增
    graph.add_node("data_retrieval", data_retrieval_node)  # 新增
    graph.add_node("data_profiling", data_profiling_node)  # 新增
    graph.add_node("spatiotemporal_anchor", spatiotemporal_scoping_anchor_node)
    graph.add_node("planning", planning_node)
    graph.add_node("check", check_node)
    graph.add_node("analysis", analyzing_node)
    graph.add_node("integrate", integrating_node)

    # 定义边与路由
    graph.add_edge(START, "intent")

    # 关键：双轨制路由
    graph.add_conditional_edges(
        "intent",
        route_after_intent,
        {
            "simple_chat": "simple_chat",  # 快分支
            "data_retrieval": "data_retrieval"  # 慢分支起点
        }
    )

    # 快分支直接结束
    graph.add_edge("simple_chat", END)

    # 慢分支执行链 (解决了盲目规划)
    graph.add_edge("data_retrieval", "data_profiling")
    graph.add_edge("data_profiling", "spatiotemporal_anchor")
    graph.add_edge("spatiotemporal_anchor", "planning")

    graph.add_edge("planning", "check")
    graph.add_conditional_edges("check", route_after_check)
    graph.add_edge("analysis", "integrate")
    graph.add_edge("integrate", END)

    # 编译并设置中断点
    memory = MemorySaver()
    compiled_graph = graph.compile(checkpointer=memory, interrupt_before=["check"])

    return compiled_graph
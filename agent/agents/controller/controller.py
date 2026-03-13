import json
import os
from datetime import datetime
from typing import List, Optional
from typing import TypedDict, Annotated, Dict, Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph import add_messages
from pydantic import BaseModel
from pydantic import Field

from agent.agents.controller.graph import create_visual_analytics_assistant
from agent.agents.sub_agent.deep_dive_agent import deep_dive_agent
from agent.agents.sub_agent.global_monitor_agent import global_monitor_agent
from agent.agents.sub_agent.relation_miner_agent import relation_miner_agent
from agent.config.llm_config import llm_qw_quick
from agent.tools.news_manager import get_news_by_id

from agent.tools.base import safe_parse_json

os.getenv("LANGCHAIN_API_KEY")
os.getenv("LANGSMITH_ENDPOINT")
os.getenv("LANGSMITH_TRACING")

# ==================== 对话模型定义 ====================
model = llm_qw_quick
# ==================== 数据模型定义 ====================

class SearchResult(BaseModel):
    """搜索结果"""
    title: str
    source: str
    url: str
    snippet: str | None
    content: str
    # relevance_score: float = Field(ge=0.0, le=1.0) | None
    publish_date: str


class ResearchFinding(BaseModel):
    """研究发现"""
    topic: str
    key_points: list[str]
    evidence: list[str]
    # confidence: float = Field(ge=0.0, le=1.0)
    sources: list[str]


class ResearchOutline(BaseModel):
    """研究大纲"""
    title: str
    abstract: str
    sections: list[str]
    key_questions: list[str]
    methodology: str


class Citation(BaseModel):
    """引用"""
    id: str
    authors: list[str]
    title: str
    source: str
    date: str
    url: Optional[str] = None


class ResearchReport(BaseModel):
    """研究报告"""
    title: str
    executive_summary: str
    introduction: str
    methodology: str
    findings: list[str]
    analysis: str
    conclusions: list[str]
    recommendations: list[str]
    citations: list[Citation]
    generated_at: str


# ==================== 状态定义 ====================

class ResearchState(TypedDict):
    """研究助手状态"""
    # 消息历史
    messages: Annotated[list, add_messages]

    # 研究主题
    research_topic: str
    research_questions: list[str]

    # 研究进展
    intent: dict
    plan: dict
    task_results: dict
    news_list: list[dict]

    # 报告
    draft_sections: dict
    final_report: str
    citations: list[dict]

    # 状态追踪
    """
    current_phase 共分为5个阶段
    ready planning confirming analyzing integrating 
    ready 状态0 开启的新对话 / 先前的任务完成
    planning 状态1 用户input之后 进行意图识别、路径规划 对input之前的状态进行识别 0 就正常新规划 2就读取当前计划并进行重新设计
    confirming 状态2 等待用户确认计划是否可行 是 -> 状态3 ；否 -> 状态1
    analyzing 状态3 基于plan 进行多个子问题的分析 然后将结果暂存
    integrating 状态4 将之前分的多个子任务的结果进行整合 形成一个规范化的报告
    """
    current_phase: str
    iteration_count: int

    query: str
    # 数据缓存
    research_list: list[ResearchReport]

    # 存储 Planner 生成的具体计划（例如需要调用哪些子Agent及参数）
    current_plan: Dict[str, Any]
    # 存储用户的审批意见：可以是 "approve"（同意），或者是一段修改意见的字符串
    user_feedback: str
    # 其他你原有的状态，如搜集到的数据、分析结果等
    analysis_results: Dict[str, Any]

def run_research_hitl(topic: str):
    """运行带有 Human-in-the-Loop (HITL) 审批机制的研究任务"""
    print("\n" + "=" * 60)
    print("🔬 启动多Agent可视分析系统 (CLI 测试版)")
    print("=" * 60)
    print(f"研究主题: {topic}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. 创建研究助手 (注意：内部必须 compile(checkpointer=memory, interrupt_before=["check"]))
    assistant = create_visual_analytics_assistant()

    # 2. 初始状态
    initial_state = {
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

    # 3. 配置 thread_id，这对记忆保存和恢复至关重要
    config = {"configurable": {"thread_id": f"research_{datetime.now().strftime('%Y%m%d%H%M%S')}"}}

    print("\n🚀 [系统] 正在进行意图识别与任务编排 (Planning)...")

    # 4. 首次启动图：它会运行到 interrupt_before 指定的节点（如 "check"）然后暂停
    assistant.invoke(initial_state, config)

    # 5. 进入交互循环（模拟前后端多次握手）
    while True:
        # 获取当前图的状态快照
        state_snapshot = assistant.get_state(config)

        # 检查图是否已经执行完毕 (没有 next node 说明跑到了 END)
        if not state_snapshot.next:
            print("\n✅ [系统] 分析与整合流程已全部完成！")
            break

        # 如果图暂停在了 check 节点前
        if "check" in state_snapshot.next:
            # 读取 Planner 刚刚生成的计划
            current_plan = state_snapshot.values.get("plan", "暂无计划内容")

            print("\n" + "=" * 60)
            print("⏸️ [审批节点] 系统需要您的确认才能继续执行")
            print("=" * 60)
            print(f"当前生成的调用计划 (Plan): \n{current_plan}")
            print("-" * 60)

            # 使用 input 模拟前端 Vue 的对话框输入
            user_input = input("💡 请审核计划 (输入 'y' 确认执行，或输入修改意见让AI重做): ").strip()
            current_phase = "analyzing"
            # 逻辑判断：确认还是打回
            if user_input.lower() in ['y', 'yes', 'ok', '同意', '确认']:
                feedback = "approve"
                print("\n▶️ [系统] 您已批准计划。正在调用各个 Agent 进行数据检索与可视化生成...")
            else:
                feedback = user_input
                current_phase = "planning"
                print(f"\n🔄 [系统] 收到您的修改意见：'{feedback}'。正在让 Planner 重新规划...")

            # 6. 核心步骤：将用户的意见注入到图的状态中
            # 注意：如果你的 state 里定义的键名不是 user_feedback，请与你的 StateDict 保持一致
            assistant.update_state(config, {"user_feedback": feedback , "current_phase": current_phase})

            # 7. 唤醒图继续执行。传入 None 表示使用已更新的 state 继续向下走
            assistant.invoke(None, config)

        else:
            # 容错：如果停在了预料之外的节点，直接尝试继续往下跑
            print(f"\n⚠️ [系统] 图暂停在了非审批节点: {state_snapshot.next}，尝试自动继续...")
            assistant.invoke(None, config)

    # 8. 循环结束，输出最终结果
    final_state = assistant.get_state(config).values
    print("\n" + "=" * 60)
    print("📄 最终可视化方案与分析报告")
    print("=" * 60)
    print(final_state.get("final_report", "报告生成失败"))

    return final_state


if __name__ == '__main__':
    topic = "对2025年第四季度美国在深海采矿方面采取的一系列行动"
    run_research_hitl(topic)
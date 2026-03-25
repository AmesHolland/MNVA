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
    
    # 新增字段
    dataset_id: str 
    spatiotemporal_blueprint: dict
    output_language: str
    research_trajectory: list
    report_count: int
    

from langchain_core.messages import HumanMessage
from agent.tools.data_manager import process_uploaded_file, load_registry, load_dataset


import os
import json
from datetime import datetime
from langchain_core.messages import HumanMessage

# 确保从你的 data_manager 模块导入核心函数
from agent.tools.data_manager import process_uploaded_file, load_registry, get_dataset_entities

def run_research_hitl(topic: str, file_path: str = None, dataset_id: str = None):
    """
    运行带有数据集接入、受控词表注入与审批机制的研究任务 (完整增强版)
    :param topic: 研究主题
    :param file_path: 可选，原始数据文件路径 (CSV/Excel/JSON)
    :param dataset_id: 可选，直接指定已有的数据集 ID
    """
    print("\n" + "=" * 60)
    print("🔬 启动多Agent可视分析系统 (Controlled Vocabulary Version)")
    print("=" * 60)
    print(f"研究主题: {topic}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # --- 1. 动态数据集路由 (Dynamic Dataset Routing) ---
    target_ds = "default_pool"
    
    # 模式 A: 重新加工新文件
    if file_path and os.path.exists(file_path):
        print(f"\n🧬 [数据入库] 检测到新源文件: {os.path.basename(file_path)}")
        print("🧠 正在解析时空蓝图，执行实体归一化重构...")
        res = process_uploaded_file(file_path, os.path.basename(file_path))
        if res and "success" in res:
            target_ds = res["dataset"]["dataset_id"]
            print(f"✅ 归一化加工完成！分配 ID: {target_ds} ({res['dataset']['row_count']} 条记录)")
        else:
            print(f"❌ 加工失败: {res.get('error') if res else '未知错误'}，降级使用默认池。")
            
    # 模式 B: 指定已有数据集
    elif dataset_id:
        print(f"\n📦 [指定挂载] 正在链接数据集仓库: {dataset_id}")
        target_ds = dataset_id
        
    # 模式 C: 自动兼容模式 (寻找最近一次成功加工的数据)
    else:
        registry = load_registry()
        if registry:
            target_ds = registry[-1]["dataset_id"]
            print(f"\n💡 [自动兼容] 未指定文件，已自动挂载最新证据库: {target_ds}")
        else:
            print("\n⚠️ [兜底模式] 未找到历史数据集，将搜索默认数据池。")

    # --- 2. 核心：提取受控词表 (Controlled Vocabulary) ---
    # 从加工好的数据集中提取“标准实体名录”，用于强力约束 Planner
    print(f"📒 [受控词表] 正在从数据集 {target_ds} 提取标准实体名录...")
    try:
        canonical_entities = get_dataset_entities(target_ds)
        print(f"✅ 已加载 {len(canonical_entities)} 个标准实体，准备注入规划指令。")
    except Exception as e:
        print(f"⚠️ 实体名录提取失败: {e}，将使用自由检索模式。")
        canonical_entities = []

    # --- 3. 初始化助手与状态 ---
    # 这里的 assistant 对应你构建好的 LangGraph 图对象
    from agent.agents.controller.controller import create_visual_analytics_assistant
    assistant = create_visual_analytics_assistant()

    # 构造初始指令：重点在于注入“受控词表”约束，防止 Planner 生成不存在的实体
    entity_constraint = ""
    if canonical_entities:
        # 限制列表长度，防止 Prompt 过长，如有需要可全量注入
        display_entities = canonical_entities[:200] 
        entity_constraint = f"\n\n【重要检索约束：受控词表】\n当前证据库中仅存在以下标准实体。在调用 retrieve_news 填写 entities 参数时，必须且只能使用此列表中的词汇。如果想搜索列表之外的内容，请将其放入 keywords 字段：\n{json.dumps(display_entities, ensure_ascii=False)}"

    initial_state = {
        "messages": [HumanMessage(content=f"请对以下主题进行深入研究：{topic}{entity_constraint}")],
        "research_topic": topic,
        "query": topic,
        "output_language": "zh",
        "dataset_id": target_ds,  # 确保所有 Agent 共享同一个数据集上下文
        "canonical_entities": canonical_entities,
        
        # 初始化其他状态字段
        "research_questions": [],
        "intent": {}, 
        "plan": {}, 
        "current_plan": {},
        "user_feedback": "", 
        "current_phase": "planning",
        "iteration_count": 0,
        "news_list": [],
        "findings": [],
        "task_results": {},
        "analysis_results": {},
        "final_report": "",
        "draft_sections": {},
        "research_list": [],
        "citations": [],
        "research_trajectory": [],
        "report_count": 0,
        "spatiotemporal_blueprint": {}
    }

    # 4. 配置 thread_id 用于持久化记忆
    config = {"configurable": {"thread_id": f"research_{datetime.now().strftime('%Y%m%d%H%M%S')}"}}

    print("\n🚀 [Planner] 正在结合受控词表进行任务规划...")

    # 5. 首次执行：进入规划阶段
    assistant.invoke(initial_state, config)

    # 6. 交互循环 (Human-In-The-Loop)
    while True:
        state_snapshot = assistant.get_state(config)

        # 如果没有下一个节点，说明流程结束
        if not state_snapshot.next:
            print("\n✅ [系统] 分析与整合流程已全部完成！")
            break

        # 检查是否处于审批环节
        # 注意：这里的条件根据你的图节点名称（如 'planner_check'）进行微调
        if any("check" in n for n in state_snapshot.next):
            plan_data = state_snapshot.values.get("plan", {})
            
            print("\n" + "=" * 60)
            print("⏸️ [审批节点] 请审核动态生成的分析计划 (受控词表模式)")
            print("=" * 60)
            
            if isinstance(plan_data, dict) and "total_plan_logic" in plan_data:
                print(f"📡 规划逻辑: {plan_data['total_plan_logic']}")
                print(f"🤖 拟调用 Agent 数量: {len(plan_data.get('tasks', []))}")
                # 展示拟检索的实体，供用户核对是否在清单内
                for i, task in enumerate(plan_data.get('tasks', [])):
                    args = task.get('args', {})
                    if 'entities' in args:
                        print(f"   └─ 任务 {i+1} 拟检索实体: {args['entities']}")
            else:
                print(f"📋 计划详情: \n{plan_data}")
            
            print("-" * 60)
            user_input = input("💡 输入 'y' 确认执行，或输入修改建议: ").strip()
            
            if user_input.lower() in ['y', 'yes', 'ok', '同意']:
                feedback = "approve"
                curr_phase = "analyzing"
                print("\n▶️ [系统] 计划获批。正在并发调度数据路由...")
            else:
                feedback = user_input
                curr_phase = "planning"
                print(f"\n🔄 [系统] 收到调整建议。正在重新对齐蓝图...")

            # 更新状态并继续执行
            assistant.update_state(config, {"user_feedback": feedback, "current_phase": curr_phase})
            assistant.invoke(None, config)
        else:
            # 自动执行非交互节点
            assistant.invoke(None, config)

    # 7. 结果展示
    final_values = assistant.get_state(config).values
    print("\n" + "=" * 60)
    print("📄 最终分析报告 (基于全量归一化证据链)")
    print("=" * 60)
    print(final_values.get("final_report", "未生成内容。"))
    
    return final_values


if __name__ == '__main__':
    topic = "对2025年第四季度美国在深海采矿方面采取的一系列行动"
    data_path = r"C:\\Users\\win11\\Desktop\\MNVA\\agent\\tools\\deep_sea_mining_news.json"
    run_research_hitl(topic, file_path=data_path)
    
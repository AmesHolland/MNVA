from typing import List, Dict, Any, Literal
from typing import TypedDict, Annotated

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class ResearchState(TypedDict):
    """研究助手状态"""
    messages: Annotated[list, add_messages]
    research_topic: str
    research_questions: list[str]
    intent: dict
    plan: dict
    task_results: dict
    news_list: list[dict]
    draft_sections: dict
    final_report: str
    citations: list[dict]
    current_phase: str
    iteration_count: int
    query: str
    research_list: list[Any] # 此处依据你原有的 ResearchReport 类型
    current_plan: Dict[str, Any]
    user_feedback: str
    analysis_results: Dict[str, Any]
    spatiotemporal_blueprint : dict
    # 【新增】任务历史，用于存储多次分析的结果
    task_history: List[Dict[str, Any]]
    # 🌟 新增：存放当前工作台选中的数据集 ID
    dataset_id: str
    output_language : str
    # 🌟 新增：系统记忆相关字段
    report_count: int
    # 如果你用的是 LangGraph，建议用 operator.add 这样每次 yield/return 会自动 append
    # 如果不是，直接用普通的 list 也可以
    research_trajectory: list

# === 整合节点的溯源输出模型 ===
class Claim(BaseModel):
    statement: str = Field(description="总结的论点、事实判断或事件描述。")
    is_direct_quote: bool = Field(description="如果是直接截取新闻原话为 True，自行归纳总结为 False。")
    source_ids: List[str] = Field(description="支撑该句话的具体新闻 DOC_ID 列表。必须从输入的上下文中提取，严禁捏造。")

from pydantic import BaseModel, Field
from typing import List

# 🌟 终极溯源声明块
class ProvenanceClaim(BaseModel):
    content: str = Field(description="The analytical statement, fact, or strategic insight.")
    source_subtask: str = Field(description="The exact task_id (e.g., 'task_1') that provided this information.")
    phase_name: str = Field(description="The name of the spatiotemporal phase this content belongs to (e.g., 'Phase 1: Incubation'). Use 'Global' if it applies to the whole timeline.")
    is_subjective_insight: bool = Field(description="True if this is an AI strategic insight/prediction. False if it is an objective fact.")
    source_ids: List[str] = Field(description="List of exact DOC_IDs supporting this claim. Empty if is_subjective_insight is True.")

# 报告章节
class ReportSection(BaseModel):
    section_title: str = Field(description="Title of the section (e.g., 'Phase 2: Escalation in the South China Sea').")
    claims: List[ProvenanceClaim] = Field(description="A coherent sequence of claims that form the narrative of this section.")

# === 2. 时序版：阶段概述骨架 (Route B 的精髓) ===
class PhaseSummary(BaseModel):
    phase_index: int = Field(description="Chronological order (1, 2, 3...).")
    phase_name: str = Field(description="Name of the phase strictly based on the blueprint.")
    phase_time_range: str = Field(description="Time range of this phase (e.g., '2025-01 to 2025-04').")
    phase_summary: str = Field(description="Several sentences summarizing the overall situation in this phase. Must not be empty.")
    related_subtasks: List[str] = Field(description="List of task_ids (e.g., ['task_2']) that are relevant to this phase.")
    source_ids: List[str] = Field(description="DOC_IDs backing up this phase summary.")

# 完整报告
class FinalReport(BaseModel):
    report_title: str = Field(description="Overall title of the strategic report.")
    executive_summary: str = Field(description="High-level overview (no strict provenance tracking needed here).")
    # 🌟 时序线：保证每个阶段都有态势兜底
    phase_summaries: List[PhaseSummary] = Field(
        description="Chronological phase-by-phase situational overviews strictly following the Blueprint.")
    sections: List[ReportSection] = Field(description="The main body of the report, strictly organized by evolutionary phases or core topics.")
    conclusion: str = Field(description="Forward-looking predictive conclusion.")

class EvolutionPhase(BaseModel):
    phase_id: int = Field(description="阶段序号，如 1, 2, 3")
    phase_name: str = Field(description="高度概括该阶段特征的名称，如 '单边勘探准备期' 或 '国际规则博弈期'")
    time_range: str = Field(description="该阶段的时间起止，如 '2025-10-01 to 2025-10-25'")
    start_date: str = Field(description="该阶段的开始日期，如 '2025-10-01")
    end_date: str = Field(description="该阶段的结束日期，如 '2025-10-01 to 2025-10-25'")
    spatial_focus: str = Field(description="该阶段的核心地理焦点，如 '华盛顿'、'CCZ海域' 或 '联合国纽约总部'")
    spatial_scale: Literal["Micro", "Regional", "Macro"] = Field(
        description="空间尺度。Micro: 特定机构/微观坐标; Regional: 特定海域/专属经济区; Macro: 跨国/全球体系"
    )
    key_entities: List[str] = Field(description="该阶段最活跃的核心实体列表")
    description: str = Field(description="一句话简述该阶段的核心事件与演化逻辑")

class SpatiotemporalBlueprint(BaseModel):
    overall_narrative: str = Field(description="对整个事件时空演变轨迹的宏观定性总结（50字以内）")
    phases: List[EvolutionPhase] = Field(description="按时间顺序排列的演化阶段列表，通常为 2 到 4 个阶段")


class TaskNode(BaseModel):
    task_id: str = Field(description="Unique task id, e.g. 'global_monitor_phase_1'")
    agent: Literal[
        "Global_Monitor_Agent",
        "Deep_Dive_Agent",
        "Relation_Miner_Agent"
    ] = Field(description="Name of the assigned sub-agent")

    action: str = Field(description="Concrete execution goal of the task")

    target_phase_ids: List[int] = Field(
        default_factory=list,
        description="Phase IDs selected from the spatiotemporal blueprint"
    )

    args: Dict[str, Any] = Field(
        default_factory=dict,
        description="Agent-specific arguments only; do not put dependency or phase routing here"
    )

    dependency: List[str] = Field(
        default_factory=list,
        description="IDs of prerequisite tasks whose structured outputs are required"
    )


class ExecutionPlan(BaseModel):
    total_plan_logic: str = Field(
        description="Brief explanation of the overall routing logic, especially how tasks are assigned to phase slices and dependencies"
    )
    tasks: List[TaskNode] = Field(
        description="A list of phase-aware tasks that can be executed as a DAG"
    )

# === 新增：可溯源的洞察基类 ===
class TraceableInsight(BaseModel):
    statement: str = Field(description="The analytical insight, prediction, or deduction statement.")
    source_ids: List[str] = Field(description="List of exact DOC_IDs that inspired or logically support this subjective insight. DO NOT fabricate IDs.")

# === 升级：释放大模型思考能力的洞察模型 ===
class StrategicInsights(BaseModel):
    core_conflict: TraceableInsight = Field(description="Highly analytical summary of the core geopolitical contradictions or friction points.")
    hidden_intentions: TraceableInsight = Field(description="Deep analysis of the underlying strategic motives of the key actors involved.")
    trend_prediction: TraceableInsight = Field(description="A forward-looking forecast of how this situation is likely to evolve.")

from typing import List, Literal, Union, Optional
from typing import TypedDict, Annotated, Dict, Any

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

# === 整合节点的溯源输出模型 ===
class Claim(BaseModel):
    statement: str = Field(description="总结的论点、事实判断或事件描述。")
    is_direct_quote: bool = Field(description="如果是直接截取新闻原话为 True，自行归纳总结为 False。")
    source_ids: List[str] = Field(description="支撑该句话的具体新闻 DOC_ID 列表。必须从输入的上下文中提取，严禁捏造。")

class ReportSection(BaseModel):
    subtitle: str = Field(description="该章节的小标题")
    content_claims: List[Claim] = Field(description="该章节的正文，必须拆解为多个逻辑连贯的论点/句子，每个句子必须附带 source_ids。")
    ref_task_ids: List[str] = Field(description="该章节分析所依赖的子任务 ID 列表 (例如 ['1', '2'])，用于前端关联图表。")

class FinalReport(BaseModel):
    report_title: str = Field(description="海洋态势深度分析报告标题")
    executive_summary: str = Field(description="高度概括的执行摘要")
    executive_source_ids: List[str] = Field(description="支撑执行摘要的核心 DOC_ID 列表")
    sections: List[ReportSection] = Field(description="报告的主体章节")
    conclusion: str = Field(description="对未来趋势的最终战略研判与结语")

class EvolutionPhase(BaseModel):
    phase_id: int = Field(description="阶段序号，如 1, 2, 3")
    phase_name: str = Field(description="高度概括该阶段特征的名称，如 '单边勘探准备期' 或 '国际规则博弈期'")
    time_range: str = Field(description="该阶段的时间起止，如 '2025-10-01 to 2025-10-25'")
    spatial_focus: str = Field(description="该阶段的核心地理焦点，如 '华盛顿'、'CCZ海域' 或 '联合国纽约总部'")
    spatial_scale: Literal["Micro", "Regional", "Macro"] = Field(
        description="空间尺度。Micro: 特定机构/微观坐标; Regional: 特定海域/专属经济区; Macro: 跨国/全球体系"
    )
    key_entities: List[str] = Field(description="该阶段最活跃的核心实体列表")
    description: str = Field(description="一句话简述该阶段的核心事件与演化逻辑")

class SpatiotemporalBlueprint(BaseModel):
    overall_narrative: str = Field(description="对整个事件时空演变轨迹的宏观定性总结（50字以内）")
    phases: List[EvolutionPhase] = Field(description="按时间顺序排列的演化阶段列表，通常为 2 到 4 个阶段")

# ==========================================
# 1. 定义规划蓝图的 Pydantic 模型 (保持不变)
# ==========================================
class TaskNode(BaseModel):
    task_id: int = Field(description="任务的唯一执行序号")
    agent: str = Field(description="需要调用的探员名称，例如 'Global_Monitor_Agent'")
    action: str = Field(description="该任务的具体执行目标和指令")
    args: Dict[str, Any] = Field(description="传递给探员的具体参数，必须包含 target_phase_ids")
    dependency: Optional[Union[int, List[int]]] = Field(description="该任务依赖的前置 task_id，如果没有则为 null")

class ExecutionPlan(BaseModel):
    total_plan_logic: str = Field(description="简述整体调度逻辑，尤其是如何根据时空蓝图进行动态路由的（1-2句话）")
    tasks: List[TaskNode] = Field(description="按执行顺序排列的任务列表")
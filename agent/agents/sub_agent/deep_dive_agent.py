from typing import List, Optional

import pandas as pd
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from agent.agents.schemas import Claim
from agent.config.llm_config import llm_qw_quick
from agent.config.prompt_template import DEEP_DIVE_PROMPT
from agent.tools.news_manager import get_news_by_id

from pydantic import BaseModel, Field
from typing import List, Optional


# 1. 行为烈度评分
class IntensityScore(BaseModel):
    military: int = Field(description="0-5 score: Military aggression or hardware involvement.")
    diplomatic: int = Field(description="0-5 score: Diplomatic pressure or formal statements.")
    media: int = Field(description="0-5 score: Public attention or propaganda intensity.")
    Technology: int = Field(description="0-5 score: Technology Development.")
    Scientific: int = Field(description="0-5 score: Scientific Expedition Activity.")


# 2. 单个事件节点 (已增加溯源字段)
class EventNode(BaseModel):
    start_date: str = Field(description="Event start date in YYYY-MM-DD format.")
    end_date: str = Field(description="Event end date in YYYY-MM-DD format. If it's a single-day event, this should be the same as start_date.")
    location_name: str = Field(description="Specific location name or region name.")
    geo_lat: Optional[float] = Field(description="Latitude of the location, if inferable. Otherwise null.")
    geo_lon: Optional[float] = Field(description="Longitude of the location, if inferable. Otherwise null.")
    # 建议将字段名从 action_type 改为 domain 或 category
    domain: str = Field(description="The core thematic domain of this event. Must be exactly one of: 'Military', 'Technology', 'Politics', 'Environment', 'Governance', 'Resources'.")
    summary: str = Field(description="Concise summary of the action (max 10 words).")
    scores: IntensityScore = Field(description="Intensity assessment of this specific event.")
    source_ids: List[str] = Field(description="List of exact DOC_IDs from the input text that describe this event. Like ['001']")

# 3. 时空演变阶段 (用于前端渲染垂直时间轴组件)
class EvolutionPhase(BaseModel):
    phase_name: str = Field(description="Name of the evolutionary phase (e.g., 'Incubation', 'Escalation', 'De-escalation').")
    start_date: str = Field(description="Start date of this phase in YYYY-MM-DD format.")
    end_date: str = Field(description="End date of this phase in YYYY-MM-DD format.")
    phase_summary: str = Field(description="Narrative summary of the entity's strategy or situation during this phase.")
    source_ids: List[str] = Field(description="List of DOC_IDs supporting this phase. Like ['001']")

# 4. 完整输出结构 (加入 evolution_phases)
class DeepDiveOutput(BaseModel):
    entity_name: str = Field(description="The normalized name of the target entity.")
    entity_type: str = Field(description="Type: 'Vessel', 'Country', 'Organization', or 'Event'.")
    overview_claims: List[Claim] = Field(description="Strategic summary of the entity's behavior broken down into traceable claims.")
    evolution_phases: List[EvolutionPhase] = Field(description="Chronological phases telling the story of the entity's spatiotemporal evolution.")
    events: List[EventNode] = Field(description="Chronological list of discrete events associated with the entity.")

# --- 初始化组件 (通常在 graph 构建前完成) ---
llm = llm_qw_quick
parser = JsonOutputParser(pydantic_object=DeepDiveOutput)

prompt = PromptTemplate(
    template=DEEP_DIVE_PROMPT,
    # 【新增】：在 input_variables 中加入 blueprint_context 和 query
    input_variables=["target_entity", "news_context", "query", "blueprint_context", "output_language"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

deep_dive_chain = prompt | llm | parser


# --- 核心节点函数 ---
def deep_dive_agent(entity, query, docs, blueprint_context="", output_language="English"):
    """
    LangGraph Node: Performs deep dive analysis on a specific entity to extract its spatiotemporal storyline.
    """
    print(f"\n--- 🔎 DEEP DIVE AGENT: Analyzing Spatiotemporal Evolution for {entity} ---")

    # 1. 严格格式化上下文，确保 DOC_ID 可见
    news_context = "\n".join(
        [f"[DOC_ID: {d.get('DOC_ID')}] - [{d.get('publish_date', 'N/A')}] {d.get('content', '')}" for d in docs])

    # 2. 执行 LLM Chain
    try:
        structured_result = deep_dive_chain.invoke({
            "target_entity": entity,
            "news_context": news_context,
            "query": query,
            "blueprint_context": blueprint_context,  # 【新增】：注入蓝图
            "output_language": output_language
        })
    except Exception as e:
        print(f"❌ Deep Dive 运行出错: {e}")
        return {"final_answer": f"Analysis failed: {str(e)}"}

    events = structured_result.get("events", [])

    # --- A. 计算雷达图数据 ---
    radar_data = {"military": 0, "diplomatic": 0, "media": 0}
    if events:
        df = pd.DataFrame([e['scores'] for e in events])
        radar_data = df.mean().round(1).to_dict()

    # --- B. 组装 Vega-Lite / ECharts 友好的数据 ---
    # 【修改】：注入 intensity (用于地图散点大小) 和 source_ids (用于点击溯源)
    map_data = [
        {
            "date": e['start_date'],
            "lat": e['geo_lat'],
            "lon": e['geo_lon'],
            "name": e['location_name'],
            "type": e['domain'],
            "summary": e['summary'],
            "intensity": e['scores']['military'],  # 默认用军事烈度决定点的大小
            "source_ids": e['source_ids']
        }
        for e in events if e.get('geo_lat') and e.get('geo_lon')
    ]

    # 【修改】：注入 source_ids
    gantt_data = [
        {
            "start": e['start_date'],  # 开始时间
            "end": e['end_date'],  # 结束时间
            "category": e['domain'],  # Y轴分类
            "intensity": e['scores']['military'],  # 烈度（用于颜色深浅）
            "summary": e['summary'],
            "source_ids": e['source_ids']
        }
        for e in events
    ]

    # 3. 更新 State
    return {
        # 文本回复 (前端可将 overview_claims 渲染为气泡，将 evolution_phases 渲染为垂直时间轴)
        "final_answer": structured_result['overview_claims'],
        # 结构化可视化数据
        "visualization_data": {
            "type": "deep_dive_dashboard",
            "evolution_timeline": structured_result['evolution_phases'],  # 【新增】：叙事线数据流向前端
            "entity_info": {
                "name": structured_result['entity_name'],
                "type": structured_result['entity_type']
            },
            "radar_chart": radar_data,
            "map_chart": map_data,  # 前端 ECharts 可以复用 global_map 的 timeline 逻辑！
            "gantt_chart": gantt_data,
            "raw_events": events
        },
        "structured_insight": structured_result
    }

if __name__ == '__main__':
    query = "对2025年第四季度美国在深海采矿方面采取的一系列行动 time-range:2025-10-08 to 2025-12-31"
    news_list = get_news_by_id([])
    result = deep_dive_agent("美国", query, news_list, "美国从10月激进部署与规则博弈，转向12月因挪威暂停及国际压力而陷入治理僵局。")
    print(result)
from typing import List, Optional

import pandas as pd
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from agent.agents.schemas import Claim
from agent.config.llm_config import llm_qw_quick, llm_qw_thinking
from agent.config.prompt_template import DEEP_DIVE_PROMPT
from agent.tools.news_manager import get_news_by_id

from pydantic import BaseModel, Field
from typing import List, Optional

from pydantic import BaseModel, Field
from typing import List, Optional
from agent.agents.schemas import Claim


# 1. 行为烈度评分 (统一小写命名，方便后续 pandas 处理)
class IntensityScore(BaseModel):
    military: int = Field(description="0-5 score: Military aggression or hardware involvement.")
    diplomatic: int = Field(description="0-5 score: Diplomatic pressure or formal statements.")
    media: int = Field(description="0-5 score: Public attention or propaganda intensity.")
    technology: int = Field(description="0-5 score: Technology Development.")
    scientific: int = Field(description="0-5 score: Scientific Expedition Activity.")


# 2. 单个事件节点
class EventNode(BaseModel):
    start_date: str = Field(description="Event start date in YYYY-MM-DD format.")
    end_date: str = Field(description="Event end date in YYYY-MM-DD format. If it's a single-day event, this should be the same as start_date.")
    location_name: str = Field(description="Specific location name or region name.")
    geo_lat: Optional[float] = Field(description="Latitude of the location, if inferable. Otherwise null.")
    geo_lon: Optional[float] = Field(description="Longitude of the location, if inferable. Otherwise null.")
    domain: str = Field(description="The core thematic domain of this event. Must be exactly one of: 'Military', 'Technology', 'Politics', 'Environment', 'Governance', 'Resources'.")
    summary: str = Field(description="Concise summary of the action (max 10 words).")
    scores: IntensityScore = Field(description="Intensity assessment of this specific event.")
    source_ids: List[str] = Field(description="List of exact DOC_IDs from the input text that describe this event.")


# 3. 时空演变阶段
class EvolutionPhase(BaseModel):
    phase_name: str = Field(description="Name of the evolutionary phase (e.g., 'Incubation', 'Escalation', 'De-escalation').")
    start_date: str = Field(description="Start date of this phase in YYYY-MM-DD format.")
    end_date: str = Field(description="End date of this phase in YYYY-MM-DD format.")
    phase_summary: str = Field(description="Narrative summary of the entity's strategy or situation during this phase.")
    source_ids: List[str] = Field(description="List of DOC_IDs supporting this phase. Like ['001']")

# === 🌟 核心新增：实体战略画像 (主观推演) ===
class EntityStrategicInsights(BaseModel):
    behavioral_pattern: str = Field(
        description="Expert analysis of the entity's modus operandi (e.g., 'grey-zone tactics', 'technology-driven expansion').")
    hidden_intentions: str = Field(
        description="Deep analysis of the underlying strategic motives driving this entity's actions.")
    future_trajectory: str = Field(
        description="Forecast of the entity's likely next moves based on its current posture and historical habits.")


# 4. 完整输出结构
class DeepDiveOutput(BaseModel):
    entity_name: str = Field(description="The normalized name of the target entity.")
    entity_type: str = Field(description="Type: 'Vessel', 'Country', 'Organization', or 'Event'.")

    # === 🌟 双轨制输出 ===
    factual_grounding: List[Claim] = Field(
        description="Objective, chronological summary of the entity's actions. MUST be traced to source_ids.")
    strategic_insights: EntityStrategicInsights = Field(
        description="Deep, subjective behavioral profiling and predictions. No source_ids required.")
    evolution_phases: List[EvolutionPhase] = Field(
        description="Chronological phases telling the story of the entity's spatiotemporal evolution.")
    events: List[EventNode] = Field(description="Chronological list of discrete events associated with the entity.")

# --- 初始化组件 (通常在 graph 构建前完成) ---
# --- 初始化组件 ---
# 这里一定要用 llm_qw_thinking，因为涉及到人物画像和预测，需要极强的推理能力！
llm = llm_qw_thinking
parser = JsonOutputParser(pydantic_object=DeepDiveOutput)

prompt = PromptTemplate(
    template=DEEP_DIVE_PROMPT,
    input_variables=["target_entity", "news_context", "query", "blueprint_context", "output_language"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

deep_dive_chain = prompt | llm | parser


# --- 核心节点函数 ---
def deep_dive_agent(entity, query, docs, blueprint_context="", output_language="English"):
    print(f"\n--- 🔎 DEEP DIVE AGENT: Analyzing Spatiotemporal Evolution for {entity} ---")

    news_context = "\n".join(
        [f"[DOC_ID: {d.get('DOC_ID')}] - [{d.get('publish_date', 'N/A')}] {d.get('content', '')}" for d in docs])

    try:
        structured_result = deep_dive_chain.invoke({
            "target_entity": entity,
            "news_context": news_context,
            "query": query,
            "blueprint_context": blueprint_context,
            "output_language": output_language
        })
    except Exception as e:
        print(f"❌ Deep Dive 运行出错: {e}")
        return {"final_answer": f"Analysis failed: {str(e)}"}

    events = structured_result.get("events", [])

    # --- A. 计算雷达图数据 (已扩展为 5 个维度) ---
    radar_data = {"military": 0, "diplomatic": 0, "media": 0, "technology": 0, "scientific": 0}
    if events:
        df = pd.DataFrame([e['scores'] for e in events])
        # 确保 DataFrame 包含所有需要的列，防止某些事件遗漏字段导致报错
        for col in radar_data.keys():
            if col not in df.columns:
                df[col] = 0
        radar_data = df[list(radar_data.keys())].mean().round(1).to_dict()

    # --- B. 组装 Vega-Lite / ECharts 友好的数据 ---
    map_data = [
        {
            "date": e['start_date'],
            "lat": e['geo_lat'],
            "lon": e['geo_lon'],
            "name": e['location_name'],
            "type": e['domain'],
            "summary": e['summary'],
            "intensity": e['scores'].get('military', 1),  # 如果没有军事分，给个基础大小1
            "source_ids": e['source_ids']
        }
        for e in events if e.get('geo_lat') and e.get('geo_lon')
    ]

    gantt_data = [
        {
            "start": e['start_date'],
            "end": e['end_date'],
            "category": e['domain'],
            "intensity": max(e['scores'].values()),  # 🌟 甘特图的颜色深浅取该事件最高的一个分数
            "summary": e['summary'],
            "source_ids": e['source_ids']
        }
        for e in events
    ]

    # 3. 更新 State
    return {
        # 🌟 双轨输出：事实与洞察同时传递
        "factual_grounding": structured_result.get('factual_grounding', []),
        "strategic_insights": structured_result.get('strategic_insights', {}),

        # 结构化可视化数据
        "visualization_data": {
            "type": "deep_dive_dashboard",
            "evolution_timeline": structured_result.get('evolution_phases', []),
            "entity_info": {
                "name": structured_result.get('entity_name', entity),
                "type": structured_result.get('entity_type', 'Unknown')
            },
            "radar_chart": radar_data,
            "map_chart": map_data,
            "gantt_chart": gantt_data,
        }
    }

if __name__ == '__main__':
    query = "对2025年第四季度美国在深海采矿方面采取的一系列行动 time-range:2025-10-08 to 2025-12-31"
    news_list = get_news_by_id([])
    result = deep_dive_agent("美国", query, news_list, "美国从10月激进部署与规则博弈，转向12月因挪威暂停及国际压力而陷入治理僵局。")
    print(result)
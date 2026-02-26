from typing import List, Optional

import pandas as pd
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from agent.agents.base import Claim
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


# 2. 单个事件节点 (已增加溯源字段)
class EventNode(BaseModel):
    date: str = Field(description="Event date in YYYY-MM-DD format.")
    location_name: str = Field(description="Specific location name or region name.")
    geo_lat: Optional[float] = Field(description="Latitude of the location, if inferable. Otherwise null.")
    geo_lon: Optional[float] = Field(description="Longitude of the location, if inferable. Otherwise null.")
    action_type: str = Field(description="Category: 'Patrol', 'Drill', 'Statement', 'Conflict', 'Visit'.")
    summary: str = Field(description="Concise summary of the action (max 10 words).")
    scores: IntensityScore = Field(description="Intensity assessment of this specific event.")

    # 【核心新增】：证据追踪字段
    source_ids: List[str] = Field(description="List of exact DOC_IDs from the input text that describe this event.")


# 3. 完整输出结构
class DeepDiveOutput(BaseModel):
    entity_name: str = Field(description="The normalized name of the target entity.")
    entity_type: str = Field(description="Type: 'Vessel', 'Country', 'Organization', or 'Event'.")
    # overall_assessment: str = Field(description="A strategic summary of the entity's behavior (approx. 50 words).")
    overview_claims: List[Claim] = Field(description="Strategic summary of the entity's behavior broken down into traceable claims.")
    events: List[EventNode] = Field(description="Chronological list of events associated with the entity.")


# --- 初始化组件 (通常在 graph 构建前完成) ---
llm = llm_qw_quick
parser = JsonOutputParser(pydantic_object=DeepDiveOutput)

prompt = PromptTemplate(
    template=DEEP_DIVE_PROMPT,
    input_variables=["target_entity", "news_context"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

deep_dive_chain = prompt | llm | parser


# --- 核心节点函数 ---
def deep_dive_agent(entity, query, docs):
    """
    LangGraph Node: Performs deep dive analysis on a specific entity.
    Expected State: {'target_entity': str, 'retrieved_docs': list}
    """
    print(f"--- DEEP DIVE AGENT: Analyzing {entity} ---")

    # 1. 准备上下文
    #entity = state.get("target_entity", "Unknown Entity")
    # docs = state.get("retrieved_docs", [])

    # 简单的 Context 格式化
    news_context = "\n".join([f"[{d.get("DOC_ID")}]- [{d.get('publish_date', 'N/A')}] {d.get('content', '')}" for d in docs])

    # 2. 执行 LLM Chain
    # 返回的是符合 DeepDiveOutput 结构的 Dict
    try:
        structured_result = deep_dive_chain.invoke({
            "target_entity": entity,
            "news_context": news_context,
            "query" : query
        })
    except Exception as e:
        # 错误处理：返回空数据或错误提示，防止整个 Graph 崩溃
        return {"final_answer": f"Analysis failed: {str(e)}"}

    # 3. 内部数据处理 (In-Node Processing)
    # 我们在这里直接把 LLM 的结果转化为前端 Visualization 组件需要的格式
    # 这样前端 Vue 只需要“傻瓜式”渲染

    events = structured_result.get("events", [])

    # --- A. 计算雷达图数据 (Radar Data Aggregation) ---
    # 必须在这里算，LLM 算平均值容易出错
    radar_data = {"military": 0, "diplomatic": 0, "media": 0}
    if events:
        df = pd.DataFrame([e['scores'] for e in events])
        # 计算平均分并保留 1 位小数
        radar_data = df.mean().round(1).to_dict()

    # --- B. 组装 Vega-Lite 友好的数据 ---
    # Map Data: 只保留有坐标的点，或者前端处理空值
    map_data = [
        {
            "date": e['date'],
            "lat": e['geo_lat'],
            "lon": e['geo_lon'],
            "name": e['location_name'],
            "type": e['action_type'],
            "summary": e['summary']
        }
        for e in events if e.get('geo_lat') and e.get('geo_lon')
    ]

    # Gantt Data: 简单的转换
    gantt_data = [
        {
            "x": e['date'],  # Time
            "y": e['action_type'],  # Category
            "color": e['scores']['military'],  # Color by intensity
            "tooltip": e['summary']
        }
        for e in events
    ]

    # 4. 更新 State
    return {
        # 文本回复
        "final_answer": structured_result['overview_claims'],

        # 结构化可视化数据
        "visualization_data": {
            "type": "deep_dive_dashboard",  # 告诉前端用哪个面板渲染
            "entity_info": {
                "name": structured_result['entity_name'],
                "type": structured_result['entity_type']
            },
            "radar_chart": radar_data,
            "map_chart": map_data,
            "gantt_chart": gantt_data,
            "raw_events": events  # 保留原始数据供通过表格查看
        },
        "structured_insight" : structured_result
    }

if __name__ == '__main__':
    query = "对2025年第四季度美国在深海采矿方面采取的一系列行动"
    news_list = get_news_by_id([])
    result = deep_dive_agent("美国", query, news_list)
    print(result)
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser

from agent.agents.base import Claim
from agent.config.llm_config import llm_qw_quick
from agent.config.prompt_template import GLOBAL_MONITOR_PROMPT
from typing import List, Optional
from pydantic import BaseModel, Field

from agent.tools.news_manager import get_news_by_id

from pydantic import BaseModel, Field
from typing import List

# 1. 带有溯源的主题模型
class TopicNode(BaseModel):
    topic_name: str = Field(description="Name of the clustered topic")
    description: str = Field(description="A brief 1-sentence explanation of this topic")
    source_ids: List[str] = Field(description="List of DOC_IDs that belong to this topic")

# 2. 带有溯源的地理坐标点
class GeoPoint(BaseModel):
    lat: float = Field(description="Latitude of the event location")
    lon: float = Field(description="Longitude of the event location")
    topic_name: str = Field(description="Name of the topic this location belongs to")
    intensity: int = Field(description="Number of related news articles for this location")
    summary: str = Field(description="Short summary for the map tooltip")
    source_ids: List[str] = Field(description="List of DOC_IDs that support this geographic event")

# 3. 带有溯源的趋势时间点
class TimePoint(BaseModel):
    date: str = Field(description="Date in YYYY-MM-DD format")
    topic_name: str = Field(description="Name of the topic")
    count: int = Field(description="Number of articles for this topic on this date")
    source_ids: List[str] = Field(description="List of DOC_IDs published on this date for this topic")

# 4. 完整的输出模型
class GlobalMonitorOutput(BaseModel):
    # overview_summary: str = Field(description="A 100-word macro summary of the overall situation.")
    overview_claims: List[Claim] = Field(description="A macro summary broken down into traceable claims.")
    overview_source_ids: List[str] = Field(description="List of all DOC_IDs used to write the overview summary.")
    topics: List[TopicNode] = Field(description="List of top 3-5 identified topics with their sources.")
    geo_data: List[GeoPoint] = Field(description="Data for rendering the map.")
    trend_data: List[TimePoint] = Field(description="Data for rendering the ThemeRiver/AreaChart.")

# 1. 初始化 LLM
llm = llm_qw_quick

# 2. 设置 Parser
parser = JsonOutputParser(pydantic_object=GlobalMonitorOutput)

# 3. 构建 Chain
prompt = PromptTemplate(
    template=GLOBAL_MONITOR_PROMPT,
    input_variables=["query", "news_context"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

global_monitor_chain = prompt | llm | parser


# 4. 定义 LangGraph 节点函数
def global_monitor_agent(docs, query):
    """
    State dict should contain: 'query', 'retrieved_docs' (list of strings/dicts)
    """
    print("--- GLOBAL MONITOR AGENT: Analyzing... ---")

    # 获取上一步(Retrieval Tool) 检索到的新闻
    # docs = state.get("retrieved_docs", [])
    # query = state.get("query", "")

    # 格式化上下文 (将文档列表转为字符串)
    news_context = "\n".join([f"[{d.get("DOC_ID")}]- [{d['publish_date']}] {d['content']}" for d in docs])

    # 执行 Chain
    result = global_monitor_chain.invoke({
        "query": query,
        "news_context": news_context
    })

    # 更新 State
    # 这里我们把生成的 visualization_data 存入 state，方便后续传递给前端
    return {
        "final_answer": result['overview_claims'],
        "topics_list": result['topics_list'],
        "visualization_data": {
            "geo": result['geo_data'],
            "trend": result['trend_data']
        },
        "structured_insight": result  # 保留完整结构化数据
    }

if __name__ == '__main__':
    query = "对2025年第四季度全球海洋相关公开信源进行多维聚类与热力分析，识别高频、持续发酵、具政策转折意义的跨区域共性热点主题，覆盖五大本体维度"
    news_list = get_news_by_id([])
    result = global_monitor_agent(news_list, query)
    print(result)

# def global_monitor_node(state):
#     # 1. 获取检索到的新闻 (假设在 search_node 中已经存入 state)
#     # state['retrieved_docs'] = [{'id': 1, 'content': '...', 'date': '...'}, ...]
#     raw_docs = state.get("retrieved_docs", [])
#
#     # 2. 调用 LLM 进行打标 (Tagging)
#     # 这一步只让 LLM 返回分类结果，不让它数数
#     tagging_result = tagging_chain.invoke({"news_list_formatted": format_docs(raw_docs)})
#     # tagging_result = [{'id': 1, 'topic': 'A', 'location': '...'}, ...]
#
#     # 3. 【关键步骤】使用 Python 进行确定性统计 (Deterministic Aggregation)
#
#     # 将 LLM 的结果转为 DataFrame 方便处理
#     df_tags = pd.DataFrame(tagging_result)
#     df_docs = pd.DataFrame(raw_docs)
#
#     # 合并数据 (以 ID 为键)
#     merged_df = pd.merge(df_tags, df_docs, on="id")
#
#     # --- A. 统计 Topic 热度 (Count) ---
#     # 这一步计算出的数值是 100% 准确的
#     topic_counts = merged_df['topic'].value_counts().reset_index()
#     topic_counts.columns = ['topic_name', 'count']
#
#     # --- B. 生成时间演化数据 (Time Evolution) ---
#     # 因为我们在 raw_docs 里有准确的 date，这里直接 groupby 即可
#     # 这样你就无需单独的 Time-Evolution-Agent 了
#     trend_data = merged_df.groupby(['date', 'topic']).size().reset_index(name='count')
#
#     # --- C. 生成地理数据 (Geo Data) ---
#     # 可以取每个 Topic 下出现频率最高的 Location，或者保留所有散点
#     geo_data = merged_df[['lat', 'lon', 'topic', 'title']].to_dict(orient='records')
#
#     # 4. 构建最终输出
#     final_output = {
#         "topics_summary": topic_counts.to_dict(orient='records'),
#         "trend_data": trend_data.to_dict(orient='records'),
#         "geo_data": geo_data,
#         "raw_evidence": merged_df.to_dict(orient='records')  # 用于溯源验证
#     }
#
#     return {"visualization_data": final_output}
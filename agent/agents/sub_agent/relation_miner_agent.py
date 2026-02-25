from typing import List, Literal

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
import pandas as pd
from collections import defaultdict

from agent.agents.base import Claim
from agent.config.llm_config import llm_qw_quick
from agent.config.prompt_template import RELATION_MINER_PROMPT

from pydantic import BaseModel, Field
from typing import List, Literal


class RelationEdge(BaseModel):
    source: str = Field(description="The entity initiating the action (Subject).")
    target: str = Field(description="The entity receiving the action (Object).")
    relation_type: Literal["Conflict", "Cooperation", "Diplomacy", "Trade", "Other"] = Field(
        description="Classify the nature of the interaction."
    )
    description: str = Field(
        description="Short phrase describing the interaction (e.g., 'fired water cannon', 'signed treaty').")
    interaction_date: str = Field(description="YYYY-MM-DD when this interaction occurred.")
    is_causal: bool = Field(description="True if the text explicitly states source CAUSED target to react.")

    # 【核心新增】：证据追踪字段
    source_ids: List[str] = Field(
        description="List of exact DOC_IDs from the input text that describe this relationship.")


class RelationExtractionOutput(BaseModel):
    # summary: str = Field(description="Summary of the inter-entity dynamics.")
    overview_claims: List[Claim] = Field(description="Summary of the inter-entity dynamics broken down into traceable claims.")
    relations: List[RelationEdge] = Field(description="List of extracted relationships.")

llm = llm_qw_quick
parser = JsonOutputParser(pydantic_object=RelationExtractionOutput)

prompt = PromptTemplate(
    template=RELATION_MINER_PROMPT,
    input_variables=["focus_entities", "news_context"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

relation_chain = prompt | llm | parser

# ... 初始化 LLM 和 Parser (同上) ...

def relation_miner_agent(entities, news_list):
    """
    Extracts relations and builds a graph structure.
    """
    print(f"--- RELATION MINER: Analyzing network for {entities} ---")



    news_context = "\n".join([f"[{d.get("DOC_ID")}]- [{d['publish_date']}] {d['content']}" for d in news_list])
    # 1. 运行 LLM 提取关系
    # 为了防止上下文过长，这里可能需要分批处理(Batch Processing)再合并，
    # 但为了简化演示，我们假设一次性处理。
    try:
        raw_result = relation_chain.invoke({
            "focus_entities": ", ".join(entities),
            "news_context": news_context
        })
    except Exception as e:
        return {"final_answer": "Network analysis failed."}

    raw_relations = raw_result.get("relations", [])

    # 2. Python 聚合逻辑 (Aggregation)
    # 目标：将多条相似的边合并为一条，并增加权重 (Weight)

    # Key = (Source, Target, Type)
    # Value = Count
    edge_aggregator = defaultdict(int)
    edge_details = defaultdict(list)  # 存储具体的描述，用于 Tooltip

    nodes = set()

    for r in raw_relations:
        # 标准化实体名称 (简单处理)
        src = r['source'].strip()
        tgt = r['target'].strip()
        rtype = r['relation_type']

        # 记录节点
        nodes.add(src)
        nodes.add(tgt)

        # 聚合边
        key = (src, tgt, rtype)
        edge_aggregator[key] += 1
        edge_details[key].append(f"[{r['interaction_date']}] {r['description']}")

    # 3. 构建 Vega-Lite / ECharts 友好的图数据
    graph_nodes = [{"id": n, "group": "Entity"} for n in nodes]

    graph_links = []
    for (src, tgt, rtype), weight in edge_aggregator.items():
        # 获取最新的描述作为 Label
        details = edge_details[(src, tgt, rtype)]

        graph_links.append({
            "source": src,
            "target": tgt,
            "type": rtype,
            "value": weight,  # 线条粗细
            "label": details[0],  # 线条上的文字（取第一条）
            "tooltip": "\n".join(details[:3])  # 鼠标悬停显示前3条细节
        })

    # 4. 构建 Sankey 数据 (仅针对有因果关系的边)
    causal_links = [
        {"source": r['source'], "target": r['target'], "value": 1}
        for r in raw_relations if r.get('is_causal')
    ]

    return {
        "final_answer": raw_result['overview_claims'],
        "visualization_data": {
            "type": "relation_network",
            "graph_chart": {
                "nodes": graph_nodes,
                "links": graph_links
            },
            "sankey_chart": causal_links
        },
        "structured_insight": raw_result
    }



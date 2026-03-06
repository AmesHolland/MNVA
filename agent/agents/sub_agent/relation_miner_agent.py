from typing import List, Literal

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
import pandas as pd
from collections import defaultdict

from agent.agents.schemas import Claim
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
        description="List of exact DOC_IDs from the input text that describe this relationship. Like ['001']")


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

from collections import defaultdict


# --- 核心节点函数 ---
# 【新增】：接收 blueprint_context
def relation_miner_agent(entities, news_list, blueprint_context=""):
    """
    Extracts relations and builds a graph structure with full provenance.
    """
    print(f"--- 🕸️ RELATION MINER: Analyzing network for {entities} ---")

    # 格式化输入，确保 DOC_ID 存在
    news_context = "\n".join(
        [f"[DOC_ID: {d.get('DOC_ID', 'UNKNOWN')}] - [{d.get('publish_date', 'N/A')}] {d.get('content', '')}" for d in
         news_list])

    # 1. 运行 LLM 提取关系
    try:
        raw_result = relation_chain.invoke({
            "focus_entities": ", ".join(entities),
            "news_context": news_context,
            "blueprint_context": blueprint_context  # 【新增】：注入蓝图
        })
    except Exception as e:
        print(f"❌ Relation Miner 运行出错: {e}")
        return {"final_answer": [{"statement": "Network analysis failed.", "is_direct_quote": False, "source_ids": []}],
                "visualization_data": {}, "structured_insight": {}}

    raw_relations = raw_result.get("relations", [])

    # 2. Python 聚合逻辑 (Aggregation)
    edge_aggregator = defaultdict(int)
    edge_details = defaultdict(list)
    edge_dates = defaultdict(set)
    # 【核心修复】：新增 evidence_aggregator，用于存放合并后的溯源 IDs
    edge_sources = defaultdict(set)

    nodes = set()

    for r in raw_relations:
        src = r['source'].strip()
        tgt = r['target'].strip()
        rtype = r['relation_type']
        s_ids = r.get('source_ids', [])  # 提取该条关系的新闻ID
        # 提取这条具体关系的发生日期
        i_date = r.get('interaction_date', '')
        nodes.add(src)
        nodes.add(tgt)

        key = (src, tgt, rtype)
        edge_aggregator[key] += 1
        edge_details[key].append(f"[{r['interaction_date']}] {r['description']}")
        # 【核心修复】：将溯源 ID 塞入集合（自动去重）
        for sid in s_ids:
            edge_sources[key].add(sid)
        # 【新增】：将日期存入 Set 进行去重
        if i_date:
            edge_dates[key].add(i_date)

    # 3. 构建 Vega-Lite / ECharts 友好的图数据
    graph_nodes = [{"id": n, "group": "Entity"} for n in nodes]

    graph_links = []
    for (src, tgt, rtype), weight in edge_aggregator.items():
        details = edge_details[(src, tgt, rtype)]
        # 【核心修复】：将 set 转回 list 传给前端
        merged_source_ids = list(edge_sources[(src, tgt, rtype)])
        # 【新增】：将日期 Set 转为 List 传给前端
        active_dates_list = list(edge_dates[(src, tgt, rtype)])

        graph_links.append({
            "source": src,
            "target": tgt,
            "type": rtype,
            "value": weight,
            "label": details[0],
            "tooltip": "\n".join(details[:3]),
            "source_ids": merged_source_ids,  # 🌟 成功把证据链绑到了粗边上！
            "active_dates": active_dates_list  # 🌟 补齐了！前端现在知道这条线在哪些天亮起
        })

    # 4. 构建 Sankey 数据
    causal_links = [
        {
            "source": r['source'],
            "target": r['target'],
            "value": 1,
            "source_ids": r.get('source_ids', [])  # 桑基图也可以带上溯源！
        }
        for r in raw_relations if r.get('is_causal')
    ]

    return {
        # 注意：这里假设 raw_result 里已经按我们上一次的约定，把 summary 改为了 overview_claims
        "final_answer": raw_result.get('overview_claims', []),
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

import os
import json
import os
import traceback
from collections import Counter
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait, FIRST_COMPLETED
from typing import Any, List

from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

from agent.agents.sub_agent.agent_wrappers import AGENT_MAPPING
from agent.config.llm_config import llm_qw_quick
from agent.config.prompt_template import get_intent_prompt, get_data_profiling_prompt, INTEGRATING_PROMPT, \
    ANCHOR_PROMPT, PLANNING_PROMPT, get_profile_merge_prompt
from agent.tools.base import safe_parse_json
# 从上面定义的模块导入依赖
from ..schemas import ResearchState, FinalReport, SpatiotemporalBlueprint, ExecutionPlan
from ...rag.mysql_store import MySQLDB
from ...tools.news_manager import get_news_by_id

model = llm_qw_quick



from datetime import datetime

def intent_node(state: ResearchState) -> dict:
    print("\n" + "=" * 50 + "\n🧭 用户意图识别阶段...")

    # 沙盒请求直接跳过
    if state.get("intent") and state.get("intent").get("is_sandbox_request"):
        print("   沙盒请求，跳过意图识别。")
        return {
            "intent": state.get("intent"),
            "current_phase": "retrieving",
            "messages": [AIMessage(content="接收到沙盒分析请求，准备进行局部数据检索...")]
        }

    topic = state["research_topic"]
    output_language = state["output_language"]
    today = datetime.now().strftime("%Y-%m-%d")
    intent_prompt = get_intent_prompt(topic, today, output_language)

    try:
        response = model.invoke([HumanMessage(content=intent_prompt)])
        parsed_intent = safe_parse_json(response.content)
        intent = normalize_intent(parsed_intent, topic)
    except Exception as e:
        print(f"   ❌ 意图解析失败: {e}")
        intent = {
            "original_query": topic,
            "task_complexity": "deep_research",
            "reasoning": "Fallback intent due to parsing failure.",
            "primary_intent": "comprehensive_situation_analysis",
            "analysis_mode": "mixed",
            "spatial_scope": [],
            "entities": [],
            "temporal_scope": {"start": "", "end": "", "type": "none"},
            "retrieval_plan": {
                "use_full_dataset": True,
                "keywords": [],
                "date_from": "",
                "date_to": "",
                "sort_by": "relevance",
                "rationale": "Fallback to full-dataset retrieval."
            }
        }

    print(f"   primary_intent: {intent.get('primary_intent')}")
    print(f"   analysis_mode: {intent.get('analysis_mode')}")
    print(f"   retrieval keywords: {intent.get('retrieval_plan', {}).get('keywords', [])}")

    next_phase = "responding" if intent.get("task_complexity") == "simple_qa" else "retrieving"

    return {
        "intent": intent,
        "research_topic": topic,
        "current_phase": next_phase,
        "messages": [AIMessage(content=f"用户意图已解读完成：{topic}")]
    }

def normalize_intent(intent: dict, topic: str) -> dict:
    retrieval_plan = intent.get("retrieval_plan") or {}
    temporal_scope = intent.get("temporal_scope") or {}

    return {
        "original_query": intent.get("original_query") or topic,
        "task_complexity": intent.get("task_complexity", "deep_research"),
        "reasoning": intent.get("reasoning", ""),
        "primary_intent": intent.get("primary_intent", "comprehensive_situation_analysis"),
        "analysis_mode": intent.get("analysis_mode", "mixed"),
        "spatial_scope": intent.get("spatial_scope", []),
        "entities": intent.get("entities", []),
        "temporal_scope": {
            "start": temporal_scope.get("start", ""),
            "end": temporal_scope.get("end", ""),
            "type": temporal_scope.get("type", "none"),
        },
        "retrieval_plan": {
            "use_full_dataset": retrieval_plan.get("use_full_dataset", True),
            "keywords": retrieval_plan.get("keywords", []),
            "date_from": retrieval_plan.get("date_from", ""),
            "date_to": retrieval_plan.get("date_to", ""),
            "sort_by": retrieval_plan.get("sort_by", "relevance"),
            "rationale": retrieval_plan.get("rationale", ""),
        },
    }

def route_after_intent(state: ResearchState) -> str:
    """根据意图复杂度进行双轨路由"""
    # 如果是沙盒请求，直接进入深度分析
    if state.get("intent", {}).get("is_sandbox_request"):
        return "data_retrieval"
        
    complexity = state.get("intent", {}).get("task_complexity", "deep_research")
    if complexity == "simple_qa":
        return "simple_chat"
    else:
        return "data_retrieval" # 进入深度分析的起点

def simple_chat_node(state: ResearchState) -> dict:
    """快分支：处理基础问答、闲聊或简单解释"""
    print("\n" + "=" * 50 + "\n💬 进入轻量级问答分支 (Fast Track)...")
    topic = state["research_topic"]
    
    # 获取历史消息
    history = state.get("messages", [])
    
    # 构造包含历史的 prompt
    prompt_messages = []
    for msg in history:
        if isinstance(msg, HumanMessage):
            prompt_messages.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            prompt_messages.append(f"AI: {msg.content}")
    
    prompt = f"你是一个海洋态势感知系统的智能助手。请根据以下对话历史，简明扼要地回答用户的最新问题。\n\n对话历史:\n{''.join(prompt_messages)}\n\n最新问题: {topic}"
    
    response = model.invoke([HumanMessage(content=prompt)])

    return {
        "current_phase": "ready",
        "final_report": response.content,
        "messages": [AIMessage(content=response.content)]
    }

# =========================================================
# 1) 通用小工具
# =========================================================

def _normalize_keywords(words: List[Any], limit: int = 8) -> List[str]:
    if not words:
        return []
    result = []
    seen = set()
    for w in words:
        w = str(w).strip()
        if not w:
            continue
        key = w.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(w)
        if len(result) >= limit:
            break
    return result


def _normalize_group_list(groups: List[Any], group_limit: int = 6, item_limit: int = 6) -> List[List[str]]:
    """
    把 [["USA", "United States"], ["South China Sea", "SCS"]] 规范化。
    """
    normalized = []
    seen_group_signatures = set()

    if not groups:
        return normalized

    for g in groups[:group_limit]:
        if not isinstance(g, list):
            continue
        cleaned = _normalize_keywords(g, limit=item_limit)
        if not cleaned:
            continue

        sig = tuple(sorted(x.lower() for x in cleaned))
        if sig in seen_group_signatures:
            continue
        seen_group_signatures.add(sig)
        normalized.append(cleaned)

    return normalized

# =========================================================
# 2) SQL 查询辅助
# =========================================================

def mysql_select(sql, param):
    db = MySQLDB(
        host="localhost",
        port=3306,
        user="root",
        password="123456",
        database="marine_news_db"
    )

    results = db.query(sql)
    print("30秒后查询结果：", results)
    for res in results:
        print(res)

def fetch_all_news_by_dataset_id(dataset_id: int, date_from: str = "", date_to: str = "") -> List[dict]:
    print("fetch_all_news_by_dataset_id", dataset_id, date_from, date_to)
    where_clauses = ["dataset_id = %s"]
    params: List[Any] = [dataset_id]

    if date_from:
        where_clauses.append("publish_date >= %s")
        params.append(date_from)

    if date_to:
        where_clauses.append("publish_date <= %s")
        params.append(date_to)

    sql = f"""
        SELECT
            id,
            title,
            content,
            publish_date,
            url,
            source
        FROM news
        WHERE {' AND '.join(where_clauses)}
        ORDER BY publish_date DESC, id DESC
        LIMIT 500
    """
    print(sql)
    return mysql_select(sql, tuple(params))


def fetch_news_by_sandbox_constraints(dataset_id: int, constraints: dict) -> List[dict]:
    start_time = constraints.get("start_time", "")
    end_time = constraints.get("end_time", "")
    spatial_labels = _normalize_keywords(constraints.get("spatial_labels", []), limit=8)

    where_clauses = ["dataset_id = %s"]
    params: List[Any] = [dataset_id]

    if start_time:
        where_clauses.append("publish_date >= %s")
        params.append(start_time)

    if end_time:
        where_clauses.append("publish_date <= %s")
        params.append(end_time)

    # 先用 title/content 的 OR 匹配做硬过滤
    if spatial_labels:
        label_parts = []
        for label in spatial_labels:
            label_parts.append("(title LIKE %s OR content LIKE %s)")
            params.append(f"%{label}%")
            params.append(f"%{label}%")
        where_clauses.append("(" + " OR ".join(label_parts) + ")")

    sql = f"""
        SELECT
            id,
            title,
            content,
            publish_date,
            url,
            source
        FROM news
        WHERE {' AND '.join(where_clauses)}
        ORDER BY publish_date DESC, id DESC
        LIMIT 300
    """
    return mysql_select(sql, tuple(params))


def fetch_news_by_compiled_rewrite(dataset_id: int, rewritten_plan: dict) -> List[dict]:
    """
    rewritten_plan 结构示例：
    {
      "use_full_dataset": false,
      "date_from": "2024-01-01",
      "date_to": "2024-12-31",
      "must_groups": [
        ["United States", "USA", "U.S.", "America"],
        ["South China Sea", "SCS"]
      ],
      "optional_groups": [
        ["coast guard", "maritime law enforcement"],
        ["navy", "military exercise"]
      ],
      "sort_by": "relevance"
    }
    """

    date_from = rewritten_plan.get("date_from", "") or ""
    date_to = rewritten_plan.get("date_to", "") or ""
    print("Must group")
    must_groups = _normalize_group_list(rewritten_plan.get("must_groups", []), group_limit=6, item_limit=6)
    optional_groups = _normalize_group_list(rewritten_plan.get("optional_groups", []), group_limit=8, item_limit=6)
    sort_by = rewritten_plan.get("sort_by", "relevance")

    where_clauses = ["dataset_id = %s"]
    where_params: List[Any] = [dataset_id]

    if date_from:
        where_clauses.append("publish_date >= %s")
        where_params.append(date_from)

    if date_to:
        where_clauses.append("publish_date <= %s")
        where_params.append(date_to)

    # must_groups: 每个 group 内 OR，不同 group 之间 AND
    # 例如：
    # (title LIKE '%USA%' OR content LIKE '%USA%' OR title LIKE '%America%' ...)
    # AND
    # (title LIKE '%South China Sea%' OR content LIKE ...)
    for group in must_groups:
        sub_parts = []
        for phrase in group:
            sub_parts.append("(title LIKE %s OR content LIKE %s)")
            where_params.append(f"%{phrase}%")
            where_params.append(f"%{phrase}%")
        where_clauses.append("(" + " OR ".join(sub_parts) + ")")

    # relevance score
    # must group 中的词也参与打分，但 optional 更体现排序差异
    score_parts = []
    score_params: List[Any] = []

    for group in must_groups:
        for phrase in group:
            score_parts.append("(CASE WHEN title LIKE %s THEN 4 ELSE 0 END)")
            score_params.append(f"%{phrase}%")
            score_parts.append("(CASE WHEN content LIKE %s THEN 2 ELSE 0 END)")
            score_params.append(f"%{phrase}%")

    for group in optional_groups:
        for phrase in group:
            score_parts.append("(CASE WHEN title LIKE %s THEN 2 ELSE 0 END)")
            score_params.append(f"%{phrase}%")
            score_parts.append("(CASE WHEN content LIKE %s THEN 1 ELSE 0 END)")
            score_params.append(f"%{phrase}%")

    relevance_sql = "0"
    if score_parts:
        relevance_sql = " + ".join(score_parts)

    order_sql = "publish_date DESC, id DESC"
    if sort_by == "relevance" and score_parts:
        order_sql = "relevance_score DESC, publish_date DESC, id DESC"

    sql = f"""
        SELECT
            id,
            title,
            content,
            publish_date,
            url,
            source,
            ({relevance_sql}) AS relevance_score
        FROM news
        WHERE {' AND '.join(where_clauses)}
        ORDER BY {order_sql}
        LIMIT 300
    """
    print("mysql select")
    params = tuple(score_params + where_params)
    print(params)
    return mysql_select(sql, params)


# =========================================================
# 3) LLM 检索重写
# =========================================================

def get_retrieval_rewrite_prompt(query: str, intent: dict, today: str) -> str:
    retrieval_plan = intent.get("retrieval_plan", {})
    entities = intent.get("entities", [])
    spatial_scope = intent.get("spatial_scope", [])
    temporal_scope = intent.get("temporal_scope", {})

    return f"""
# Context
Today is {today}.

Original user query:
{query}

Intent JSON:
{json.dumps(intent, ensure_ascii=False, indent=2)}

# Role
You are a retrieval-query rewriting assistant for a maritime news analysis system.

# Goal
Convert the coarse retrieval plan into a SQL-compilable retrieval rewrite plan.

# Important
1. Do NOT generate SQL.
2. You must only output JSON.
3. The downstream backend will compile your JSON into parameterized SQL.
4. Expand aliases / synonyms conservatively.
5. Do not invent dates or entities not implied by the query.
6. Prefer high-precision rewriting over overly broad expansion.

# Output JSON Schema
{{
  "use_full_dataset": true,
  "date_from": "YYYY-MM-DD or empty string",
  "date_to": "YYYY-MM-DD or empty string",
  "must_groups": [
    ["canonical term", "alias 1", "alias 2"]
  ],
  "optional_groups": [
    ["related term 1", "related term 2"]
  ],
  "sort_by": "relevance | time_desc",
  "rationale": "brief explanation"
}}

# Rewriting Rules
1. Each item in must_groups represents one concept group.
   - Terms inside a group are aliases/synonyms/alternate forms of the same concept.
   - Different groups mean different required concepts.
2. optional_groups are relevance boosters only, not strict filters.
3. If the query is broad and exploratory, keep must_groups compact and conservative.
4. If the query is very specific, use more precise must_groups.
5. Normalize time expressions such as:
   - recently
   - this year
   - past 3 years
   - last 6 months
6. Example:
   - "United States" can expand to ["United States", "USA", "U.S.", "America"]
   - "coast guard" can expand to ["coast guard", "maritime law enforcement"]
7. If the upstream retrieval_plan says use_full_dataset=true, you may keep must_groups empty.

# Upstream retrieval hints
- entities: {json.dumps(entities, ensure_ascii=False)}
- spatial_scope: {json.dumps(spatial_scope, ensure_ascii=False)}
- temporal_scope: {json.dumps(temporal_scope, ensure_ascii=False)}
- coarse retrieval_plan: {json.dumps(retrieval_plan, ensure_ascii=False, indent=2)}

# Output JSON only
"""


def normalize_rewritten_plan(raw_plan: dict, coarse_plan: dict) -> dict:
    coarse_keywords = _normalize_keywords(coarse_plan.get("keywords", []), limit=6)

    use_full_dataset = raw_plan.get("use_full_dataset", coarse_plan.get("use_full_dataset", True))
    sort_by = raw_plan.get("sort_by", coarse_plan.get("sort_by", "relevance"))
    date_from = raw_plan.get("date_from", coarse_plan.get("date_from", ""))
    date_to = raw_plan.get("date_to", coarse_plan.get("date_to", ""))

    must_groups = _normalize_group_list(raw_plan.get("must_groups", []), group_limit=6, item_limit=6)
    optional_groups = _normalize_group_list(raw_plan.get("optional_groups", []), group_limit=8, item_limit=6)

    # 如果 LLM 没给出 must_groups，但 coarse keywords 有内容，则退回一词一组
    if not must_groups and coarse_keywords and not use_full_dataset:
        must_groups = [[kw] for kw in coarse_keywords]

    # broad 模式下允许空 must_groups
    return {
        "use_full_dataset": bool(use_full_dataset),
        "date_from": str(date_from or ""),
        "date_to": str(date_to or ""),
        "must_groups": must_groups,
        "optional_groups": optional_groups,
        "sort_by": sort_by if sort_by in {"relevance", "time_desc"} else "relevance",
        "rationale": raw_plan.get("rationale", ""),
    }


def rewrite_retrieval_plan_with_llm(query: str, intent: dict) -> dict:

    coarse_plan = intent.get("retrieval_plan", {})
    today = datetime.now().strftime("%Y-%m-%d")
    prompt = get_retrieval_rewrite_prompt(query, intent, today)

    try:
        response = model.invoke([HumanMessage(content=prompt)])
        raw = safe_parse_json(response.content)
        rewritten = normalize_rewritten_plan(raw, coarse_plan)
        return rewritten
    except Exception as e:
        print(f"   ⚠️ retrieval rewrite 失败，使用 coarse retrieval_plan 回退: {e}")

        coarse_keywords = _normalize_keywords(coarse_plan.get("keywords", []), limit=6)
        return {
            "use_full_dataset": bool(coarse_plan.get("use_full_dataset", True)),
            "date_from": coarse_plan.get("date_from", "") or "",
            "date_to": coarse_plan.get("date_to", "") or "",
            "must_groups": [] if coarse_plan.get("use_full_dataset", True) else [[kw] for kw in coarse_keywords],
            "optional_groups": [],
            "sort_by": coarse_plan.get("sort_by", "relevance"),
            "rationale": "Fallback rewritten plan from coarse retrieval_plan.",
        }


# =========================================================
# 4) 主节点
# =========================================================

def data_retrieval_node(state: dict):
    print("\n" + "=" * 50 + "\n🔎 数据检索阶段...")

    intent = state.get("intent", {})
    query = state.get("research_topic", "")
    dataset_id = state.get("dataset_id") or state.get("selected_dataset_id")

    if not dataset_id:
        raise ValueError("data_retrieval_node 缺少 dataset_id / selected_dataset_id")

    # simple qa 不需要检索
    if intent.get("task_complexity") == "simple_qa":
        print("   ℹ️ simple_qa，跳过数据检索。")
        return {
            "news_list": [],
            "retrieval_meta": {
                "mode": "skip_for_simple_qa",
                "dataset_id": dataset_id,
            }
        }

    # ---------------------------
    # 沙盒模式：硬过滤
    # ---------------------------
    if intent.get("is_sandbox_request"):
        constraints = intent.get("sandbox_constraints", {})
        news_list = fetch_news_by_sandbox_constraints(dataset_id, constraints)

        retrieval_meta = {
            "mode": "sandbox_hard_filter",
            "dataset_id": dataset_id,
            "constraints": constraints,
            "retrieved_count": len(news_list),
        }

        print(f"   ✅ sandbox 检索完成，命中 {len(news_list)} 条新闻")
        return {
            "news_list": news_list,
            "retrieval_meta": retrieval_meta,
        }

    # ---------------------------
    # 正常模式：coarse plan -> LLM rewrite -> code-compiled SQL
    # ---------------------------
    coarse_plan = intent.get("retrieval_plan", {})
    rewritten_plan = rewrite_retrieval_plan_with_llm(query, intent)
    print("rewritten_plan: ", rewritten_plan)
    use_full_dataset = rewritten_plan.get("use_full_dataset", True)
    date_from = rewritten_plan.get("date_from", "")
    date_to = rewritten_plan.get("date_to", "")
    print("Yes or no")
    if use_full_dataset:
        print("use_full_dataset")
        # news_list = fetch_all_news_by_dataset_id(
        #     dataset_id=dataset_id,
        #     date_from=date_from,
        #     date_to=date_to,
        # )
        news_list = get_news_by_id([])
        retrieval_meta = {
            "mode": "full_dataset",
            "dataset_id": dataset_id,
            "coarse_plan": coarse_plan,
            "rewritten_plan": rewritten_plan,
            "retrieved_count": len(news_list),
            "fallback_to_full_dataset": False,
        }
    else:
        # news_list = fetch_news_by_compiled_rewrite(dataset_id, rewritten_plan)
        news_list = get_news_by_id([])
        print("Not use_full_dataset")
        # 小样本回退
        if len(news_list) < 10:
            fallback_news = fetch_all_news_by_dataset_id(
                dataset_id=dataset_id,
                date_from=date_from,
                date_to=date_to,
            )

            retrieval_meta = {
                "mode": "keyword_date_with_llm_rewrite",
                "dataset_id": dataset_id,
                "coarse_plan": coarse_plan,
                "rewritten_plan": rewritten_plan,
                "retrieved_count": len(news_list),
                "fallback_to_full_dataset": True,
                "fallback_count": len(fallback_news),
            }
            news_list = fallback_news
        else:
            retrieval_meta = {
                "mode": "keyword_date_with_llm_rewrite",
                "dataset_id": dataset_id,
                "coarse_plan": coarse_plan,
                "rewritten_plan": rewritten_plan,
                "retrieved_count": len(news_list),
                "fallback_to_full_dataset": False,
            }

    print(f"   ✅ 数据检索完成，最终用于分析的新闻数: {len(news_list)}")

    return {
        "news_list": news_list,
        "retrieval_meta": retrieval_meta,
    }


profiler_model = model   # 建议用快模型；没有就改成 model
BATCH_SIZE = 20
MAX_WORKERS = 4


def chunk_list(lst, size):
    return [lst[i:i + size] for i in range(0, len(lst), size)]


def profile_one_batch(news_batch: list) -> dict:
    prompt = get_data_profiling_prompt(news_batch)
    response = profiler_model.invoke([HumanMessage(content=prompt)])
    return safe_parse_json(response.content)


def merge_profile_results(batch_profiles: list) -> dict:
    if len(batch_profiles) == 1:
        return batch_profiles[0]

    prompt = get_profile_merge_prompt(batch_profiles)
    response = profiler_model.invoke([HumanMessage(content=prompt)])
    return safe_parse_json(response.content)


def data_profiling_node(state: ResearchState) -> dict:
    print("\n" + "=" * 50 + "\n📊 数据探路 (Data Profiling)...")
    news_list = state.get("news_list", [])

    if not news_list:
        return {"analysis_results": {"data_profile": {}}}

    batches = chunk_list(news_list, BATCH_SIZE)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        batch_profiles = list(executor.map(profile_one_batch, batches))

    profile_data = merge_profile_results(batch_profiles)

    # 时间统计交给代码做
    dates = sorted([str(n.get("publish_date", ""))[:10] for n in news_list if n.get("publish_date")])
    time_range_arr = [dates[0], dates[-1]] if dates else ["", ""]
    actual_time_range = f"{time_range_arr[0]} to {time_range_arr[1]}" if dates else ""

    month_counts = Counter(d[:7] for d in dates if len(d) >= 7)
    temporal_signals = [{"time": m, "count": c} for m, c in sorted(month_counts.items())]

    default_date = time_range_arr[0] if time_range_arr[0] else ""
    preview_map_data = [
        {
            "topic_name": item.get("name", "Unknown"),
            "lon": item["coord"][0],
            "lat": item["coord"][1],
            "intensity": item.get("intensity", 5),
            "summary": f"[{item.get('type', 'region')}] Identified during data profiling.",
            "date": default_date,
            "source_ids": []
        }
        for item in profile_data.get("geo_coordinates", [])
        if item.get("coord") and len(item["coord"]) == 2
    ]

    grounded_entities = list(dict.fromkeys(
        profile_data.get("actual_countries", []) + profile_data.get("actual_entities", [])
    ))

    profile_data["news_count"] = len(news_list)
    profile_data["batch_count"] = len(batches)
    profile_data["actual_time_range"] = actual_time_range
    profile_data["time_range_arr"] = time_range_arr
    profile_data["temporal_signals"] = temporal_signals
    profile_data["grounded_entities"] = grounded_entities
    profile_data["preview_map_data"] = preview_map_data

    return {
        "analysis_results": {
            "data_profile": profile_data
        }
    }

# === 4. 节点函数实现 ===
def spatiotemporal_scoping_anchor_node(state: dict) -> dict:
    """
    时空范围锚定节点：读取轻量级元数据骨架，生成时空演化蓝图 (Blueprint)
    """
    print("\n" + "=" * 50)
    print("🧭 进入时空范围锚定节点 (Spatiotemporal Scoping Anchor)...")

    output_language = state["output_language"]

    # === 2. 初始化 Parser 与 Prompt ===
    anchor_parser = JsonOutputParser(pydantic_object=SpatiotemporalBlueprint)

    anchor_prompt = PromptTemplate(
        template=ANCHOR_PROMPT,
        input_variables=["intent", "metadata_skeleton", "output_language"],
        partial_variables={"format_instructions": anchor_parser.get_format_instructions()}
    )

    # === 3. 组装 Chain (注意将 model 替换为你实际的 llm 实例) ===
    anchor_chain = anchor_prompt | model | anchor_parser

    intent = state.get("intent", {})
    raw_news_list = state.get("news_list", [])

    if not raw_news_list:
        print("   ⚠️ 未检索到任何新闻数据，跳过锚定。")
        return {"spatiotemporal_blueprint": None}

    # 1. 组装极致轻量级的“新闻骨架” (极低 Token 消耗)
    skeleton_lines = []
    for news in raw_news_list:
        date = news.get("publish_date", "Unknown Date")
        title = news.get("title", "No Title")

        # 兼容列表或字符串形式的 region/country
        loc_val = news.get("region", [])
        locs = ", ".join(loc_val) if isinstance(loc_val, list) else str(loc_val)

        ent_val = news.get("country", [])
        ents = ", ".join(ent_val) if isinstance(ent_val, list) else str(ent_val)

        summary = news.get("summary", "")

        line = f"[{date}] Title: {title} | Loc: {locs} | Ent: {ents} | Sum: {summary}"
        skeleton_lines.append(line)

    metadata_skeleton_str = "\n".join(skeleton_lines)

    # 2. 调用大模型生成蓝图
    print("   🧠 正在审视全局时空骨架，生成演化蓝图 (JsonOutputParser)...")
    try:
        # invoke 直接返回符合 SpatiotemporalBlueprint 结构的字典
        blueprint_dict = anchor_chain.invoke({
            "intent": json.dumps(intent, ensure_ascii=False),
            "metadata_skeleton": metadata_skeleton_str,
            "output_language":output_language
        })

        print(f"   ✅ 蓝图生成成功！共切分为 {len(blueprint_dict.get('phases', []))} 个阶段。")
        for phase in blueprint_dict.get('phases', []):
            print(
                f"      - Phase {phase.get('phase_id')}: [{phase.get('spatial_scale')}] {phase.get('phase_name')} ({phase.get('spatial_focus')})")

    except Exception as e:
        print(f"   ❌ 蓝图生成失败: {e}")
        blueprint_dict = None

    # 3. 将蓝图写入 State
    return {
        "spatiotemporal_blueprint": blueprint_dict
    }

def normalize_plan(plan: dict, blueprint: dict, profile_data: dict) -> dict:
    allowed_agents = {
        "Global_Monitor_Agent",
        "Deep_Dive_Agent",
        "Relation_Miner_Agent",
    }

    allowed_entities = set(profile_data.get("actual_entities", []))
    allowed_phase_ids = {
        int(p["phase_id"])
        for p in blueprint.get("phases", [])
        if "phase_id" in p
    }

    normalized_tasks = []
    seen_ids = set()

    for i, task in enumerate(plan.get("tasks", []), start=1):
        agent = task.get("agent")
        if agent not in allowed_agents:
            continue

        task_id = str(task.get("task_id") or f"task_{i}")
        if task_id in seen_ids:
            task_id = f"{task_id}_{i}"
        seen_ids.add(task_id)

        dependency = task.get("dependency") or []
        if not isinstance(dependency, list):
            dependency = [dependency]
        dependency = [str(d) for d in dependency if d]

        target_phase_ids = task.get("target_phase_ids") or []
        target_phase_ids = [
            int(pid) for pid in target_phase_ids
            if int(pid) in allowed_phase_ids
        ]

        # 若 planner 没给出合法 phase，就保底给全阶段
        if not target_phase_ids:
            target_phase_ids = sorted(allowed_phase_ids)

        args = dict(task.get("args", {}))

        # grounded entity 保底修正
        if agent == "Deep_Dive_Agent":
            ent = args.get("target_entity")
            if ent not in allowed_entities:
                # 找不到就跳过这个 task，或者改成 Global_Monitor_Agent 也行
                continue

        if agent == "Relation_Miner_Agent":
            print("===================")
            print(args)
            print(args.get("focus_entities"))
            print(args["focus_entities"])
            print("===================")
            focus_entities = args.get("focus_entities", [])
            if not isinstance(focus_entities, list):
                focus_entities = []
            # focus_entities = [e for e in focus_entities if e in allowed_entities]
            if len(focus_entities) < 2:
                continue
            args["focus_entities"] = focus_entities

        normalized_tasks.append({
            "task_id": task_id,
            "agent": agent,
            "action": task.get("action", ""),
            "target_phase_ids": target_phase_ids,
            "args": args,
            "dependency": dependency,
        })

    valid_ids = {t["task_id"] for t in normalized_tasks}
    for task in normalized_tasks:
        task["dependency"] = [d for d in task["dependency"] if d in valid_ids and d != task["task_id"]]

    return {
        "total_plan_logic": plan.get("total_plan_logic", ""),
        "tasks": normalized_tasks
    }

def get_overlapping_phase_ids(blueprint: dict, start_time: str, end_time: str) -> list[int]:
    if not start_time or not end_time:
        return [int(p["phase_id"]) for p in blueprint.get("phases", []) if "phase_id" in p]

    s = datetime.fromisoformat(start_time)
    e = datetime.fromisoformat(end_time)

    overlap_ids = []
    for phase in blueprint.get("phases", []):
        try:
            ps = datetime.fromisoformat(phase["start_date"])
            pe = datetime.fromisoformat(phase["end_date"])
            if not (pe < s or ps > e):
                overlap_ids.append(int(phase["phase_id"]))
        except Exception:
            continue

    return overlap_ids or [int(p["phase_id"]) for p in blueprint.get("phases", []) if "phase_id" in p]

def planning_node(state: dict) -> dict:
    print("\n" + "=" * 50)
    print("📋 [Planner] 进入任务规划阶段 (Task Orchestration)...")

    intent = state.get("intent", {})
    blueprint = state.get("spatiotemporal_blueprint", {})
    profile_data = state.get("analysis_results", {}).get("data_profile", {})
    output_language = state["output_language"]
    if intent.get("is_sandbox_request"):
        constraints = intent.get("sandbox_constraints", {})
        sandbox_phase_ids = get_overlapping_phase_ids(
            blueprint=blueprint,
            start_time=constraints.get("start_time"),
            end_time=constraints.get("end_time"),
        )

        plan = {
            "total_plan_logic": (
                f"Sandbox drill-down plan. The analysis scope is constrained to "
                f"{constraints.get('start_time')} - {constraints.get('end_time')}. "
                f"Tasks are routed only to the overlapping phases {sandbox_phase_ids}."
            ),
            "tasks": [
                {
                    "task_id": "sandbox_deep_dive",
                    "agent": "Deep_Dive_Agent",
                    "action": "Trace the micro-level spatiotemporal storyline inside the selected sandbox scope",
                    "target_phase_ids": sandbox_phase_ids,
                    "args": {
                        "target_entity": "AUTO_FROM_SANDBOX_CONTEXT"
                    },
                    "dependency": []
                },
                {
                    "task_id": "sandbox_relation_mining",
                    "agent": "Relation_Miner_Agent",
                    "action": "Mine conflict/cooperation relations inside the selected sandbox scope",
                    "target_phase_ids": sandbox_phase_ids,
                    "args": {
                        "focus_entities": ["AUTO_ENTITY_A", "AUTO_ENTITY_B"]
                    },
                    "dependency": []
                }
            ]
        }

    else:
        print("not is_sandbox_request")
        plan_parser = JsonOutputParser(pydantic_object=ExecutionPlan)

        planning_prompt = PromptTemplate(
            template=PLANNING_PROMPT,
            input_variables=[
                "user_query",
                "intent_data",
                "review",
                "blueprint",
                "actual_entities",
                "actual_topics",
                "data_richness",
                "output_language"
            ],
            partial_variables={
                "format_instructions": plan_parser.get_format_instructions()
            }
        )

        user_query = intent.get("original_query", "")
        feedback = state.get("user_feedback", "")
        plan_history = state.get("plan")

        review = ""
        if feedback and feedback != "approve":
            review = (
                "[Human-in-the-Loop Feedback]\n"
                f"The user rejected the previous plan: {json.dumps(plan_history, ensure_ascii=False)}\n"
                f"User revision request: {feedback}\n"
                "You must revise the plan accordingly."
            )

        print("   🧠 正在解析时空蓝图，生成动态数据路由计划...")

        try:
            planning_chain = planning_prompt | model | plan_parser

            raw_plan = planning_chain.invoke({
                "user_query": user_query,
                "intent_data": json.dumps(intent, ensure_ascii=False, indent=2),
                "review": review,
                "blueprint": json.dumps(blueprint, ensure_ascii=False, indent=2),
                "actual_entities": json.dumps(profile_data.get("actual_entities", []), ensure_ascii=False),
                "actual_topics": json.dumps(profile_data.get("actual_topics", []), ensure_ascii=False),
                "data_richness": profile_data.get("data_richness", "unknown"),
                "output_language" : output_language
            })
            print("raw plan")
            plan = normalize_plan(raw_plan, blueprint, profile_data)
            print(plan)
            if not plan["tasks"]:
                raise ValueError("Planner returned no valid tasks after normalization.")

            print(f"   ✅ 规划逻辑: {plan.get('total_plan_logic')}")

        except Exception as e:
            print(f"   ❌ 规划解析失败: {e}")
            all_phase_ids = [
                int(p["phase_id"])
                for p in blueprint.get("phases", [])
                if "phase_id" in p
            ] or [1]

            plan = {
                "total_plan_logic": "Planner parsing failed. Fallback to a conservative macro monitoring task.",
                "tasks": [{
                    "task_id": "fallback_global_monitor",
                    "agent": "Global_Monitor_Agent",
                    "action": "Fallback macro monitoring",
                    "target_phase_ids": all_phase_ids,
                    "args": {"query": user_query},
                    "dependency": []
                }]
            }

    return {
        "plan": plan,
        "current_phase": "confirming",
        "messages": [{
            "role": "ai",
            "content": f"已规划任务：{plan.get('total_plan_logic')}"
        }]
    }

def check_node(state: ResearchState) -> dict:
    feedback = state.get('user_feedback')
    print(f"--- 接收到用户反馈: {feedback} ---")
    return {}

def route_after_check(state: ResearchState) -> str:
    feedback = state.get("user_feedback", "")
    return "analysis" if feedback == "approve" else "planning"


def _to_list(value):
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _parse_datetime_safe(value):
    if not value:
        return None
    if isinstance(value, datetime):
        return value

    text = str(value).strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except Exception:
        pass

    # 兼容常见日期格式
    for fmt in [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
    ]:
        try:
            return datetime.strptime(text, fmt)
        except Exception:
            continue
    return None


def _extract_news_datetime(news_item: dict):
    """
    按常见字段名提取新闻时间。
    你如果自己的字段名固定，可以只保留一个。
    """
    for key in ["date", "publish_date", "published_at", "pub_date", "time"]:
        if key in news_item:
            dt = _parse_datetime_safe(news_item.get(key))
            if dt is not None:
                return dt
    return None


def _normalize_plan_tasks(tasks: list[dict], output_language: str = "English") -> list[dict]:
    """
    适配新 schema：
    - task_id: str
    - dependency: List[str]
    - target_phase_ids: List[int]
    """
    normalized = []

    for i, raw_task in enumerate(tasks, start=1):
        task_id = str(raw_task.get("task_id") or f"task_{i}")
        dependency = [str(x) for x in _to_list(raw_task.get("dependency")) if x]
        target_phase_ids = [int(x) for x in _to_list(raw_task.get("target_phase_ids")) if x is not None]
        args = dict(raw_task.get("args", {}))
        args["output_language"] = output_language


        normalized.append({
            "task_id": task_id,
            "agent": raw_task.get("agent"),
            "action": raw_task.get("action", ""),
            "args": args,
            "dependency": dependency,
            "target_phase_ids": target_phase_ids,
        })

    return normalized


def _build_phase_lookup(blueprint: dict) -> dict[int, dict]:
    phase_lookup = {}
    for phase in blueprint.get("phases", []):
        if "phase_id" in phase:
            try:
                phase_lookup[int(phase["phase_id"])] = phase
            except Exception:
                continue
    return phase_lookup


def _slice_news_by_phase_ids(news_list: list[dict], blueprint: dict, target_phase_ids: list[int]) -> list[dict]:
    """
    根据 target_phase_ids 自动切出任务所需新闻片段。
    如果你的 state 里已经有 phase -> news 的预计算映射，可以优先直接读取。
    """
    if not news_list:
        return []

    # 没指定 phase，默认给全量
    if not target_phase_ids:
        return news_list

    phase_lookup = _build_phase_lookup(blueprint)
    valid_ranges = []

    for pid in target_phase_ids:
        phase = phase_lookup.get(pid)
        if not phase:
            continue

        start_dt = _parse_datetime_safe(phase.get("start_date"))
        end_dt = _parse_datetime_safe(phase.get("end_date"))
        if start_dt and end_dt:
            valid_ranges.append((start_dt, end_dt))

    # 如果 blueprint 不完整，保底返回全量
    if not valid_ranges:
        return news_list

    sliced = []
    for item in news_list:
        news_dt = _extract_news_datetime(item)
        if news_dt is None:
            continue

        for start_dt, end_dt in valid_ranges:
            if start_dt <= news_dt <= end_dt:
                sliced.append(item)
                break

    return sliced


def _build_execution_args(task: dict, state: dict, dep_results_snapshot: dict) -> dict:
    raw_args = dict(task.get("args", {}))
    blueprint = state.get("spatiotemporal_blueprint") or {}
    global_news_list = state.get("news_list", [])
    target_phase_ids = task.get("target_phase_ids", [])

    phase_lookup = _build_phase_lookup(blueprint)
    phase_contexts = [phase_lookup[pid] for pid in target_phase_ids if pid in phase_lookup]
    phase_news_list = _slice_news_by_phase_ids(global_news_list, blueprint, target_phase_ids)

    execution_args = {
        **raw_args,
        "task_id": task["task_id"],
        "action": task.get("action", ""),
        "target_phase_ids": target_phase_ids,
        "phase_contexts": phase_contexts,                  # 给 agent 用于理解阶段语义
        "global_news_list": global_news_list,              # 全局新闻（必要时仍可访问）
        "phase_news_list": phase_news_list,                # 当前 task 的 phase 切片新闻
        "spatiotemporal_blueprint": blueprint,
        "blueprint_overall_narrative": blueprint.get("overall_narrative", ""),
    }

    dep_results = [
        dep_results_snapshot[dep_id]
        for dep_id in task.get("dependency", [])
        if dep_id in dep_results_snapshot
    ]
    if dep_results:
        execution_args["input_data"] = dep_results[0] if len(dep_results) == 1 else dep_results

    return execution_args


def _validate_and_build_graph(tasks: list[dict]):
    task_map = {task["task_id"]: task for task in tasks}
    graph = defaultdict(list)
    remaining_deps = {task["task_id"]: 0 for task in tasks}

    for task in tasks:
        task_id = task["task_id"]
        for dep_id in task.get("dependency", []):
            if dep_id not in task_map:
                raise ValueError(f"任务 [{task_id}] 依赖了不存在的任务 [{dep_id}]")
            graph[dep_id].append(task_id)
            remaining_deps[task_id] += 1

    # Kahn 检测环
    indegree_copy = dict(remaining_deps)
    q = deque([tid for tid, deg in indegree_copy.items() if deg == 0])
    visited = 0

    while q:
        cur = q.popleft()
        visited += 1
        for nxt in graph[cur]:
            indegree_copy[nxt] -= 1
            if indegree_copy[nxt] == 0:
                q.append(nxt)

    if visited != len(tasks):
        raise ValueError("任务图存在环，无法执行。")

    return task_map, graph, remaining_deps


def _run_single_task(task: dict, state: dict, dep_results_snapshot: dict):
    task_id = task["task_id"]
    agent_name = task["agent"]

    print(f"\n执行任务 [{task_id}] | Agent={agent_name} | phases={task.get('target_phase_ids', [])}")

    if agent_name not in AGENT_MAPPING:
        return task_id, {"error": f"未定义的 Agent: {agent_name}"}

    execution_args = _build_execution_args(task, state, dep_results_snapshot)
    result = AGENT_MAPPING[agent_name](execution_args)
    return task_id, result


def analyzing_node(state: dict) -> dict:
    print("\n" + "=" * 50 + "\n🚀 进入分析执行阶段...")

    plan = state.get("plan", {})
    raw_tasks = plan.get("tasks", [])
    tasks = _normalize_plan_tasks(raw_tasks)

    if not tasks:
        return {
            "task_results": {},
            "current_phase": "integrating",
            "messages": [AIMessage(content="没有需要执行的分析任务，直接进入整合阶段。")]
        }

    task_map, graph, remaining_deps = _validate_and_build_graph(tasks)
    ready_queue = deque([tid for tid, deg in remaining_deps.items() if deg == 0])

    task_results = {}
    failed_or_skipped = set()

    max_workers = min(
        int(os.getenv("ANALYSIS_MAX_WORKERS", "4")),
        max(1, len(tasks))
    )

    def submit_task(executor, task_id, running_futures):
        task = task_map[task_id]

        dep_snapshot = {
            dep_id: task_results[dep_id]
            for dep_id in task.get("dependency", [])
            if dep_id in task_results
        }

        future = executor.submit(_run_single_task, task, state, dep_snapshot)
        running_futures[future] = task_id
        print(f"   🚚 已提交任务 [{task_id}]")

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="analysis") as executor:
        running_futures = {}

        while ready_queue and len(running_futures) < max_workers:
            submit_task(executor, ready_queue.popleft(), running_futures)

        while running_futures:
            done_set, _ = wait(running_futures.keys(), return_when=FIRST_COMPLETED)

            for future in done_set:
                task_id = running_futures.pop(future)

                try:
                    finished_task_id, result = future.result()
                    task_results[finished_task_id] = result

                    if isinstance(result, dict) and result.get("error"):
                        failed_or_skipped.add(finished_task_id)
                        print(f"   ❌ 任务 [{finished_task_id}] 返回错误结果")
                    else:
                        print(f"   ✅ 任务 [{finished_task_id}] 完成")

                except Exception as e:
                    err_msg = f"执行出错: {str(e)}"
                    traceback.print_exc()
                    task_results[task_id] = {"error": err_msg}
                    failed_or_skipped.add(task_id)
                    finished_task_id = task_id
                    print(f"   ❌ 任务 [{task_id}] 执行异常: {err_msg}")

                # 释放后继任务
                completion_queue = deque([finished_task_id])

                while completion_queue:
                    completed_id = completion_queue.popleft()

                    for child_id in graph[completed_id]:
                        remaining_deps[child_id] -= 1
                        if remaining_deps[child_id] != 0:
                            continue

                        child_task = task_map[child_id]
                        child_deps = child_task.get("dependency", [])
                        failed_deps = [dep for dep in child_deps if dep in failed_or_skipped]

                        if failed_deps:
                            msg = f"跳过执行，失败依赖: {failed_deps}"
                            task_results[child_id] = {"error": msg}
                            failed_or_skipped.add(child_id)
                            completion_queue.append(child_id)
                            print(f"   ⏭️ 任务 [{child_id}] 被跳过：{msg}")
                        else:
                            ready_queue.append(child_id)

            while ready_queue and len(running_futures) < max_workers:
                submit_task(executor, ready_queue.popleft(), running_futures)

    missing = [task["task_id"] for task in tasks if task["task_id"] not in task_results]
    if missing:
        raise RuntimeError(f"存在未完成任务: {missing}")

    return {
        "task_results": task_results,
        "current_phase": "integrating",
        "messages": [AIMessage(content="所有子分析任务已完成，准备进入整合阶段。")]
    }

def integrating_node(state: ResearchState) -> dict:
    print("\n" + "=" * 50 + "\n📝 进入整合报告阶段...")
    # 初始化 Parser，绑定你的 Pydantic 模型
    integrating_parser = JsonOutputParser(pydantic_object=FinalReport)

    # 创建 PromptTemplate，注入 format_instructions
    integrating_prompt = PromptTemplate(
        template=INTEGRATING_PROMPT,
        input_variables=["intent", "context", "output_language"],
        partial_variables={"format_instructions": integrating_parser.get_format_instructions()}
    )

    # 组装 Chain (注意：这里的 model 替换为你实际使用的 llm 实例，比如 llm_qw_quick)
    integrating_chain = integrating_prompt | model | integrating_parser

    intent = state.get("intent", {})
    raw_results = state.get("task_results", {})
    evidence_pool = state.get("news_list", {})
    blueprint = state.get("spatiotemporal_blueprint", {})  # 【极度重要】：获取 Anchor 的蓝图
    profile_data = state.get("analysis_results", {}).get("data_profile", {})
    output_language = state.get("output_language")
    # 1. 组装上下文区块
    context_blocks = []
    for task_id, res in raw_results.items():
        agent_name = res.get("agent_name", "Unknown-Agent")
        summary_claims = res.get("summary", [])
        block = f"### Task ID: {task_id} (Agent: {agent_name})\n"
        if isinstance(summary_claims, list):
            for i, claim in enumerate(summary_claims):
                statement = claim.get("statement") if isinstance(claim, dict) else getattr(claim, "statement",
                                                                                           str(claim))
                src_ids = claim.get("source_ids", []) if isinstance(claim, dict) else getattr(claim, "source_ids", [])
                block += f"- 论点 {i + 1}: {statement} [证据源: {', '.join(src_ids)}]\n"
        else:
            block += f"- 分析结论: {summary_claims}\n"
        context_blocks.append(block)

    formatted_context = "\n\n".join(context_blocks)

    print("   🧠 正在调用 LLM 进行带有证据链的汇总写作 (JsonOutputParser)...")

    # 2. 执行 LLM Chain
    try:
        # parser 会自动将大模型的 JSON 字符串输出解析为字典，并验证 Pydantic 规则
        final_report = integrating_chain.invoke({
            "intent": json.dumps(intent, ensure_ascii=False),
            "context": formatted_context,
            "output_language" : output_language
        })
        print(f"   ✅ 报告生成成功: {final_report.get('report_title')}")


    except Exception as e:
        print(f"   ❌ 报告生成失败: {e}")
        # 失败时的兜底逻辑，返回符合格式的默认字典
        final_report = {
            "report_title": "分析报告生成失败",
            "executive_summary": str(e),
            "executive_source_ids": [],
            "sections": [],
            "conclusion": "分析中止"
        }

    integrated_payload = {
        "report": final_report,
        "tasks": raw_results,
        "evidence_pool": evidence_pool,
        "spatiotemporal_blueprint": blueprint,
        "profile_data":profile_data
    }
    
    # 【新增】将本次分析结果存入历史
    task_history = state.get("task_history", [])
    new_task_entry = {
        "task_id": f"task_{len(task_history) + 1}",
        "query": state.get("research_topic"),
        "results": integrated_payload,
        "is_sandbox": state.get("intent", {}).get("is_sandbox_request", False)
    }
    task_history.append(new_task_entry)


    return {
        "final_report": final_report,
        "analysis_results": integrated_payload,
        "current_phase": "ready",
        "messages": [{"role": "ai", "content": f"Report '{final_report.get('report_title')}' generated successfully."}],
        "task_history": task_history
    }

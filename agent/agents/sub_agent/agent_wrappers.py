from agent.agents.sub_agent.deep_dive_agent import deep_dive_agent
from agent.agents.sub_agent.global_monitor_agent import global_monitor_agent
from agent.agents.sub_agent.relation_miner_agent import relation_miner_agent
from agent.tools.news_manager import get_news_by_id


def global_monitor_agent_wrapper(args: dict) -> dict:
    query = f"query: {args.get('query')} time_range: {args.get('time_range')}"
    blueprint_context = args.get("blueprint_overall_narrative", "无特定演化阶段参考")
    focus_regions = args.get("focus_regions", [])
    output_language = args.get("output_language", "English")
    target_phase_name = args.get("target_phase_name", "Global")

    if focus_regions:
        query += f" | 重点关注区域(必须精准匹配坐标): {', '.join(focus_regions)}"

    news_list = args.get("phase_news_list", [])

    print(f"   🚀 [Exec] Global_Monitor 启动 (阶段: {target_phase_name})...")
    result = global_monitor_agent(news_list, query, blueprint_context, output_language)

    return {
        "agent_name": "Global_Monitor_Agent",
        "target_phase_name": target_phase_name,
        # 🌟 核心修正：精准透传双轨数据
        "factual_grounding": result.get("factual_grounding", []),
        "strategic_insights": result.get("strategic_insights", {}),
        "visualization_data": result.get("visualization_data", {})
    }


def deep_dive_agent_wrapper(args: dict) -> dict:
    entity = args.get("target_entity")
    query = f"query: {args.get('query')} time_range: {args.get('time_range')}"
    blueprint_context = args.get("blueprint_overall_narrative", "无特定演化阶段参考")
    focus_regions = args.get("focus_regions", [])
    output_language = args.get("output_language", "English")
    target_phase_name = args.get("target_phase_name", "Global")

    if focus_regions:
        query += f" | 重点关注区域(提取坐标时请优先考虑): {', '.join(focus_regions)}"

    news_list = args.get("phase_news_list", [])
    print(f"   🚀 [Exec] Deep_Dive 启动: 深挖 '{entity}' (阶段: {target_phase_name})...")

    result = deep_dive_agent(entity, query, news_list, blueprint_context, output_language)

    return {
        "agent_name": "Deep_Dive_Agent",
        "target_phase_name": target_phase_name,
        # 🌟 核心修正：精准透传双轨数据
        "factual_grounding": result.get("factual_grounding", []),
        "strategic_insights": result.get("strategic_insights", {}),
        "visualization_data": result.get("visualization_data", {})
    }


def relation_miner_agent_wrapper(args: dict) -> dict:
    entities = args.get("focus_entities", [])
    news_list = args.get("phase_news_list", [])
    output_language = args.get("output_language", "English")
    blueprint_context = args.get("blueprint_overall_narrative", "无特定宏观演化阶段参考")
    target_phase_name = args.get("target_phase_name", "Global")

    entities_str = ", ".join(entities) if entities else "无指定实体"
    print(f"   🚀 [Exec] Relation_Miner 启动: 挖掘 {entities_str} 博弈关系 (阶段: {target_phase_name})...")

    result = relation_miner_agent(entities, news_list, blueprint_context, output_language)

    return {
        "agent_name": "Relation_Miner_Agent",
        "target_phase_name": target_phase_name,
        # 🌟 核心修正：精准透传双轨数据
        "factual_grounding": result.get("factual_grounding", []),
        "strategic_insights": result.get("strategic_insights", {}),
        "visualization_data": result.get("visualization_data", {})
    }


def search_agent_wrapper(args: dict) -> dict:
    keywords = args.get("keywords")
    target_phase_name = args.get("target_phase_name", "Global")
    news_list = get_news_by_id(keywords)

    return {
        "agent_name": "Search_Agent",
        "target_phase_name": target_phase_name,
        # 搜索探员没有复杂的双轨推演，直接塞空数组/字典占位，保证下游不报错
        "factual_grounding": [],
        "strategic_insights": {},
        "visualization_data": None,
        "news_list": news_list,
    }


AGENT_MAPPING = {
    "Global_Monitor_Agent": global_monitor_agent_wrapper,
    "Deep_Dive_Agent": deep_dive_agent_wrapper,
    "Relation_Miner_Agent": relation_miner_agent_wrapper,
    "Search_Agent": search_agent_wrapper
}
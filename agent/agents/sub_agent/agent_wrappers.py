from agent.agents.sub_agent.deep_dive_agent import deep_dive_agent
from agent.agents.sub_agent.global_monitor_agent import global_monitor_agent
from agent.agents.sub_agent.relation_miner_agent import relation_miner_agent
from agent.tools.news_manager import get_news_by_id


def global_monitor_agent_wrapper(args: dict) -> dict:
    # 提取基础参数
    query = f"query: {args.get('query')} time_range: {args.get('time_range')}"

    # 【新增】：获取 Planner 传过来的蓝图上下文和重点区域列表
    blueprint_context = args.get("blueprint_overall_narrative", "无特定演化阶段参考")
    focus_regions = args.get("focus_regions", [])
    output_language = args.get("output_language")

    if focus_regions:
        query += f" | 重点关注区域(必须精准匹配坐标): {', '.join(focus_regions)}"

    news_list = args.get("global_news_list", [])

    # 将 blueprint_context 一并传给底层 Agent
    result = global_monitor_agent(news_list, query, blueprint_context, output_language)

    return {
        "agent_name": "Global_Monitor_Agent",
        "summary": result["final_answer"],
        "visualization_data": result["visualization_data"],
        "structured_insight": result["structured_insight"]
    }

def deep_dive_agent_wrapper(args: dict) -> dict:
    entity = args.get("target_entity")
    query = f"query: {args.get('query')} time_range: {args.get('time_range')}"

    # 【新增】：获取 Planner 传过来的蓝图上下文和重点区域列表
    blueprint_context = args.get("blueprint_overall_narrative", "无特定演化阶段参考")
    focus_regions = args.get("focus_regions", [])
    output_language = args.get("output_language", "English")
    # 强制空间锚定
    if focus_regions:
        query += f" | 重点关注区域(提取坐标时请优先考虑): {', '.join(focus_regions)}"

    news_list = args.get("global_news_list", [])
    print(f"   🚀 [Exec] Deep_Dive 启动: 深挖 '{entity}' 的行为画像...")

    # 【新增】：传入 blueprint_context
    result = deep_dive_agent(entity, query, news_list, blueprint_context, output_language)

    return {
        "agent_name": "Deep_Dive_Agent",
        "summary": result["final_answer"],
        "visualization_data": result["visualization_data"],
        "structured_insight": result["structured_insight"]
    }


def relation_miner_agent_wrapper(args: dict) -> dict:
    entities = args.get("focus_entities", [])
    news_list = args.get("global_news_list", [])
    output_language = args.get("output_language", "English")

    # 【新增】：获取 Planner 传过来的蓝图上下文
    blueprint_context = args.get("blueprint_overall_narrative", "无特定宏观演化阶段参考")

    entities_str = ", ".join(entities) if entities else "无指定实体"
    print(f"   🚀 [Exec] Relation_Miner 启动: 挖掘 {entities_str} 之间的博弈关系...")

    # 【新增】：传入 blueprint_context
    result = relation_miner_agent(entities, news_list, blueprint_context, output_language)

    return {
        "agent_name": "Relation_Miner_Agent",
        "summary": result["final_answer"],
        "visualization_data": result["visualization_data"],
        "structured_insight": result["structured_insight"]
    }

def search_agent_wrapper(args: dict) -> dict:
    keywords = args.get("keywords")
    news_list = get_news_by_id(keywords) # 假设底层方法已导入
    return {
        "agent_name": "Search_Agent",
        "summary": f"检索到关于 {keywords} 的 10 条基础简讯。",
        "visualization_data": None,
        "news_list": news_list
    }

# 统一定义路由映射，供 Node 使用
AGENT_MAPPING = {
    "Global_Monitor_Agent": global_monitor_agent_wrapper,
    "Deep_Dive_Agent": deep_dive_agent_wrapper,
    "Relation_Miner_Agent": relation_miner_agent_wrapper,
    "Search_Agent": search_agent_wrapper
}
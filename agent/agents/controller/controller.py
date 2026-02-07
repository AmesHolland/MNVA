
import json
from datetime import datetime
from typing import TypedDict, Annotated, Optional

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START
from langgraph.graph import add_messages, StateGraph
from pydantic import BaseModel

from agent.config.llm_config import llm_qw_quick
from agent.tools.base import safe_parse_json

# ==================== 对话模型定义 ====================
model = llm_qw_quick
# ==================== 数据模型定义 ====================

class SearchResult(BaseModel):
    """搜索结果"""
    title: str
    source: str
    url: str
    snippet: str | None
    content: str
    # relevance_score: float = Field(ge=0.0, le=1.0) | None
    publish_date: str


class ResearchFinding(BaseModel):
    """研究发现"""
    topic: str
    key_points: list[str]
    evidence: list[str]
    # confidence: float = Field(ge=0.0, le=1.0)
    sources: list[str]


class ResearchOutline(BaseModel):
    """研究大纲"""
    title: str
    abstract: str
    sections: list[str]
    key_questions: list[str]
    methodology: str


class Citation(BaseModel):
    """引用"""
    id: str
    authors: list[str]
    title: str
    source: str
    date: str
    url: Optional[str] = None


class ResearchReport(BaseModel):
    """研究报告"""
    title: str
    executive_summary: str
    introduction: str
    methodology: str
    findings: list[str]
    analysis: str
    conclusions: list[str]
    recommendations: list[str]
    citations: list[Citation]
    generated_at: str


# ==================== 状态定义 ====================

class ResearchState(TypedDict):
    """研究助手状态"""
    # 消息历史
    messages: Annotated[list, add_messages]

    # 研究主题
    research_topic: str
    research_questions: list[str]

    # 研究进展
    intent: dict
    plan: dict
    task_results: dict

    # 报告
    draft_sections: dict
    final_report: str
    citations: list[dict]

    # 状态追踪
    """
    current_phase 共分为5个阶段
    ready planning confirming analyzing integrating 
    ready 状态0 开启的新对话 / 先前的任务完成
    planning 状态1 用户input之后 进行意图识别、路径规划 对input之前的状态进行识别 0 就正常新规划 2就读取当前计划并进行重新设计
    confirming 状态2 等待用户确认计划是否可行 是 -> 状态3 ；否 -> 状态1
    analyzing 状态3 基于plan 进行多个子问题的分析 然后将结果暂存
    integrating 状态4 将之前分的多个子任务的结果进行整合 形成一个规范化的报告
    """
    current_phase: str
    iteration_count: int
    # quality_score: float

    # 数据缓存
    research_list: list[ResearchReport]

def create_visual_analytics_assistant():
    """创建研究助手系统"""

    # 意图识别
    def intent_node(state: ResearchState) -> dict:
        """识别用户意图 + 规划研究方向和大纲"""
        print("\n" + "=" * 50)
        print(" 用户意图识别阶段...")

        topic = state["research_topic"]

        # 定义Prompt 明确要求JSON格式
        intent_prompt = f"""
        用户Query：{topic}
        
        # Role
        你是一位精通全球地缘政治、海洋安全与海洋资源的资深情报分析专家。你的任务是解析用户关于海洋新闻的查询意图。
        
        # Task Description
        分析用户输入的 Query，提取其核心意图、时空约束及分析范式，并输出为规范的 JSON 格式。
        
        # Marine Domain Ontology (核心关注点)
        - 地缘政治 (Geopolitics): 主权争议、外交声明、国际法(UNCLOS)
        - 安全态势 (Security): 军事演习、非法捕捞(IUU)、海上对峙、航行自由
        - 资源开发 (Resources): 油气开采、深海采矿、渔业资源、BBNJ
        - 合作与搜救 (Cooperation): 海上联合搜救、生态保护、人道主义援助
        - 海洋科技发展 (Technology) : 海底电缆、深海采矿设备、环境监测设备、科考活动
        
        # Output Schema (JSON)
        {{
          "primary_intent": "事件追溯 | 区域对比 | 热点发现 | 实体跟踪 | 综合态势分析",
          "spatial_scope": ["具体海域或国家列表，若无则为空"],
          "entity":["涉及到的国家、公司或船只等实体列表，若无则为空"],
          "temporal_scale": {{
            "start": "YYYY-MM-DD",
            "end": "YYYY-MM-DD",
            "type": "point | range | evolution"
          }},
          "analysis_paradigm": {{
            "type": "Trend(趋势) | Correlation(关联) | Sentiment(情绪) | Contrast(对比)",
            "description": "简述分析逻辑"
          }},
          # "visual_suggestion": "Map | Time-Series | Sankey | Relationship-Graph | Rank",
          # "uncertainty_level": "low | medium | high (用户指令的模糊程度)"
        }}
        
        # Few-Shot Examples
        
        ### Example 1
        User: "对比过去三年中菲在黄岩岛附近的对峙频率和双方媒体的调门差异。"
        Output:
        {{
          "primary_intent": "区域对比",
          "spatial_scope": ["黄岩岛", "南海"],
          "entity":["中国", "菲律宾"]
          "temporal_scale": {{"start": "2023-01-01", "end": "2026-02-06", "type": "evolution"}},
          "analysis_paradigm": {{
            "type": "Contrast",
            "description": "对比中菲双方在同一地理坐标下的行动强度与舆论立场"
          }},
          "visual_suggestion": "Time-Series (对峙频率) + Sentiment-Heatmap (媒体调门)",
          "uncertainty_level": "low"
        }}
        
        ### Example 2
        User: "最近南太平洋有什么值得关注的新趋势吗？"
        Output:
        {{
          "primary_intent": "热点发现",
          "spatial_scope": ["南太平洋"],
          "temporal_scale": {{"start": "recently", "end": "now", "type": "range"}},
          "analysis_paradigm": {{
            "type": "Trend",
            "description": "多维扫描南太平洋的新闻，识别突发或持续升温的主题"
          }},
          "visual_suggestion": "Map (热点分布) + WordCloud (主题词)",
          "uncertainty_level": "medium"
        }}
        
        # Constraints
        1. 必须严格遵守 JSON 格式输出。
        2. 如果用户提到的地理概念模糊（如“周边海域”），请根据海洋常识自动补全可能的关联区域。
        3. 识别出潜在的“冲突”意图时，需特别标注需要进行对比分析。
        """
        response = model.invoke([HumanMessage(content=intent_prompt)])

        intent = safe_parse_json(response.content)

        print(f" primary intent: {intent["primary_intent"]}")
        print(f" spatial_scope: {intent["spatial_scope"]}")


        return {
            "intent" : intent,
            "research_topic": topic,
            "current_phase" : "planning",
             "messages": [AIMessage(content=f"用户意图已经解读完成：{topic}")]
        }

    # 路径规划
    def planning_node(state:ResearchState) -> dict:
        """规划研究方向和大纲"""
        print("\n" + "=" * 50)
        print("📋 研究规划阶段...")

        intent = state["intent"]

        planning_prompt = f"""
        
        用户intent {intent}
                
        # Role
        你是一位精通全球地缘政治、海洋安全与海洋资源的资深情报分析及复杂任务规划专家，专门负责编排海洋新闻分析工作流。你将基于对用户意图识别的结果，调用可用的子智能体（Sub-Agents）来构建执行路径。
        
        # Available Sub-Agents (工具箱)
        1. **Search-Agent**: 负责获取原始新闻列表。参数: keywords, time_range, count_limit(K值).
        2. **Time-Evolution-Agent**: 负责时序重排与阶段性总结。输入: 新闻列表.
        3. **Topic-Analysis-1-Agent (Hotspot Discovery)**: 识别未知热点与关键词。输入: 新闻列表.
        4. **Topic-Analysis-2-Agent (Hotspot Tracking)**: 针对已知热点进行深度挖掘。输入: 关键词 + 新闻列表.
        5. **Entity-Tracking-Agent**: 追踪特定对象（如船只、国家、组织）的动向演变。参数: entity_name, time_range.
        
        # Input Specification
        你将收到来自 Intent_Analyzer 的 JSON 输出，包含 primary_intent, spatial_scope, temporal_scale 等。
        
        # Output Format (JSON)
        输出必须是一个逻辑严密的 DAG (有向无环图) 列表：
        {{
          "total_plan_logic": "简述整体执行思路",
          "tasks": [
            {{
              "task_id": 1,
              "agent": "Agent名称",
              "action": "具体操作描述",
              "args": {{ "key-params": "values" }},
              "dependency": null,
            }},
            ...
          ]
        }}
        
        # Task Decomposition Strategies Examples
        - **若意图为[热点发现]**: 先执行 Search-Agent 获取大范围数据 -> 再执行 Topic-Analysis-1 提取热点。
        - **若意图为[区域对比]**: 需拆分为并行的两个搜索任务（如区域A vs 区域B）-> 分别进行 Topic-Analysis -> 最后汇总。
        - **若意图为[实体跟踪]**: 直接调用 Entity-Tracking-Agent -> 衔接 Time-Evolution-Agent 做时序总结。
        
        # Example
        Input: {{"primary_intent": "实体跟踪", "spatial_scope": ["黄岩岛"], "entity": "中国海警", "temporal_scale": {{"type": "evolution"}}
        Output:
        {{
          "total_plan_logic": "首先检索特定实体在目标海域的新闻，随后按时间线梳理其行动演化趋势。",
          "tasks": [
            {{
              "task_id": 1,
              "agent": "Entity-Tracking-Agent",
              "action": "追踪‘中国海警’在黄岩岛附近的活动新闻",
              "args": {{"entity_name": "中国海警", "location": "黄岩岛", "limit": 20}},
              "dependency": null,
            }},
            {{
              "task_id": 2,
              "agent": "Time-Evolution-Agent",
              "action": "对获取的新闻进行时序梳理和事态演变阶段划分",
              "args": {{"input_from_task": 1}},
              "dependency": 1,
            }}
          ]
        }}
        """

        response = model.invoke([HumanMessage(content=planning_prompt)])

        plan = safe_parse_json(response.content)

        print(f" total_plan_logic: {plan["total_plan_logic"]}")
        for task in plan["tasks"]:
            print(task)

        return {
            "plan": plan,
            #"research_topic": topic,
            "current_phase": "planning",
            "messages": [AIMessage(content=f"任务编排已经完成：{plan["total_plan_logic"]}")]
        }

    # 具体分析工作
    def analyzing_node(state: ResearchState) -> dict:
        """基于规划执行具体分析任务（修复多重依赖支持）"""
        print("\n" + "=" * 50)
        print("🚀 进入分析执行阶段...")

        plan = state["plan"]
        tasks = plan.get("tasks", [])

        # 初始化 task_results
        task_results = state.get("task_results", {}) or {}

        agent_mapping = {
            "Search-Agent": search_agent_tool,
            "Time-Evolution-Agent": time_evolution_tool,
            "Topic-Analysis-1-Agent": topic_analysis_discovery_tool,
            "Topic-Analysis-2-Agent": topic_analysis_tracking_tool,
            "Entity-Tracking-Agent": entity_tracking_tool
        }

        for task in tasks:
            task_id = task["task_id"]
            agent_name = task["agent"]
            raw_args = task["args"]
            dependency_ids = task.get("dependency")  # 获取依赖，可能是 int，也可能是 list

            print(f"\n执行任务 [{task_id}]: {agent_name} - {task['action']}")

            # 创建参数副本
            execution_args = raw_args.copy()

            # === 核心修复：处理多重依赖 ===
            input_data = None

            if dependency_ids is not None:
                # 情况 A: 依赖是列表 (例如 [1, 2]) -> 获取多个结果
                if isinstance(dependency_ids, list):
                    print(f"   └── 🔗 检测到多重依赖: {dependency_ids}")
                    # 将所有依赖的结果打包成一个列表
                    input_data = [task_results.get(dep_id) for dep_id in dependency_ids if
                                  task_results.get(dep_id) is not None]

                    # 如果是多重依赖，通常我们也希望保留 ID 信息，可以用字典形式 (可选)
                    # input_data = {dep_id: task_results.get(dep_id) for dep_id in dependency_ids}

                # 情况 B: 依赖是单个 ID (例如 1) -> 获取单个结果
                else:
                    input_data = task_results.get(dependency_ids)

            # 情况 C: 从 args 的 input_from_task 中获取 (作为备选)
            if input_data is None:
                for key, value in raw_args.items():
                    if key == "input_from_task":
                        # 同样要处理 value 是 list 的情况
                        if isinstance(value, list):
                            input_data = [task_results.get(v) for v in value]
                        else:
                            input_data = task_results.get(value)
                        break

            # === 注入数据 ===
            if input_data:
                data_type = "List" if isinstance(input_data, list) else type(input_data).__name__
                print(f"   └── 📦 成功注入数据 (Type: {data_type})")
                execution_args["input_data"] = input_data
            elif dependency_ids is not None:
                print(f"   ⚠️ 警告: 依赖任务的结果为空")

            # === 执行工具 ===
            if agent_name in agent_mapping:
                tool_func = agent_mapping[agent_name]
                try:
                    result = tool_func(execution_args)
                    task_results[task_id] = result

                    # 打印摘要
                    res_str = str(result)
                    preview = (res_str[:50] + "...") if len(res_str) > 50 else res_str
                    print(f"   ✅ 完成. 结果: {preview}")

                except Exception as e:
                    error_msg = f"执行出错: {str(e)}"
                    print(f"   ❌ {error_msg}")
                    task_results[task_id] = {"error": error_msg}
            else:
                print(f"   ⚠️ 未定义的 Agent: {agent_name}")
                task_results[task_id] = "Agent Not Found"

        return {
            "task_results": task_results,
            "current_phase": "analyzing",
            "messages": [AIMessage(content="所有子分析任务已完成，准备进入整合阶段。")]
        }

    def search_agent_tool(args: dict):
        """
        Search-Agent 逻辑
        实现：从 Weaviate 或 API 获取新闻
        """
        keywords = args.get("keywords")
        limit = args.get("count_limit", 10)
        print(f"   [Tool] 正在检索关键词: {keywords}，数量: {limit}")

        # 模拟检索过程
        # results = weaviate_client.query.get("MarineNews").with_near_text({"concepts": [keywords]}).with_limit(limit).do()
        mock_news = [
            {"title": f"新闻_{i}", "content": "内容...", "date": "2026-02-01", "location": "南海"}
            for i in range(limit)
        ]
        return mock_news

    def time_evolution_tool(args: dict):
        """
        Time-Evolution-Agent 逻辑
        实现：LLM 对新闻进行时序切片和总结
        """
        news_list = args.get("input_data", [])
        print(f"   [Tool] 正在对 {len(news_list)} 条新闻进行时序分析...")

        # 调用 LLM 进行总结
        # prompt = f"请将以下新闻按时间线划分阶段：{news_list}"
        # summary = model.invoke(prompt)

        analysis_result = {
            "timeline": ["阶段1: 冲突爆发", "阶段2: 舆论升级"],
            "summary": "事态呈现螺旋式上升趋势..."
        }
        return analysis_result

    def entity_tracking_tool(args: dict):
        """
        Entity-Tracking-Agent 逻辑
        """
        entity = args.get("entity_name")
        print(f"   [Tool] 正在追踪实体: {entity} 的历史轨迹...")
        return f"关于 {entity} 的动向追踪结果数据"

    def topic_analysis_discovery_tool(args: dict):
        """
        Entity-Tracking-Agent 逻辑
        """
        entity = args.get("entity_name")
        print(f"   [Tool] 正在追踪实体: {entity} 的历史轨迹...")
        return f"关于 {entity} 的动向追踪结果数据"

    def topic_analysis_tracking_tool(args: dict):
        """
        Entity-Tracking-Agent 逻辑
        """
        entity = args.get("entity_name")
        print(f"   [Tool] 正在追踪实体: {entity} 的历史轨迹...")
        return f"关于 {entity} 的动向追踪结果数据"

    def integrating_node(state: ResearchState) -> dict:
        """
        整合节点：汇总子Agent的文本结果，生成叙事报告。
        可视化组件的渲染数据假设已在各子Agent中生成并存储，此处仅做ID引用。
        """
        print("\n" + "=" * 50)
        print("📝 进入整合报告阶段 (Integrating Node)...")

        intent = state.get("intent", {})
        # 过滤掉非文本的大型数据，只保留 task_id, agent_name 和 文本结论/summary
        # 这样可以减少 Token 消耗，避免把 huge raw data 塞给 LLM
        simplified_results = []

        raw_results = state.get("task_results", {})

        for task_id, res in raw_results.items():
            # 假设子Agent的结果结构是 {"summary": "...", "viz_data": "...", "raw_data": "..."}
            # 我们只提取 summary 部分给 Integrating Node
            summary_text = res.get("summary") if isinstance(res, dict) else str(res)
            agent_name = res.get("agent_name", "Unknown-Agent") if isinstance(res, dict) else "Task"

            simplified_results.append({
                "task_id": task_id,
                "agent": agent_name,
                "analysis_text": summary_text
            })

        # 1. 构造 Prompt
        prompt_content = f"""
        # Role
        你是一名资深的海洋情报主编。你的任务是根据多个子任务的分析片段，撰写一份“海洋态势深度分析报告”。

        # Context
        用户意图: {json.dumps(intent, ensure_ascii=False)}

        # Input Data (各子任务的分析结论)
        {json.dumps(simplified_results, ensure_ascii=False, indent=2)}

        # Output Format (JSON)
        请输出如下 JSON 格式，其中 `ref_task_ids` 字段用于关联已生成的图表：
        {{
          "report_title": "报告标题",
          "executive_summary": "摘要",
          "sections": [
            {{
              "subtitle": "小标题",
              "content": "详细分析...",
              "ref_task_ids": [1] 
            }}
          ],
          "conclusion": "结语"
        }}
        """

        # 2. 调用 LLM
        print("   正在调用 LLM 进行汇总写作...")
        response = model.invoke([HumanMessage(content=prompt_content)])

        # 3. 解析结果
        try:
            final_report = safe_parse_json(response.content)
            print(f"   ✅ 报告生成成功: {final_report.get('report_title')}")
        except Exception as e:
            print(f"   ❌ 报告解析失败: {e}")
            # Fallback 机制
            final_report = {
                "report_title": "分析报告生成失败",
                "executive_summary": "无法解析 LLM 输出。",
                "sections": [],
                "raw_output": response.content
            }

        # 4. 更新状态
        # 注意：这里我们把生成的 final_report 存入状态
        # 状态流转回 ready (0) 或 finish，等待用户下一次交互
        return {
            "final_report": final_report,
            "current_phase": "ready",
            "messages": [AIMessage(content=f"报告《{final_report.get('report_title')}》已生成。")]
        }


    graph = StateGraph(ResearchState)

    graph.add_node("intent", intent_node)
    graph.add_node("planning", planning_node)
    graph.add_node("analysis", analyzing_node)
    graph.add_node("integrate", integrating_node)

    graph.add_edge(START, "intent" )
    graph.add_edge("intent", "planning")
    graph.add_edge("planning", "analysis")
    graph.add_edge("analysis", "integrate")

    # # 条件路由：根据质量决定是否重新迭代
    # graph.add_conditional_edges(
    #     "quality_check",
    #     should_continue,
    #     {
    #         "continue": "analysis",  # 重新分析
    #         "complete": END
    #     }
    # )

    # 编译
    memory = MemorySaver()
    compiled_graph = graph.compile(checkpointer=memory)

    return compiled_graph


def run_research(topic: str):
    """运行研究任务"""
    print("\n" + "=" * 60)
    print("🔬 启动研究任务")
    print("=" * 60)
    print(f"研究主题: {topic}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 创建研究助手
    assistant = create_visual_analytics_assistant()

    # 初始状态
    initial_state = {
        "messages": [HumanMessage(content=f"请对以下主题进行深入研究：{topic}")],
        "research_topic": topic,
        "research_questions": [],
        "intent": [],
        "plan": {},
        "findings": [],
        "task_results": {},
        "final_report": "",
        "draft_sections": {},
        "current_phase":"",
        "iteration_count": 0,
        "research_list": []
    }

    # 运行研究流程
    config = {"configurable": {"thread_id": f"research_{datetime.now().strftime('%Y%m%d%H%M%S')}"}}
    result = assistant.invoke(initial_state, config)

    # 输出结果
    print("\n" + "=" * 60)
    print("📄 研究报告")
    print("=" * 60)
    print(result.get("final_report", "报告生成失败"))

    # print("\n" + "-" * 60)
    # print("📚 参考文献")
    # print("-" * 60)
    # for citation in result.get("citations", []):
    #     authors = ", ".join(citation.get("authors", ["Unknown"]))
    #     print(f"{citation['id']} {authors}. {citation['title']}. {citation['source']}, {citation['year']}.")
    #
    # print("\n" + "-" * 60)
    # print("📊 研究统计")
    # print("-" * 60)
    # print(f"  - 收集资料数: {len(result.get('search_results', []))}")
    # print(f"  - 分析来源数: {len(result.get('analyzed_sources', []))}")
    # print(f"  - 迭代次数: {result.get('iteration_count', 0)}")
    # print(f"  - 质量评分: {result.get('quality_score', 0):.1f}/10")
    # print(f"  - 报告字数: {len(result.get('final_report', ''))}")

    return result

if __name__ == '__main__':
    topic = "美国2025年关于深海采矿的一系列操作"
    run_research(topic)

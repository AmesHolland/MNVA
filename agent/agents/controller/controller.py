import json
import os
from datetime import datetime
from typing import List, Optional
from typing import TypedDict, Annotated, Dict, Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph import add_messages
from pydantic import BaseModel
from pydantic import Field

from agent.agents.sub_agent.deep_dive_agent import deep_dive_agent
from agent.agents.sub_agent.global_monitor_agent import global_monitor_agent
from agent.agents.sub_agent.relation_miner_agent import relation_miner_agent
from agent.config.llm_config import llm_qw_quick
from agent.tools.news_manager import get_news_by_id

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
    news_list: list[dict]

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

    query: str
    # 数据缓存
    research_list: list[ResearchReport]

    # 存储 Planner 生成的具体计划（例如需要调用哪些子Agent及参数）
    current_plan: Dict[str, Any]
    # 存储用户的审批意见：可以是 "approve"（同意），或者是一段修改意见的字符串
    user_feedback: str
    # 其他你原有的状态，如搜集到的数据、分析结果等
    analysis_results: Dict[str, Any]

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
    # def planning_node(state:ResearchState) -> dict:
    #     """规划研究方向和大纲"""
    #     print("\n" + "=" * 50)
    #     print("📋 研究规划阶段...")
    #
    #     intent = state["intent"]
    #
    #     planning_prompt = f"""
    #
    #     用户intent {intent}
    #
    #     # Role
    #     你是一位精通全球地缘政治、海洋安全与海洋资源的资深情报分析及复杂任务规划专家，专门负责编排海洋新闻分析工作流。你将基于对用户意图识别的结果，调用可用的子智能体（Sub-Agents）来构建执行路径。
    #
    #     # Available Sub-Agents (工具箱)
    #     1. **Search-Agent**: 负责获取原始新闻列表。参数: keywords, time_range, count_limit(K值).
    #     2. **Time-Evolution-Agent**: 负责时序重排与阶段性总结。输入: 新闻列表.
    #     3. **Topic-Analysis-1-Agent (Hotspot Discovery)**: 识别未知热点与关键词。输入: 新闻列表.
    #     4. **Topic-Analysis-2-Agent (Hotspot Tracking)**: 针对已知热点进行深度挖掘。输入: 关键词 + 新闻列表.
    #     5. **Entity-Tracking-Agent**: 追踪特定对象（如船只、国家、组织）的动向演变。参数: entity_name, time_range.
    #
    #     # Input Specification
    #     你将收到来自 Intent_Analyzer 的 JSON 输出，包含 primary_intent, spatial_scope, temporal_scale 等。
    #
    #     # Output Format (JSON)
    #     输出必须是一个逻辑严密的 DAG (有向无环图) 列表：
    #     {{
    #       "total_plan_logic": "简述整体执行思路",
    #       "tasks": [
    #         {{
    #           "task_id": 1,
    #           "agent": "Agent名称",
    #           "action": "具体操作描述",
    #           "args": {{ "key-params": "values" }},
    #           "dependency": null,
    #         }},
    #         ...
    #       ]
    #     }}
    #
    #     # Task Decomposition Strategies Examples
    #     - **若意图为[热点发现]**: 先执行 Search-Agent 获取大范围数据 -> 再执行 Topic-Analysis-1 提取热点。
    #     - **若意图为[区域对比]**: 需拆分为并行的两个搜索任务（如区域A vs 区域B）-> 分别进行 Topic-Analysis -> 最后汇总。
    #     - **若意图为[实体跟踪]**: 直接调用 Entity-Tracking-Agent -> 衔接 Time-Evolution-Agent 做时序总结。
    #
    #     # Example
    #     Input: {{"primary_intent": "实体跟踪", "spatial_scope": ["黄岩岛"], "entity": "中国海警", "temporal_scale": {{"type": "evolution"}}
    #     Output:
    #     {{
    #       "total_plan_logic": "首先检索特定实体在目标海域的新闻，随后按时间线梳理其行动演化趋势。",
    #       "tasks": [
    #         {{
    #           "task_id": 1,
    #           "agent": "Entity-Tracking-Agent",
    #           "action": "追踪‘中国海警’在黄岩岛附近的活动新闻",
    #           "args": {{"entity_name": "中国海警", "location": "黄岩岛", "limit": 20}},
    #           "dependency": null,
    #         }},
    #         {{
    #           "task_id": 2,
    #           "agent": "Time-Evolution-Agent",
    #           "action": "对获取的新闻进行时序梳理和事态演变阶段划分",
    #           "args": {{"input_from_task": 1}},
    #           "dependency": 1,
    #         }}
    #       ]
    #     }}
    #     """
    #
    #     response = model.invoke([HumanMessage(content=planning_prompt)])
    #
    #     plan = safe_parse_json(response.content)
    #
    #     print(f" total_plan_logic: {plan["total_plan_logic"]}")
    #     for task in plan["tasks"]:
    #         print(task)
    #
    #     return {
    #         "plan": plan,
    #         #"research_topic": topic,
    #         "current_phase": "planning",
    #         "messages": [AIMessage(content=f"任务编排已经完成：{plan["total_plan_logic"]}")]
    #     }

    def planning_node(state: ResearchState) -> dict:
        """
        【规划节点】
        根据意图识别结果，编排 Global_Monitor, Deep_Dive, Relation_Miner 的执行路径。
        """
        print("\n" + "=" * 50)
        print("📋 [Planner] 进入任务规划阶段 ")

        # 获取上一步意图分析的结果
        intent_data = state.get("intent", {})
        user_query = intent_data.get("original_query", "")  # 假设 intent 中保留了原始查询

        query = state.get("query")
        feedback = state.get("user_feedback", "")
        plan = state.get("plan")

        review = ""
        if feedback != "approve" and feedback != "":
            review = f"这是之前生成的计划 {plan} 用户提出了修改意见 {feedback}"

        # 构造规划器的 Prompt
        planning_prompt = f"""
        # Context
        
        用户正在使用“海洋新闻态势感知系统”。
        用户的原始需求: "{user_query}"
        意图识别结果: {json.dumps(intent_data, ensure_ascii=False)}
        
        {review}
        
        
        # Role
        你是一名精通海洋地缘政治的情报指挥官。你的任务是将用户的需求拆解为具体的分析任务，并指派给最合适的下属智能体（Sub-Agents）。

        # Available Sub-Agents (你的工具箱)

        1. **Search_Agent (基础检索)**
           - **适用场景**: 为其他高级Agent的前置新闻搜索，适用于用户仅仅想知道近期新闻内容，
           - **能力**: 通过语义相似度和关键词匹配等方式检索和keyword最相关的新闻
           - **参数**: `keywords` (str).
           
        2. **Global_Monitor_Agent (宏观态势感知)**
           - **适用场景**: 用户询问模糊、宽泛、探索性问题（如“最近南海发生了什么？”，“热点分布”）。
           - **能力**: 聚类热点话题，生成热力图，总结宏观趋势。
           - **参数**: 
             - `query`: (str) 搜索关键词或描述。
             - `time_range`: (str) 例如 "Last 30 days"。

        3. **Deep_Dive_Agent (微观情报研判)**
           - **适用场景**: 用户关注**特定实体**（船只、国家、组织）或**特定事件**。
           - **能力**: 绘制时空轨迹（地图）、行为序列（甘特图）、多维烈度画像（雷达图）。
           - **参数**: 
             - `target_entity`: (str) **必须**是具体的实体名称（如"菲律宾海警"、"里根号"、"仁爱礁冲突"）。
             - `time_range`: (str) 例如 "2025-11-01 - 2025-12-15"。

        4. **Relation_Miner_Agent (隐性关系挖掘)**
           - **适用场景**: 用户询问多方博弈、因果关系、或通过“A对B的影响”进行分析。
           - **能力**: 构建冲突/合作网络图，挖掘事件传导链。
           - **参数**: 
             - `focus_entities`: (List[str]) 涉及的多个实体名称（如 ["中国", "菲律宾", "美国"]）。

        # Planning Logic (任务编排策略)
        - **单点突破**: 如果用户明确问“山东舰的动向”，
          - Task 1: Search_Agent (查找和山东舰相关的新闻)
          - Task 2:`Deep_Dive_Agent` 。
        - **多点关联**: 如果用户问“中菲最近的摩擦”，
          - Task 1: Search_Agent (查找和中菲相关的新闻)
          - Task 2: Relation_Miner_Agent (进行中菲之间隐形关系的探索)
        - **先面后点**: 如果用户问“南海局势及美军动态”，建议并行：
          - Task 1: Search_Agent （查找和南海、美军相关的新闻）
          - Task 2: `Global_Monitor_Agent` (看南海整体)。
          - Task 3: `Deep_Dive_Agent` (看美军具体)。
        - **信息基础**: 在进行一系列分析之前 需要先调用Search_Agent进行相关新闻的搜搜 
        - **避免冗余**: 不要对同一个实体同时调用 Monitor 和 Deep_Dive，除非用户明确要求。

        # Output Format (JSON Only)
        Strictly output a JSON object representing the DAG of tasks:
        {{
          "total_plan_logic": "简述你的调度思路 (1-2句话)",
          "tasks": [
            {{
              "task_id": 1,
              "agent": "AgentName",
              "action": "任务描述",
              "args": {{ "arg_name": "value" }},
              "dependency": null
            }},
            ...
          ]
        }}
        """

        # 调用 LLM 生成计划
        response = model.invoke([HumanMessage(content=planning_prompt)])

        # 解析 JSON
        try:
            # 这里建议使用更健壮的解析函数，防止 markdown 代码块干扰
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]

            plan = json.loads(content)

            print(f"   🧠 规划思路: {plan.get('total_plan_logic')}")
            for task in plan.get("tasks", []):
                print(f"   Correlation Task: [{task['task_id']}] {task['agent']} -> {task['args']}")

        except Exception as e:
            print(f"   ❌ 规划解析失败: {e}")
            # Fallback Plan (默认走宏观监测)
            plan = {
                "total_plan_logic": "解析失败，默认执行宏观态势感知。",
                "tasks": [{
                    "task_id": 1,
                    "agent": "Global_Monitor_Agent",
                    "args": {"query": user_query},
                    "action": "Fallback execution"
                }]
            }

        return {
            "plan": plan,
            "current_phase": "confirming",
            "messages": [AIMessage(content=f"已规划任务：{plan.get('total_plan_logic')}")]
        }

    def check_node(state: ResearchState):
        # 图被唤醒后，会执行这个节点。
        feedback = state.get('user_feedback')
        # 此时 state["user_feedback"] 已经被后端 API 注入了用户的真实意图
        print(f"--- 接收到用户反馈: {feedback} ---")
        # 这里可以做一些格式化或日志记录，不需要修改状态
        # if feedback == "approve":
        #     return { "current_phase": "analyzing"}
        # else:
        #     # 如果用户给出修改意见，则流回 planning 节点重新规划
        #     return { "current_phase": "planning"}
        {}

    # --- 路由逻辑 ---

    def route_after_check(state: ResearchState) -> str:
        feedback = state.get("user_feedback", "")
        if feedback == "approve":
            return "analysis"
        else:
            # 如果用户给出修改意见，则流回 planning 节点重新规划
            return "planning"

    # 具体分析工作
    def analyzing_node(state: ResearchState) -> dict:
        """基于规划执行具体分析任务（修复多重依赖支持）"""
        print("\n" + "=" * 50)
        print("🚀 进入分析执行阶段...")

        plan = state["plan"]
        tasks = plan.get("tasks", [])

        # 初始化 task_results
        task_results = state.get("task_results", {}) or {}

        # agent_mapping = {
        #     "Search-Agent": search_agent_tool,
        #     "Time-Evolution-Agent": time_evolution_tool,
        #     "Topic-Analysis-1-Agent": topic_analysis_discovery_tool,
        #     "Topic-Analysis-2-Agent": topic_analysis_tracking_tool,
        #     "Entity-Tracking-Agent": entity_tracking_tool
        # }

        agent_mapping = {
            "Global_Monitor_Agent": global_monitor_agent_wrapper,
            "Deep_Dive_Agent": deep_dive_agent_wrapper,
            "Relation_Miner_Agent": relation_miner_agent_wrapper,
            # 兼容旧名称（如果有）
            "Search_Agent": search_agent_wrapper
        }

        for task in tasks:
            task_id = task["task_id"]
            agent_name = task["agent"]
            raw_args = task["args"]
            dependency_ids = task.get("dependency")  # 获取依赖，可能是 int，也可能是 list
            news_list = []

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
                print(f"execution_args: {execution_args}")
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
            "current_phase": "integrating",
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
    #
    #     # 模拟检索过程
    #     # results = weaviate_client.query.get("MarineNews").with_near_text({"concepts": [keywords]}).with_limit(limit).do()
    #     mock_news = [
    #         {"title": f"新闻_{i}", "content": "内容...", "date": "2026-02-01", "location": "南海"}
    #         for i in range(limit)
    #     ]
    #     return mock_news
    #
    # def time_evolution_tool(args: dict):
    #     """
    #     Time-Evolution-Agent 逻辑
    #     实现：LLM 对新闻进行时序切片和总结
    #     """
    #     news_list = args.get("input_data", [])
    #     print(f"   [Tool] 正在对 {len(news_list)} 条新闻进行时序分析...")
    #
    #     # 调用 LLM 进行总结
    #     # prompt = f"请将以下新闻按时间线划分阶段：{news_list}"
    #     # summary = model.invoke(prompt)
    #
    #     analysis_result = {
    #         "timeline": ["阶段1: 冲突爆发", "阶段2: 舆论升级"],
    #         "summary": "事态呈现螺旋式上升趋势..."
    #     }
    #     return analysis_result
    #
    # def entity_tracking_tool(args: dict):
    #     """
    #     Entity-Tracking-Agent 逻辑
    #     """
    #     entity = args.get("entity_name")
    #     print(f"   [Tool] 正在追踪实体: {entity} 的历史轨迹...")
    #     return f"关于 {entity} 的动向追踪结果数据"
    #
    # def topic_analysis_discovery_tool(args: dict):
    #     """
    #     Entity-Tracking-Agent 逻辑
    #     """
    #     entity = args.get("entity_name")
    #     print(f"   [Tool] 正在追踪实体: {entity} 的历史轨迹...")
    #     return f"关于 {entity} 的动向追踪结果数据"
    #
    # def topic_analysis_tracking_tool(args: dict):
    #     """
    #     Entity-Tracking-Agent 逻辑
    #     """
    #     entity = args.get("entity_name")
    #     print(f"   [Tool] 正在追踪实体: {entity} 的历史轨迹...")
    #     return f"关于 {entity} 的动向追踪结果数据"

    def global_monitor_agent_wrapper(args: dict) -> dict:
        """
        对应 Global_Monitor_Agent
        输入: {"query": "...", "time_range": "..."}
        """
        query = f"query: {args.get("query")} time_range: {args.get('time_range')}"
        news_list = args.get("input_data")["news_list"]

        result = global_monitor_agent( news_list, query)
        # print(result)
        # return result
        # print("global_monitor_agent_wrapper 分析完毕")
        return {
            "agent_name": "Global_Monitor_Agent",
            "summary": result["final_answer"],
            "visualization_data": result["visualization_data"],
            "structured_insight": result["structured_insight"]
        }


    def deep_dive_agent_wrapper(args: dict) -> dict:
        """
        对应 Deep_Dive_Agent
        输入: {"target_entity": "..."}
        """
        entity = args.get("target_entity")
        query = f"query: {args.get("query")} time_range: {args.get('time_range')}"
        news_list = args.get("input_data")["news_list"]

        result = deep_dive_agent(entity, query, news_list)

        print(f"   🚀 [Exec] Deep_Dive 启动: 深挖 '{entity}' 的行为画像...")

        # 真实场景：调用 deep_dive_node 逻辑
        return {
            "agent_name": "Deep_Dive_Agent",
            "summary": result["final_answer"],
            "visualization_data": result["visualization_data"],
            "structured_insight": result["structured_insight"]
        }

    def relation_miner_agent_wrapper(args: dict) -> dict:
        """
        对应 Relation_Miner_Agent
        输入: {"focus_entities": ["A", "B"], "retrieved_docs": [...]}
        """
        # 1. 从 args 中提取核心参数，增加默认值保证鲁棒性
        entities = args.get("focus_entities", [])
        retrieved_docs = args.get("retrieved_docs", [])
        news_list = args.get("input_data")["news_list"]

        # 2. 打印标准化启动日志（对齐 Deep_Dive 风格）
        entities_str = ", ".join(entities) if entities else "无指定实体"
        print(f"   🚀 [Exec] Relation_Miner 启动: 挖掘 {entities_str} 之间的博弈关系...")

        # 3. 构造 state 并调用核心逻辑函数
        state = {
            "focus_entities": entities,
            "retrieved_docs": retrieved_docs
        }
        result = relation_miner_agent(entities, news_list)

        # 4. 按统一格式返回结果（对齐 Deep_Dive 的返回结构）
        return {
            "agent_name": "Relation_Miner_Agent",
            "summary": result["final_answer"],
            "visualization_data": result["visualization_data"],
            "structured_insight": result["structured_insight"]
        }

    def search_agent_wrapper(args: dict) -> dict:
        keywords = args.get("keywords")
        news_list = get_news_by_id(keywords)

        return {
            "agent_name": "Search_Agent",
            "summary": f"检索到关于 {keywords} 的 10 条基础简讯。",
            "visualization_data": None,
            "news_list": news_list
        }



    # ==========================================
    # 1. 定义强制输出的 Pydantic 模型 (核心溯源结构)
    # ==========================================
    class Claim(BaseModel):
        statement: str = Field(description="总结的论点、事实判断或事件描述。")
        is_direct_quote: bool = Field(description="如果是直接截取新闻原话为 True，自行归纳总结为 False。")
        source_ids: List[str] = Field(description="支撑该句话的具体新闻 DOC_ID 列表。必须从输入的上下文中提取，严禁捏造。")

    class ReportSection(BaseModel):
        subtitle: str = Field(description="该章节的小标题")
        content_claims: List[Claim] = Field(
            description="该章节的正文，必须拆解为多个逻辑连贯的论点/句子，每个句子必须附带 source_ids。")
        ref_task_ids: List[str] = Field(
            description="该章节分析所依赖的子任务 ID 列表 (例如 ['1', '2'])，用于前端关联图表。")

    class FinalReport(BaseModel):
        report_title: str = Field(description="海洋态势深度分析报告标题")
        executive_summary: str = Field(description="高度概括的执行摘要")
        executive_source_ids: List[str] = Field(description="支撑执行摘要的核心 DOC_ID 列表")
        sections: List[ReportSection] = Field(description="报告的主体章节")
        conclusion: str = Field(description="对未来趋势的最终战略研判与结语")

    # ==========================================
    # 2. 整合节点核心逻辑
    # ==========================================
    def integrating_node(state: dict) -> dict:  # 建议使用你定义的 ResearchState 类型
        """
        整合节点：汇总子Agent的文本和溯源结果，生成带有证据链的叙事报告。
        """
        print("\n" + "=" * 50)
        print("📝 进入整合报告阶段 (Integrating Node)...")

        intent = state.get("intent", {})
        raw_results = state.get("task_results", {})
        evidence_pool = state.get("evidence_pool", {})  # 获取全局新闻字典

        # 1. 构造高质量的 LLM 上下文 (Context)
        # 将子 Agent 的 Claims 转换为清晰的文本，暴露 DOC_ID 给 Integrator
        context_blocks = []
        for task_id, res in raw_results.items():
            agent_name = res.get("agent_name", "Unknown-Agent")
            # 提取子 Agent 的 summary (现在是一个 Claim 列表或结构化字典)
            summary_claims = res.get("summary", [])

            block = f"### Task ID: {task_id} (Agent: {agent_name})\n"
            if isinstance(summary_claims, list):
                for i, claim in enumerate(summary_claims):
                    # 兼容字典或 Pydantic 对象
                    statement = claim.get("statement") if isinstance(claim, dict) else getattr(claim, "statement",
                                                                                               str(claim))
                    src_ids = claim.get("source_ids", []) if isinstance(claim, dict) else getattr(claim, "source_ids",
                                                                                                  [])
                    block += f"- 论点 {i + 1}: {statement} [证据源: {', '.join(src_ids)}]\n"
            else:
                block += f"- 分析结论: {summary_claims}\n"

            context_blocks.append(block)

        formatted_context = "\n\n".join(context_blocks)

        # 2. 构造 Prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一名资深的海洋情报主编。你的任务是根据多个子任务的分析片段，撰写一份带有严格证据链的“海洋态势深度分析报告”。

    【溯源追踪与结构要求】：
    1. 报告必须包含标题、带有源 ID 的执行摘要、2-4个主体章节（sections）以及结语。
    2. 每个章节的正文必须拆分为多个 Claim（论点）。
    3. 对于你写下的每一个 Claim，必须从下方的上下文中找到支撑它的 `DOC_ID`，并填入 `source_ids` 数组。
    4. 确保 `ref_task_ids` 准确映射了该章节引用的子任务 ID。
    5. 仅使用下方提供的信息，严禁捏造任何内容或 DOC_ID。如果信息不足，请如实总结当前已知情况。"""),
            ("user", "用户意图: {intent}\n\n# Input Data (各子任务结论与证据):\n{context}")
        ])

        # 3. 调用 LLM 并强制结构化输出
        print("   🧠 正在调用 LLM 进行带有证据链的汇总写作...")
        try:
            # 注意：这里假设你的 model 是支持 with_structured_output 的（如 ChatOpenAI）
            structured_llm = model.with_structured_output(FinalReport)
            chain = prompt | structured_llm

            final_report_obj = chain.invoke({
                "intent": json.dumps(intent, ensure_ascii=False),
                "context": formatted_context
            })

            # 将 Pydantic 对象转换为字典
            final_report = final_report_obj.model_dump()
            print(f"   ✅ 报告生成成功: {final_report.get('report_title')}")

        except Exception as e:
            print(f"   ❌ 报告生成或解析失败: {e}")
            # 优雅降级的 Fallback 机制
            final_report = {
                "report_title": "分析报告生成失败",
                "executive_summary": "整合阶段发生错误，无法生成结构化报告。",
                "executive_source_ids": [],
                "sections": [],
                "conclusion": f"Error details: {str(e)}"
            }

        # 4. 构建终极 Payload
        integrated_payload = {
            "report": final_report,
            "tasks": raw_results,
            "evidence_pool": evidence_pool  # 将全局证据字典一并打包给前端
        }

        # 5. 返回更新状态
        return {
            "final_report": final_report,
            "analysis_results": integrated_payload,
            "current_phase": "ready",  # 或者 "finish"，取决于你的图定义
            "messages": [{"role": "ai", "content": f"报告《{final_report.get('report_title')}》已生成，并完成全链路溯源。"}]
            # 根据你的状态定义调整 Message 格式
        }

    # def integrating_node(state: ResearchState) -> dict:
    #     """
    #     整合节点：汇总子Agent的文本结果，生成叙事报告。
    #     可视化组件的渲染数据假设已在各子Agent中生成并存储，此处仅做ID引用。
    #     """
    #     print("\n" + "=" * 50)
    #     print("📝 进入整合报告阶段 (Integrating Node)...")
    #
    #     intent = state.get("intent", {})
    #     # 过滤掉非文本的大型数据，只保留 task_id, agent_name 和 文本结论/summary
    #     # 这样可以减少 Token 消耗，避免把 huge raw data 塞给 LLM
    #     simplified_results = []
    #
    #     raw_results = state.get("task_results", {})
    #
    #     for task_id, res in raw_results.items():
    #         # 假设子Agent的结果结构是 {"summary": "...", "viz_data": "...", "raw_data": "..."}
    #         # 我们只提取 summary 部分给 Integrating Node
    #         summary_text = res.get("summary") if isinstance(res, dict) else str(res)
    #         agent_name = res.get("agent_name", "Unknown-Agent") if isinstance(res, dict) else "Task"
    #
    #         simplified_results.append({
    #             "task_id": task_id,
    #             "agent": agent_name,
    #             "analysis_text": summary_text
    #         })
    #
    #     # 1. 构造 Prompt
    #     prompt_content = f"""
    #     # Role
    #     你是一名资深的海洋情报主编。你的任务是根据多个子任务的分析片段，撰写一份“海洋态势深度分析报告”。
    #
    #     # Context
    #     用户意图: {json.dumps(intent, ensure_ascii=False)}
    #
    #     # Input Data (各子任务的分析结论)
    #     {json.dumps(simplified_results, ensure_ascii=False, indent=2)}
    #
    #     # Output Format (JSON)
    #     请输出如下 JSON 格式，其中 `ref_task_ids` 字段用于关联已生成的图表：
    #     {{
    #       "report_title": "报告标题",
    #       "executive_summary": "摘要",
    #       "sections": [
    #         {{
    #           "subtitle": "小标题",
    #           "content": "详细分析...",
    #           "ref_task_ids": [1]
    #         }}
    #       ],
    #       "conclusion": "结语"
    #     }}
    #
    #     # Constraints
    #     1. 请仅使用子任务的分析结果进行汇总整理，不要编造或自行搜索相关结果，如果前序的子任务结果不足以支撑完整一份报告，请你如实回答
    #     """
    #
    #     # 2. 调用 LLM
    #     print("   正在调用 LLM 进行汇总写作...")
    #     response = model.invoke([HumanMessage(content=prompt_content)])
    #     # response = {}
    #     # 3. 解析结果
    #     try:
    #         final_report = safe_parse_json(response.content)
    #         print(f"   ✅ 报告生成成功: {final_report.get('report_title')}")
    #     except Exception as e:
    #         print(f"   ❌ 报告解析失败: {e}")
    #         # Fallback 机制
    #         final_report = {
    #             "report_title": "分析报告生成失败",
    #             "executive_summary": "无法解析 LLM 输出。",
    #             "sections": [],
    #             "raw_output": response.content
    #         }
    #
    #
    #     task_results = state.get("task_results", {})
    #
    #     # 构建最终统一的输出负载
    #     integrated_payload = {
    #         "report": final_report,  # 包含标题、摘要和各个 section (含 ref_task_ids)
    #         "tasks": task_results  # 包含所有 Agent 的图表数据、摘要和具体新闻/事件列表
    #     }
    #
    #     # 4. 更新状态
    #     # 注意：这里我们把生成的 final_report 存入状态
    #     # 状态流转回 ready (0) 或 finish，等待用户下一次交互
    #     return {
    #         "final_report": final_report,
    #         "analysis_results": integrated_payload,
    #         "current_phase": "ready",
    #         "messages": [AIMessage(content=f"报告《{final_report.get('report_title')}》已生成。")]
    #     }


    graph = StateGraph(ResearchState)

    graph.add_node("intent", intent_node)
    graph.add_node("planning", planning_node)
    graph.add_node("check", check_node)  # 加入 check 节点
    graph.add_node("analysis", analyzing_node)
    graph.add_node("integrate", integrating_node)

    graph.add_edge(START, "intent" )
    graph.add_edge("intent", "planning")
    #$graph.add_edge("planning", "analysis")
    graph.add_edge("planning", "check")  # planning 结束后走到 check

    # 使用条件边决定 check 之后的去向
    graph.add_conditional_edges("check", route_after_check)
    graph.add_edge("analysis", "integrate")
    graph.add_edge("integrate", END)
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
    compiled_graph = graph.compile(checkpointer=memory, interrupt_before=["check"])

    return compiled_graph


# def run_research(topic: str):
#     """运行研究任务"""
#     print("\n" + "=" * 60)
#     print("🔬 启动研究任务")
#     print("=" * 60)
#     print(f"研究主题: {topic}")
#     print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#
#     # 创建研究助手
#     assistant = create_visual_analytics_assistant()
#
#     # 初始状态
#     initial_state = {
#         "messages": [HumanMessage(content=f"请对以下主题进行深入研究：{topic}")],
#         "research_topic": topic,
#         "research_questions": [],
#         "intent": [],
#         "plan": {},
#         "findings": [],
#         "task_results": {},
#         "final_report": "",
#         "draft_sections": {},
#         "current_phase":"",
#         "iteration_count": 0,
#         "research_list": []
#     }
#
#     # 运行研究流程
#     config = {"configurable": {"thread_id": f"research_{datetime.now().strftime('%Y%m%d%H%M%S')}"}}
#     result = assistant.invoke(initial_state, config)
#     # 输出结果
#     print("\n" + "=" * 60)
#     print("📄 研究报告")
#     print("=" * 60)
#     print(result.get("final_report", "报告生成失败"))
#
#     return result
#
# if __name__ == '__main__':
#     topic = "对2025年第四季度美国在深海采矿方面采取的一系列行动"
#     run_research(topic)



# 假设这里导入了你的图构建函数
# from your_module import create_visual_analytics_assistant

def run_research_hitl(topic: str):
    """运行带有 Human-in-the-Loop (HITL) 审批机制的研究任务"""
    print("\n" + "=" * 60)
    print("🔬 启动多Agent可视分析系统 (CLI 测试版)")
    print("=" * 60)
    print(f"研究主题: {topic}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. 创建研究助手 (注意：内部必须 compile(checkpointer=memory, interrupt_before=["check"]))
    assistant = create_visual_analytics_assistant()

    # 2. 初始状态
    initial_state = {
        "messages": [HumanMessage(content=f"请对以下主题进行深入研究：{topic}")],
        "research_topic": topic,
        "research_questions": [],
        "intent": [],
        "plan": {},  # 等待 planner 填充
        "user_feedback": "",  # 新增：用于接收 CLI 输入的反馈
        "findings": [],
        "task_results": {},
        "final_report": "",
        "draft_sections": {},
        "current_phase": "",
        "iteration_count": 0,
        "research_list": []
    }

    # 3. 配置 thread_id，这对记忆保存和恢复至关重要
    config = {"configurable": {"thread_id": f"research_{datetime.now().strftime('%Y%m%d%H%M%S')}"}}

    print("\n🚀 [系统] 正在进行意图识别与任务编排 (Planning)...")

    # 4. 首次启动图：它会运行到 interrupt_before 指定的节点（如 "check"）然后暂停
    assistant.invoke(initial_state, config)

    # 5. 进入交互循环（模拟前后端多次握手）
    while True:
        # 获取当前图的状态快照
        state_snapshot = assistant.get_state(config)

        # 检查图是否已经执行完毕 (没有 next node 说明跑到了 END)
        if not state_snapshot.next:
            print("\n✅ [系统] 分析与整合流程已全部完成！")
            break

        # 如果图暂停在了 check 节点前
        if "check" in state_snapshot.next:
            # 读取 Planner 刚刚生成的计划
            current_plan = state_snapshot.values.get("plan", "暂无计划内容")

            print("\n" + "=" * 60)
            print("⏸️ [审批节点] 系统需要您的确认才能继续执行")
            print("=" * 60)
            print(f"当前生成的调用计划 (Plan): \n{current_plan}")
            print("-" * 60)

            # 使用 input 模拟前端 Vue 的对话框输入
            user_input = input("💡 请审核计划 (输入 'y' 确认执行，或输入修改意见让AI重做): ").strip()
            current_phase = "analyzing"
            # 逻辑判断：确认还是打回
            if user_input.lower() in ['y', 'yes', 'ok', '同意', '确认']:
                feedback = "approve"
                print("\n▶️ [系统] 您已批准计划。正在调用各个 Agent 进行数据检索与可视化生成...")
            else:
                feedback = user_input
                current_phase = "planning"
                print(f"\n🔄 [系统] 收到您的修改意见：'{feedback}'。正在让 Planner 重新规划...")

            # 6. 核心步骤：将用户的意见注入到图的状态中
            # 注意：如果你的 state 里定义的键名不是 user_feedback，请与你的 StateDict 保持一致
            assistant.update_state(config, {"user_feedback": feedback , "current_phase": current_phase})

            # 7. 唤醒图继续执行。传入 None 表示使用已更新的 state 继续向下走
            assistant.invoke(None, config)

        else:
            # 容错：如果停在了预料之外的节点，直接尝试继续往下跑
            print(f"\n⚠️ [系统] 图暂停在了非审批节点: {state_snapshot.next}，尝试自动继续...")
            assistant.invoke(None, config)

    # 8. 循环结束，输出最终结果
    final_state = assistant.get_state(config).values
    print("\n" + "=" * 60)
    print("📄 最终可视化方案与分析报告")
    print("=" * 60)
    print(final_state.get("final_report", "报告生成失败"))

    return final_state


if __name__ == '__main__':
    topic = "对2025年第四季度美国在深海采矿方面采取的一系列行动"
    run_research_hitl(topic)
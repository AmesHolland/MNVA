import json
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from agent.config.llm_config import llm_qw_quick
from agent.tools.base import safe_parse_json
# 从上面定义的模块导入依赖
from ..schemas import ResearchState, FinalReport, SpatiotemporalBlueprint, ExecutionPlan
from agent.config.prompt_template import get_intent_prompt, get_planning_prompt, \
    get_data_profiling_prompt, INTEGRATING_PROMPT, ANCHOR_PROMPT, PLANNING_PROMPT
from agent.agents.sub_agent.agent_wrappers import AGENT_MAPPING
from ...tools.news_manager import get_news_by_id

model = llm_qw_quick



def intent_node(state: ResearchState) -> dict:
    print("\n" + "=" * 50 + "\n 用户意图识别阶段...")
    topic = state["research_topic"]
    intent_prompt = get_intent_prompt(topic)

    response = model.invoke([HumanMessage(content=intent_prompt)])
    intent = safe_parse_json(response.content)  # 假设该工具函数已导入

    print(f" primary intent: {intent.get('primary_intent')}")
    print(f" spatial_scope: {intent.get('spatial_scope')}")

    return {
        "intent": intent,
        "research_topic": topic,
        "current_phase": "planning",
        "messages": [AIMessage(content=f"用户意图已经解读完成：{topic}")]
    }


def route_after_intent(state: ResearchState) -> str:
    """根据意图复杂度进行双轨路由"""
    complexity = state.get("intent", {}).get("task_complexity", "deep_research")
    if complexity == "simple_qa":
        return "simple_chat"
    else:
        return "data_retrieval" # 进入深度分析的起点

def simple_chat_node(state: ResearchState) -> dict:
    """快分支：处理基础问答、闲聊或简单解释"""
    print("\n" + "=" * 50 + "\n💬 进入轻量级问答分支 (Fast Track)...")
    topic = state["research_topic"]

    # 直接调用 LLM 回答，或者结合少量历史信息
    prompt = f"你是一个海洋态势感知系统的智能助手。请简明扼要地回答用户的问题：{topic}"
    response = model.invoke([HumanMessage(content=prompt)])

    return {
        "current_phase": "ready",
        "final_report": response.content,  # 可以复用这个字段，前端判断如果有 report 则直接展示
        "messages": [AIMessage(content=response.content)]
    }

def data_retrieval_node(state: ResearchState) -> dict:
    """前置检索：根据用户意图，去数据库捞取基础数据池"""
    print("\n" + "=" * 50 + "\n 捞取基础数据 (Data Retrieval)...")
    query = state.get("research_topic")

    # 这里的底层逻辑就是你原来 Search_Agent 的逻辑
    # 获取初步的 20-30 条新闻
    news_list = get_news_by_id(query)

    return {"news_list": news_list}

def data_profiling_node(state: ResearchState) -> dict:
    """数据探路：扫描检索到的新闻，提取真实存在的实体和热点"""
    print("\n" + "=" * 50 + "\n 数据探路 (Data Profiling)...")
    news_list = state.get("news_list", [])

    profiling_prompt = get_data_profiling_prompt(news_list)
    response = model.invoke([HumanMessage(content=profiling_prompt)])
    profile_data = safe_parse_json(response.content)
    print(f"   🎯 真实存在的实体: {profile_data.get('actual_entities')}")

    return {"analysis_results": {"data_profile": profile_data}}  # 暂存在状态中供 Planner 使用


# === 4. 节点函数实现 ===
def spatiotemporal_scoping_anchor_node(state: dict) -> dict:
    """
    时空范围锚定节点：读取轻量级元数据骨架，生成时空演化蓝图 (Blueprint)
    """
    print("\n" + "=" * 50)
    print("🧭 进入时空范围锚定节点 (Spatiotemporal Scoping Anchor)...")

    # === 2. 初始化 Parser 与 Prompt ===
    anchor_parser = JsonOutputParser(pydantic_object=SpatiotemporalBlueprint)

    anchor_prompt = PromptTemplate(
        template=ANCHOR_PROMPT,
        input_variables=["intent", "metadata_skeleton"],
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
            "metadata_skeleton": metadata_skeleton_str
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

def planning_node(state: dict) -> dict:
    print("\n" + "=" * 50)
    print("📋 [Planner] 进入任务规划阶段 (Task Orchestration)...")

    # ==========================================
    # 初始化 Parser 与 Prompt Template
    # ==========================================
    plan_parser = JsonOutputParser(pydantic_object=ExecutionPlan)

    # 使用 LangChain 原生的 PromptTemplate 处理变量注入
    planning_prompt = PromptTemplate(
        template=PLANNING_PROMPT,
        input_variables=["user_query", "intent_data", "review", "blueprint", "actual_entities", "actual_topics"],
        partial_variables={"format_instructions": plan_parser.get_format_instructions()}
    )

    intent_data = state.get("intent", {})
    user_query = intent_data.get("original_query", "")
    feedback = state.get("user_feedback", "")
    plan_history = state.get("plan")

    # 获取来自 Profiling 和 Anchor 节点的时空数据
    profile_data = state.get("analysis_results", {}).get("data_profile", {})
    blueprint = state.get("spatiotemporal_blueprint", {})

    review = ""
    if feedback and feedback != "approve":
        review = f"【Human-in-the-Loop 反馈】: 用户拒绝了之前的计划 ({plan_history})，并提出了修改意见：'{feedback}'。请严格按照用户的意见修正规划！"

    print("   🧠 正在解析时空蓝图，生成动态数据路由计划 (JsonOutputParser)...")
    try:
        # 注意：这里的 model 需要是你已经实例化好的大模型对象 (例如 llm_qw_quick)
        planning_chain = planning_prompt | model | plan_parser

        # invoke 会自动填充变量，并返回经过 Pydantic 验证后的 Python 字典
        plan_dict = planning_chain.invoke({
            "user_query": user_query,
            "intent_data": json.dumps(intent_data, ensure_ascii=False),
            "review": review,
            "blueprint": json.dumps(blueprint, ensure_ascii=False, indent=2),
            "actual_entities": json.dumps(profile_data.get('actual_entities', []), ensure_ascii=False),
            "actual_topics": json.dumps(profile_data.get('actual_topics', []), ensure_ascii=False)
        })

        print(f"   ✅ 规划逻辑: {plan_dict.get('total_plan_logic')}")
        plan = plan_dict

    except Exception as e:
        print(f"   ❌ 规划解析失败: {e}")
        plan = {
            "total_plan_logic": "解析失败，启动保底执行策略。",
            "tasks": [{
                "task_id": 1,
                "agent": "Global_Monitor_Agent",
                "action": "Fallback execution",
                "args": {"query": user_query, "target_phase_ids": []},
                "dependency": None
            }]
        }

    return {
        "plan": plan,
        "current_phase": "confirming",  # 流转至 HITL 审批卡片
        "messages": [{"role": "ai", "content": f"已规划任务：{plan.get('total_plan_logic')}"}]
    }


def check_node(state: ResearchState) -> dict:
    feedback = state.get('user_feedback')
    print(f"--- 接收到用户反馈: {feedback} ---")
    return {}

def route_after_check(state: ResearchState) -> str:
    feedback = state.get("user_feedback", "")
    return "analysis" if feedback == "approve" else "planning"

def analyzing_node(state: ResearchState) -> dict:
    print("\n" + "=" * 50 + "\n🚀 进入分析执行阶段...")
    plan = state["plan"]
    tasks = plan.get("tasks", [])
    task_results = {}
    # 【修复点 2】：直接从全局状态提取新闻列表
    global_news_list = state.get("news_list", [])
    spatiotemporal_blueprint = state.get("spatiotemporal_blueprint")

    for task in tasks:
        task_id = task["task_id"]
        agent_name = task["agent"]
        raw_args = task["args"]
        action = task["action"]
        dependency_ids = task.get("dependency")
        print(f"\n执行任务 [{task_id}]: {agent_name} - {task['action']}")

        execution_args = raw_args.copy()

        # 【修复点 3】：无视 Planner 的任务依赖，直接将全局新闻强塞进参数中
        execution_args["global_news_list"] = global_news_list
        execution_args["action"] = action
        execution_args["blueprint_overall_narrative"] = spatiotemporal_blueprint["overall_narrative"]

        input_data = None

        # === 核心依赖处理逻辑 (保持原样) ===
        if dependency_ids is not None:
            if isinstance(dependency_ids, list):
                print(f"   └── 🔗 检测到多重依赖: {dependency_ids}")
                input_data = [task_results.get(dep_id) for dep_id in dependency_ids if
                              task_results.get(dep_id) is not None]
            else:
                input_data = task_results.get(dependency_ids)

        if input_data is None:
            for key, value in raw_args.items():
                if key == "input_from_task":
                    if isinstance(value, list):
                        input_data = [task_results.get(v) for v in value]
                    else:
                        input_data = task_results.get(value)
                    break

        if input_data:
            data_type = "List" if isinstance(input_data, list) else type(input_data).__name__
            print(f"   └── 📦 成功注入数据 (Type: {data_type})")
            execution_args["input_data"] = input_data

        # === 执行调用 ===
        if agent_name in AGENT_MAPPING:
            try:
                result = AGENT_MAPPING[agent_name](execution_args)

                task_results[task_id] = result
                print(f"   ✅ 完成. 结果摘要已保存。")
            except Exception as e:
                print(f"   ❌ 执行出错: {str(e)}")
                task_results[task_id] = {"error": f"执行出错: {str(e)}"}
        else:
            print(f"   ⚠️ 未定义的 Agent: {agent_name}")
            task_results[task_id] = "Agent Not Found"

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
        input_variables=["intent", "context"],
        partial_variables={"format_instructions": integrating_parser.get_format_instructions()}
    )

    # 组装 Chain (注意：这里的 model 替换为你实际使用的 llm 实例，比如 llm_qw_quick)
    integrating_chain = integrating_prompt | model | integrating_parser

    intent = state.get("intent", {})
    raw_results = state.get("task_results", {})
    evidence_pool = state.get("news_list", {})
    blueprint = state.get("spatiotemporal_blueprint", {})  # 【极度重要】：获取 Anchor 的蓝图
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
            "context": formatted_context
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
        "spatiotemporal_blueprint": blueprint
    }

    return {
        "final_report": final_report,
        "analysis_results": integrated_payload,
        "current_phase": "ready",
        "messages": [{"role": "ai", "content": f"报告《{final_report.get('report_title')}》已生成。"}]
    }

# def integrating_node(state: ResearchState) -> dict:
#     # 初始化 Parser，绑定你的 Pydantic 模型
#     integrating_parser = JsonOutputParser(pydantic_object=FinalReport)
#
#     # 创建 PromptTemplate，注入 format_instructions
#     integrating_prompt = PromptTemplate(
#         template=INTEGRATING_PROMPT,
#         input_variables=["intent", "context"],
#         partial_variables={"format_instructions": integrating_parser.get_format_instructions()}
#     )
#
#     # 组装 Chain (注意：这里的 model 替换为你实际使用的 llm 实例，比如 llm_qw_quick)
#     integrating_chain = integrating_prompt | model | integrating_parser
#
#     print("\n" + "=" * 50 + "\n📝 进入整合报告阶段...")
#     intent = state.get("intent", {})
#     raw_results = state.get("task_results", {})
#     evidence_pool = state.get("news_list", {})
#
#     context_blocks = []
#     for task_id, res in raw_results.items():
#         agent_name = res.get("agent_name", "Unknown-Agent")
#         summary_claims = res.get("summary", [])
#         block = f"### Task ID: {task_id} (Agent: {agent_name})\n"
#         if isinstance(summary_claims, list):
#             for i, claim in enumerate(summary_claims):
#                 statement = claim.get("statement") if isinstance(claim, dict) else getattr(claim, "statement",str(claim))
#                 src_ids = claim.get("source_ids", []) if isinstance(claim, dict) else getattr(claim, "source_ids", [])
#                 block += f"- 论点 {i + 1}: {statement} [证据源: {', '.join(src_ids)}]\n"
#         else:
#             block += f"- 分析结论: {summary_claims}\n"
#         context_blocks.append(block)
#
#     formatted_context = "\n\n".join(context_blocks)
#
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", INTEGRATING_SYSTEM_PROMPT),
#         ("user", "用户意图: {intent}\n\n# Input Data (各子任务结论与证据):\n{context}")
#     ])
#
#     print("   🧠 正在调用 LLM 进行带有证据链的汇总写作...")
#     try:
#         structured_llm = model.with_structured_output(FinalReport)
#         chain = prompt | structured_llm
#         final_report_obj = chain.invoke({
#             "intent": json.dumps(intent, ensure_ascii=False),
#             "context": formatted_context
#         })
#         final_report = final_report_obj.model_dump()
#         print(f"   ✅ 报告生成成功: {final_report.get('report_title')}")
#     except Exception as e:
#         print(f"   ❌ 报告生成失败: {e}")
#         final_report = {"report_title": "分析报告生成失败", "executive_summary": str(e), "executive_source_ids": [],
#                         "sections": [], "conclusion": ""}
#
#     integrated_payload = {"report": final_report, "tasks": raw_results, "evidence_pool": evidence_pool}
#
#     return {
#         "final_report": final_report,
#         "analysis_results": integrated_payload,
#         "current_phase": "ready",
#         "messages": [{"role": "ai", "content": f"报告《{final_report.get('report_title')}》已生成。"}]
#     }
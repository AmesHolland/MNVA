---

# 海洋新闻多 Agent 后端开发手册（LangGraph 1.x）

## 0. 你在开发什么

这是一个“主控编排 + 多子 Agent 工具执行”的 Agentic-RAG 框架：

* **主 Agent（Orchestrator）**：负责意图识别、任务规划、任务调度、HITL 审批闭环、最终回答生成、可视化参数汇总与落地。
* **子 Agent（Search/Time/Topic/Entity…）**：负责执行具体能力（检索、时序、热点发现、实体跟踪…），把产物写回共享 State。
* **状态 State（OceanState）**：贯穿图中所有节点与所有 Agent 的“共享工作记忆”。各模块通过“读 state → 产出 partial update dict → 合并回 state”协作。
* **持久化（checkpointer + thread_id）**：同一对话线程可以中断/恢复、重启服务也能恢复；适配 HITL。

> 重要原则：
> **主控图不做业务分析，只做流程/状态管理。子 Agent 才做业务分析。**

---

## 1. 代码目录结构与职责（给新同伴看的快速地图）

推荐目录结构（你现在框架）：

```
agent/
  config/
    rag_config.py            # 
    app_config.py            # 全局配置：checkpoint路径、最大步数等
  state.py                   # OceanState schema + reducers（状态合并规则）

  memory/
    checkpoint.py            # checkpointer 工厂（SqliteSaver / InMemory）
    store.py                 # 用户长期记忆 store（可选）

  agents/
    base.py                  # BaseAgent 抽象：invoke(state, config) -> partial_update
    registry.py              # agent 名称 -> agent 实例
    
    sub_agent/
        search_agent.py          # 子Agent示例（你填业务）
        time_evolution_agent.py  # 子Agent示例（你填业务）
    ...

    controller/
      types.py               # Plan/Task/Approval 的数据结构定义
      planner.py             # Planner：把 query/constraints -> 任务列表
      graph.py               # 主工作图：节点、边、interrupt、循环
      main_agent.py          # MainAgent封装：run_turn/resume_turn/get_state

  services/
    runner.py                # 对外调用封装：start/resume/state

scripts/
  run_local_interactive.py   # 交互式运行：单步决策+打印中间变量
```

---

## 2. 核心概念：State、Plan、Task、Agent 输出规范

### 2.1 State 是团队协作的“接口”

所有子 Agent 都要遵守一个约定：

* **输入**：从 `state` 里拿到你需要的信息（query、constraints、上游新闻列表等）
* **输出**：返回一个 `dict`（partial state update，也就是返回的dict是state的一部分），主控图会把它合并进 state
* **不要**直接在 `invoke()` 内修改传入的 `state`（推荐约定：只通过返回 dict 更新）

`state.py` 里定义了 OceanState 的核心字段（团队成员需要统一理解它们的语义）。

### 2.2 Plan 与 Task：主控如何调度子 Agent

* **Plan**：本轮解题的路线图（goal + tasks_primary + stop_rule）
* **Task**：路线图里的具体执行步骤（调用哪个 agent、输入参数是什么、状态是什么）

主控图会把 Plan 中的 `tasks_primary` 转成 `state.tasks`（map 结构：task_id -> task），并循环执行：

1. 找到下一个 `todo` 的 task
2. 标记 running
3. 调对应子 Agent：`agent.invoke(state)`
4. 合并子 Agent 输出
5. 标记 done
6. 循环直到没有 todo task → 生成最终答案

---

## 3. 子 Agent 的“输入从哪来”（统一输入解析策略）

为了让子 Agent 稳定、可复用、可被 Planner 精确控制，推荐统一按优先级取输入：

1. **当前 task.inputs（最优先）**

   * `state["tasks"][state["current_task_id"]]["inputs"]`
2. **全局 constraints（用户偏好/范围约束）**

   * `state.get("constraints", {})`
3. **query（兜底）**

   * `state.get("query", "")`
4. **上游产物（数据流承接）**

   * 如：`news_list / news_selected_ids / findings ...`

强烈建议提供一个统一 helper（放 `agents/utils.py`）：

```python
def resolve_inputs(state: dict) -> dict:
    task_id = state.get("current_task_id")
    task_inputs = {}
    if task_id and "tasks" in state and task_id in state["tasks"]:
        task_inputs = state["tasks"][task_id].get("inputs", {}) or {}

    constraints = state.get("constraints", {}) or {}
    query = state.get("query", "") or ""

    merged = {**constraints, **task_inputs}
    merged.setdefault("query", query)
    merged.setdefault("task_id", task_id)
    return merged
```

> 团队约定：子 Agent **尽量不要**直接依赖 Planner 的内部实现，只依赖 `resolve_inputs()` 得到的 merged inputs。

---

## 4. 子 Agent 应该返回什么（输出规范：必须/推荐/禁止）

### 4.1 必须字段（强烈建议强制执行）

每个子 Agent 至少返回：

1. `messages: [AIMessage(...)]`

   * 用于日志、LangSmith、交互式打印：你做了什么、结论是什么

2. `tool_trace: [ {...} ]`

   * 用于审计/回放/排错：输入参数摘要、输出统计、耗时、失败原因等

推荐的 `tool_trace` 结构（统一字段，大家看起来一致）：

```python
{
  "agent": "TimeEvolutionAgent",
  "task_id": "t_time",
  "status": "ok" | "failed" | "stub",
  "inputs_summary": {...},
  "outputs_summary": {...},
  "cost": {...},           # 可选：tokens/latency
}
```

### 4.2 推荐字段（按你的业务需要返回）

* `news_list`：若本步产生/更新新闻列表（只放 meta，不放全文）
* `news_selected_ids`：若本步筛选出子集供后续深分析
* `findings`：结构化发现（主题簇、实体链、拐点、迁移路径…）
* `evidence_index`：可验证性映射（claim_id -> [news_id…]）
* `viz_buffer`：可视化 params（字段 + 来源 + data_key 指针）
* `errors`：错误/冲突/缺数据（列表）

### 4.3 禁止/不推荐（避免 state 膨胀）

* 新闻全文、长文本、成千上万行的可视化数据数组
  → 应外置存储（DB/Redis/对象存储），state 里只存 `store_key/data_key`

---

## 5. 各子 Agent 推荐的读写字段

### SearchAgent（检索 + rerank）

**读**：`query`、`constraints(time_range/geo/sources/lang)`
**写**：`news_list`、`messages`、`tool_trace`（可选：`viz_buffer` 初步趋势）

### TimeEvolutionAgent（时序分析）

**读**：`news_selected_ids`（优先）或 `news_list`（兜底）、`constraints.time_range`、`inputs.granularity`
**写**：`findings.time_evolution`、`viz_buffer(line_trend/timeline)`、`messages`、`tool_trace`

### TopicAnalysis（热点发现/跟踪）

**读**：`news_list`、`constraints.time_range`、`inputs.hotspot_definition`
**写**：`findings.hotspots`、`evidence_index`、`viz_buffer`、`messages`、`tool_trace`

### EntityTrackingAgent（实体跟踪）

**读**：`inputs.entities`（强依赖）、`constraints.time_range/geo`
**写**：`findings.entity_timelines`、`evidence_index`、`viz_buffer`、`messages`、`tool_trace`

---

## 6. 主工作图工作流程

主图的核心流程（忽略 recall）：

1. **ingest_input**：把 query 写入 messages（HumanMessage）
2. **intent_router**：得到 intent（规则/LLM）
3. **plan_draft**：Planner 生成 plan + tasks（任务列表）
4. **hitl_plan_approval**：interrupt 让用户确认计划（可选：你可关掉 HITL）
5. **execute_next_task**：循环取下一个 todo task，调用对应子 Agent
6. **draft_answer**：基于 state 汇总生成最终回答（LLM）
7. **hitl_answer_approval**：interrupt 让用户确认最终回答（可选）
8. **finalize**：落地 viz_buffer → viz_key（可选）
9. END

> 关键点：
>
> * 子 Agent 的输出都写进 state
> * 最终回答只读取 state 的结构化产物（findings/evidence/news_meta）
> * HITL 通过 interrupt/resume 控制流程“走向”

---

## 7. LLM 在哪里用

LLM 不会写在“图结构”里，而是在**节点/Agent 内部**调用：

* `intent_router`：LLM 做意图分类 + 约束抽取（time/geo/entities）
* `planner.draft_plan`：LLM 把 query → tasks_primary（更稳定建议 structured output）
* `draft_answer`：LLM 基于 `findings + evidence_index + news_list(meta)` 生成可验证回答
* 子 Agent 内部：LLM 抽取事件、聚类、总结、冲突检测、多模型交叉验证等

> LLM可以直接调用 agent/config/llm_config.py llm

---

## 8. 如何开发一个新的子 Agent

以 `TimeEvolutionAgent` 为例，同伴需要做 4 步：

### Step 1：创建类，继承 BaseAgent

```python
class TimeEvolutionAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="TimeEvolutionAgent")

    def invoke(self, state: Dict[str, Any], config=None) -> Dict[str, Any]:
        inp = resolve_inputs(state)
        ...
        return {...}
```

### Step 2：从 state 读取输入（统一用 resolve_inputs）

* 必须知道：要处理哪批新闻

  * 优先：`state.news_selected_ids`
  * 兜底：`state.news_list`

### Step 3：写回结构化产物（findings/viz_buffer）

* 把“时序桶统计/拐点/阶段总结”写在 `findings["time_evolution"]`
* 把“趋势线/时间轴 spec” append 到 `viz_buffer`

### Step 4：保证 messages + tool_trace

* messages：一句给人看的总结
* tool_trace：记录 inputs_summary、outputs_summary

---

## 9. 可观测性与调试（print / 交互式脚本 / LangSmith）

### 9.1 推荐每个子 Agent 都输出 messages + tool_trace

这样你不用进 LangSmith，也能从交互式脚本看到每一步在做什么。

### 9.2 交互式脚本（推荐）

用 `scripts/run_local_interactive.py`：

* `new`：输入问题 → 跑到 interrupt 或 END
* `resume`：给 interrupt 继续输入（approved/feedback/json）
* `state`：随时查看当前 state（含 findings/tool_trace）

---

## 10. 团队开发规范

### 10.1 命名与稳定性

* `agent.name` 要和 Planner 里 task 的 `agent` 字段一致（registry 通过 name 匹配）
* task_id 必须唯一（建议 `t_xxx` + uuid 后缀）

### 10.2 不要让 state 变成垃圾场

* 长文本/全文/大数组 → 外置存储，只在 state 存 key
* findings 保持结构化、可复用、可汇总
* tool_trace 只放摘要，不放全文

### 10.3 错误处理

* 子 Agent 失败：返回 `errors += {...}` 并在 tool_trace 标记 status=failed
* 主控可以选择：失败即停 / 继续下一个 task / 触发重新规划（未来扩展）

---

## 11. 最小“子 Agent 输出模板”

```python
return {
  "messages": [AIMessage(content="本步做了什么 + 关键结论（尽量短）")],
  "tool_trace": [{
      "agent": self.name,
      "task_id": inp.get("task_id"),
      "status": "ok",
      "inputs_summary": {...},
      "outputs_summary": {...},
  }],
  # 可选：
  "news_list": [...],
  "news_selected_ids": [...],
  "findings": {...},
  "evidence_index": {...},
  "viz_buffer": [...],
  "errors": [...],
}
```

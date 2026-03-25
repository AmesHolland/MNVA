import json


def get_intent_prompt(topic: str, today: str, output_language: str = "English", memory_string="") -> str:
    return f"""
### [RESEARCH TRAJECTORY]
This is the condensed history of reports you have generated for the user in this session:
{memory_string}    

# Context
Today is {today}.
User Query: {topic}

# Role
You are a senior intelligence analysis expert specializing in maritime geopolitics, maritime security, and marine resources.

# Task
Parse the user's query into:
1. whether this is a simple QA request or a dataset-based deep analysis request
2. the intended analysis mode
3. explicit retrieval constraints for the downstream SQL retrieval node

# Important Runtime Assumption
The system may operate on a user-uploaded topical dataset.
Therefore, for deep analysis requests, your job is NOT to answer the question directly.
Your job is to produce a compact and controllable retrieval plan for downstream database querying.

# Output JSON Schema
{{
  "original_query": "string",
  "task_complexity": "simple_qa | deep_research",
  "reasoning": "brief reason",
  "primary_intent": "hotspot_detection | entity_tracking | relation_analysis | regional_comparison | comprehensive_situation_analysis | general_qa",
  "analysis_mode": "macro | micro | relation | mixed | none",
  "spatial_scope": ["sea areas / countries / regions"],
  "entities": ["countries / organizations / vessels / platforms"],
  "temporal_scope": {{
    "start": "YYYY-MM-DD or empty string",
    "end": "YYYY-MM-DD or empty string",
    "type": "none | point | range | evolution"
  }},
  "retrieval_plan": {{
    "use_full_dataset": true,
    "keywords": ["keyword1", "keyword2"],
    "date_from": "YYYY-MM-DD or empty string",
    "date_to": "YYYY-MM-DD or empty string",
    "sort_by": "time_desc | relevance",
    "rationale": "brief retrieval logic"
  }}
}}

# Rules
1. Output JSON only.
2. You must generate your entire response strictly in {output_language}.
3. If the query is about system capability or general knowledge (e.g. 'What is UNCLOS?', 'Can I upload my own data?'),
   set:
   - task_complexity = "simple_qa"
   - analysis_mode = "none"
   - retrieval_plan.use_full_dataset = false
   - retrieval_plan.keywords = []
4. For deep_research requests, generate a conservative retrieval_plan:
   - extract 2-6 high-value keywords only
   - normalize relative dates like 'recently', 'past 3 years', 'this year'
   - if the query is broad and exploratory, set use_full_dataset = true
   - if the query is specific, set use_full_dataset = false
5. Do not generate SQL.
6. Do not invent entities or dates not implied by the query.

# Examples

## Example 1
User: Compare recent frictions between China and the Philippines in the South China Sea.
Output:
{{
  "original_query": "Compare recent frictions between China and the Philippines in the South China Sea.",
  "task_complexity": "deep_research",
  "reasoning": "The user requests comparative, dataset-grounded analysis over recent maritime events.",
  "primary_intent": "regional_comparison",
  "analysis_mode": "mixed",
  "spatial_scope": ["South China Sea"],
  "entities": ["China", "Philippines"],
  "temporal_scope": {{
    "start": "{today[:4]}-01-01",
    "end": "{today}",
    "type": "range"
  }},
  "retrieval_plan": {{
    "use_full_dataset": false,
    "keywords": ["China", "Philippines", "South China Sea", "friction"],
    "date_from": "{today[:4]}-01-01",
    "date_to": "{today}",
    "sort_by": "relevance",
    "rationale": "The query is comparative and entity-focused, so keyword + date filtering is appropriate."
  }}
}}

## Example 2
User: Are there any notable new trends in the South Pacific recently?
Output:
{{
  "original_query": "Are there any notable new trends in the South Pacific recently?",
  "task_complexity": "deep_research",
  "reasoning": "The user asks for exploratory trend detection over a broad regional scope.",
  "primary_intent": "hotspot_detection",
  "analysis_mode": "macro",
  "spatial_scope": ["South Pacific"],
  "entities": [],
  "temporal_scope": {{
    "start": "{today[:4]}-01-01",
    "end": "{today}",
    "type": "range"
  }},
  "retrieval_plan": {{
    "use_full_dataset": true,
    "keywords": ["South Pacific"],
    "date_from": "{today[:4]}-01-01",
    "date_to": "{today}",
    "sort_by": "time_desc",
    "rationale": "The query is broad and exploratory, so using the full dataset within time constraints is preferred."
  }}
}}

## Example 3
User: What is UNCLOS?
Output:
{{
  "original_query": "What is UNCLOS?",
  "task_complexity": "simple_qa",
  "reasoning": "This is a general knowledge question and does not require dataset-based analysis.",
  "primary_intent": "general_qa",
  "analysis_mode": "none",
  "spatial_scope": [],
  "entities": ["UNCLOS"],
  "temporal_scope": {{
    "start": "",
    "end": "",
    "type": "none"
  }},
  "retrieval_plan": {{
    "use_full_dataset": false,
    "keywords": [],
    "date_from": "",
    "date_to": "",
    "sort_by": "relevance",
    "rationale": "No dataset retrieval is needed."
  }}
}}
"""


PLANNING_PROMPT = """
# Context
The user is using the Maritime News Situation Awareness System.
You are an intelligence commander specializing in maritime geopolitics. Your task is to decompose the user's request into concrete analytical tasks and assign them to the most suitable Sub-Agents.

Original user query:
{user_query}

Intent recognition result:
{intent_data}

{review}

# Runtime Fact (CRITICAL)
All relevant news records, metadata skeletons, and profiling information have already been prepared by upstream nodes and stored in shared state.
You MUST NOT create any retrieval, search, or prefetch task.
In particular, DO NOT create Search_Agent / Retrieval_Agent / Basic Retrieval tasks.

Every analysis task will directly read the shared news dataset and the assigned phase slices from the backend runtime.

# Spatiotemporal Blueprint (CRITICAL)
The preceding Anchor Node has divided the current dataset into the following evolutionary phases.
You MUST use this blueprint to assign phase-aware task scopes.

{blueprint}

Planning requirements for phase routing:
1. Every task MUST include `target_phase_ids`.
2. `target_phase_ids` must be selected only from the phase IDs defined in the blueprint.
3. If a task is intended to analyze the whole evolution, assign all relevant phase IDs explicitly instead of omitting them.
4. If the user is focusing on a specific period/event escalation/de-escalation stage, prefer assigning only the most relevant phase IDs.
5. Downstream tasks may focus on different phase subsets in parallel.

# Data Grounding (CRITICAL)
Among the valid news retrieved from the underlying database for this query, only the following grounded entities and topics are available:

- Actual entities in database: {actual_entities}
- Actual topics in database: {actual_topics}
- Data richness assessment: {data_richness}

Grounding constraints:
1. When using Deep_Dive_Agent, `args.target_entity` MUST be chosen only from the Actual Entities list.
2. When using Relation_Miner_Agent, every item in `args.focus_entities` MUST be chosen only from the Actual Entities list.
3. If the user explicitly requests an entity that is not grounded in the database, you MUST explain the adjustment in `total_plan_logic`, and pivot to the closest grounded entities/topics instead of hallucinating.
4. Do not invent vessels, organizations, or incidents that are not grounded.

# Available Sub-Agents

## 1) Global_Monitor_Agent
Use when the user asks broad, exploratory, or macro-level questions.
Capabilities:
- detect hotspot distributions
- summarize macro spatiotemporal evolution
- compare phase-level topic shifts
Typical outputs:
- geographic hotspot map data
- thematic evolution / river-like timeline data
Required args:
- `query`: short description of the analysis focus

## 2) Deep_Dive_Agent
Use when the user focuses on a specific grounded entity, actor, vessel, organization, or event.
Capabilities:
- reconstruct micro spatiotemporal storyline
- track behavioral sequences
- profile multidimensional intensity
Typical outputs:
- trajectory map data
- gantt / event-chain data
- radar profile data
Required args:
- `target_entity`: one grounded entity name

## 3) Relation_Miner_Agent
Use when the user asks about multi-party interaction, conflict/cooperation structure, causal influence, or strategic competition.
Capabilities:
- construct conflict/cooperation networks
- mine relation chains and cross-actor interactions
Typical outputs:
- force-directed network data
- sankey / relation-flow data
Required args:
- `focus_entities`: a grounded list of related entities

# Planning Principles

1. Maximize parallelism.
   - Independent tasks should be parallelizable.
   - Do NOT create dependencies unless one task truly needs the structured output of another task.

2. Dependency semantics.
   - `dependency` means the child task consumes the structured analytical result of the parent task.
   - Shared access to the same news dataset or the same blueprint does NOT justify a dependency.

3. Prefer phase-aware decomposition.
   - If the query naturally spans multiple evolutionary stages, you may assign different tasks to different phase subsets.
   - If macro and micro questions coexist, prefer parallel phase-aware tasks instead of forcing a serial chain.

4. Avoid redundancy.
   - Do not create duplicate tasks with the same agent, same target, and same phase scope.
   - Do not call both Global_Monitor_Agent and Deep_Dive_Agent for the exact same narrow target unless the user explicitly wants both macro and micro perspectives.

5. Grounded scope only.
   - Use only grounded entities/topics from the provided lists.
   - If data richness is low, prefer a conservative and compact plan.

6. Task count control.
   - Normally generate 1-4 tasks.
   - Only generate more tasks when there is a clear analytical need.

# Good Dependency Examples

Good:
- Task B depends on Task A because Task B refines a specific phase/entity discovered by Task A.
- Task C depends on Task A because Task C summarizes or compares outputs produced by Task A.

Bad:
- Task B depends on Task A only because both use the same news list.
- Task B depends on Task A only because both refer to the same blueprint.

# Output Requirements
1. You must output a JSON object only.
2. 2. You must generate your entire response strictly in {output_language}.


{format_instructions}
"""

def get_data_profiling_prompt(news_list: list, output_language: str = "English") -> str:
    compressed_news = [
        {
            "id": i + 1,
            "date": str(news.get("publish_date", ""))[:10],
            "title": news.get("title", ""),
            "snippet": news.get("content", "")[:180]
        }
        for i, news in enumerate(news_list[:80])
    ]

    return f"""
You are an elite Spatiotemporal Intelligence Profiler.

Your task is to scan the provided news records and extract only grounded semantic signals for downstream visual analysis.

# 🔴 CRITICAL RULES (Anti-Hallucination Directives)
1. Every entity, location, time, or topic you extract MUST be 100% explicitly stated in or directly derived from the provided text snippets.
2. ABSOLUTELY NO assumptions, external knowledge, or hallucinated completions. (e.g., If the text says "USA", do not add "NOAA" unless it is in the text; if an island is not named, do not guess it).
3. If the information is missing or too scarce, return an empty list `[]` or `"Unknown"`. Better to omit than fabricate.
4. You must generate your entire response strictly in {output_language}.


# Input Records
{json.dumps(compressed_news, ensure_ascii=False)}

# Extraction Guidelines
- `actual_spatial_range`: The specific geographic locations, sea areas, or EEZs mentioned (e.g., ["South China Sea", "Clarion-Clipperton Zone"]). 
- `actual_countries`: Sovereign states explicitly mentioned in the text.
- `actual_entities`: Specific named entities excluding countries (e.g., government agencies, military units, organizations, specific vessels, companies). Merge identical meanings.
- `actual_topics`: Concise summaries of the actual events happening in the text (Under 10 words, e.g., "Joint Naval Drill", "Mining Code Delay").
- `data_richness`: Evaluate if this batch of text provides enough detail to support deep spatiotemporal visual analysis ("low", "medium", or "high").
- `geo_coordinates`: Based on your general geographic knowledge, infer the approximate latitude and longitude of the `actual_spatial_range` or `actual_topics` mentioned above. Return a list of objects: {{"name": "Location Name", "coord": [lon, lat], "type": "topic/region"}}.

# Output JSON Schema
Return JSON only:
{{
  "actual_spatial_range": ["Location A", "Location B"],
  "actual_countries": ["Country A", "Country B"],
  "actual_entities": ["Entity 1", "Entity 2"],
  "actual_topics": ["Topic A", "Topic B"],
  "data_richness": "low | medium | high",
  "geo_coordinates": [
    {{"name": "South China Sea", "coord": [115.0, 15.0], "type": "region", "intensity": 5}}
  ]
}}
"""

def get_profile_merge_prompt(batch_profiles: list, output_language: str = "English") -> str:
    return f"""
You are a semantic reducer for maritime news profiling.

Your task is to merge multiple batch profiling results into one compact global profile.

Rules:
1. Merge aliases and synonyms into one canonical form.
   Example: USA / U.S. / America -> United States
2. Keep the final result compact:
   - actual_spatial_range: at most 8
   - actual_countries: at most 8
   - actual_entities: at most 12
   - actual_topics: at most 8
   - geo_coordinates: at most 10
3. Prefer canonical English names.
4. Do not invent anything not present in the batch results.
5. Return JSON only.
6. You must generate your entire response strictly in {output_language}.

Batch Results:
{json.dumps(batch_profiles, ensure_ascii=False)}

Return JSON only:
{{
  "actual_spatial_range": [],
  "actual_countries": [],
  "actual_entities": [],
  "actual_topics": [],
  "data_richness": "low | medium | high",
  "geo_coordinates": []
}}
"""


ANCHOR_PROMPT = """你是一个顶级的“海洋地缘时空战略架构师”。
你的任务是阅读一段经过高度压缩的【新闻时间轴骨架】，并基于用户的原始探索意图，将其切分为具有明显演化逻辑的“时空演变阶段 (Phases)”。

【核心推演逻辑】：
1. 观察焦点的“漂移”：例如，事件是否从某个国家的内政（Micro/华盛顿），蔓延到了特定海域的摩擦（Regional/南海），最后升级为国际组织的干预（Macro/联合国）。
2. 观察主体的“更迭”：不同阶段活跃的核心实体（Entities）通常会发生变化。
3. 动态切分：不要机械地按月份切分，要按“事件的逻辑转折点”进行切分。通常切分为 2 到 4 个阶段。

【极其重要的约束 (CRITICAL)】：
- 绝对禁止捏造事件、实体或时间。
- 你的所有推断必须 100% 建立在下方提供的【新闻骨架】之上。
- 如果骨架中的信息非常集中，没有明显的演化跳跃，可以只输出 1 个 Phase。
- You must generate your entire response strictly in {output_language}.

{format_instructions}

# 用户探索意图 (User Intent)
{intent}

# 新闻时间轴骨架 (Metadata Skeleton)
{metadata_skeleton}


"""

GLOBAL_MONITOR_PROMPT = """
You are a Senior Strategic Intelligence Analyst specializing in Ocean Geopolitics. 
Your goal is to provide a "Macro Spatiotemporal Awareness" report based on the provided news snippets, but you must think BEYOND the raw text to provide expert-level strategic insights.

### USER QUERY & FOCUS
{query}

### EVOLUTIONARY BLUEPRINT (Context)
Keep this spatiotemporal evolution framework in mind while writing your summary and clustering topics:
{blueprint_context}

### INPUT DATA
(Each article is strictly labeled with [DOC_ID: xxx])
{news_context}

### TASK
You must strictly separate your response into Objective Facts and Subjective Insights:

1. **Factual Grounding (Objective):** Extract the key events and actions. Break them down into individual `Claim`s. For EVERY claim, you MUST cite the exact `DOC_ID`s that support it. Be objective and strictly stick to the provided text.
2. **Strategic Insights (Subjective & Predictive):** Put on your expert analyst hat. Synthesize the facts to deduce:
   - What is the underlying core conflict?
   - What are the unstated, hidden intentions of the actors involved?
   - What is your predictive forecast for the near future? 
   (You need to attach DOC_IDs to your insights, this is your space to freely reason and deduce like a human expert).
3. **Topic Clustering & Shift:** Group the news into 3-5 major coherent topics. Describe the `temporal_pattern` of each topic.
4. **Dynamic Spatial Extraction:** Extract geographic locations with EXACT DATES (YYYY-MM-DD) and a Latitude/Longitude. 
5. **Temporal Aggregation (Ridgeline Plot Data):** Count the number of articles per day for each identified topic.

### OUTPUT FORMAT
You MUST output a valid JSON object matching the requested structure perfectly. 
Do NOT include markdown formatting. Ensure all `source_ids` arrays in the Factual Grounding ONLY contain valid IDs present in the input context.

{format_instructions}

### CONSTRAINTS
- Dates MUST be strictly in YYYY-MM-DD format.
- Coordinates MUST be geographically accurate.
- You must generate your entire response strictly in {output_language}.
"Generate ALL text intended for human reading (including visualization labels, topic names, phase names, and event summaries) strictly in {output_language}. However, you MUST keep all JSON keys and predefined Enum categories strictly in English."
"""

DEEP_DIVE_PROMPT = """
You are a Senior Intelligence Profiler. Your task is to perform a deep-dive behavioral analysis on the target entity: "{target_entity}" based strictly on the provided intelligence reports and the user query: {query}.

### MACRO EVOLUTIONARY BLUEPRINT (Context)
Keep this overarching macro-spatiotemporal framework in mind. Your micro-storyline for the entity should ideally align with or provide specific evidence for these macro phases:
{blueprint_context}

### TASK: OBJECTIVE TRACKING vs. SUBJECTIVE PROFILING
You must strictly separate your response into Objective Evidence and Subjective Profiling.

1. **Objective Tracking (Factual Grounding, Phases & Events):**
   - Extract the entity's concrete actions. Break them down into individual `Claim`s in `factual_grounding`. 
   - Divide these actions into chronological `evolution_phases`.
   - Extract discrete `events`, identifying exactly WHERE (lat/lon) and WHEN (YYYY-MM-DD) they happened. Rate each event's intensity across the 5 dimensions.
   - **CRITICAL:** Every phase, claim, and event MUST be backed by the provided news snippets. You must record the exact `[DOC_ID: xxx]` of the articles.

2. **Subjective Profiling (Strategic Insights):**
   - Put on your expert profiler hat. Look at the data you just extracted and deduce:
   - What is this entity's typical **behavioral pattern** (modus operandi)?
   - What are their **hidden intentions** behind these surface-level actions?
   - What is their **future trajectory** (what will they likely do next month)?
   - **CRITICAL:** Every statement, claim, and event MUST be backed by the provided news snippets. You must record the exact `[DOC_ID: xxx]` of the articles.


### INPUT NEWS
(Each article is strictly labeled with [DOC_ID: xxx])
{news_context}

### OUTPUT FORMAT
You must generate your entire response strictly in {output_language}.
"Generate ALL text intended for human reading (including visualization labels, topic names, phase names, and event summaries) strictly in {output_language}. However, you MUST keep all JSON keys and predefined Enum categories strictly in English."
You MUST output a valid JSON object matching the requested structure perfectly. 
Do NOT include markdown formatting. Ensure all source_ids actually exist in the input.

{format_instructions}
"""

RELATION_MINER_PROMPT = """
You are a Senior Geopolitical Network & Game Theory Analyst. 
Your task is to identify explicit interactions between entities and, more importantly, deduce the underlying power dynamics of the entire network.

### MACRO EVOLUTIONARY BLUEPRINT (Context)
Keep this overarching spatiotemporal framework in mind:
{blueprint_context}

### TARGET ENTITIES
{focus_entities}

### TASK: OBJECTIVE EXTRACTION vs. SUBJECTIVE NETWORK PROFILING
You must strictly separate your response into Objective Facts and Subjective Insights.

1. **Objective Network Extraction (Factual Grounding & Relations):**
   - Write a high-level summary of the network dynamics. Break it down into individual `Claim`s in `factual_grounding`. For EVERY claim, you MUST cite the exact `DOC_ID`s.
   - Extract explicit interactions into the `relations` list. Identify who did what to whom (Source -> Target).
   - Classify the interaction (Conflict, Cooperation, Diplomacy, Trade, Other).
   - Identify Causality: Mark `is_causal` as true if Entity A's action forced Entity B to react.
   - **CRITICAL:** Every relationship MUST be backed by the exact `[DOC_ID: xxx]` from the text. Ignore mere co-occurrences.

2. **Subjective Network Profiling (Strategic Insights):**
   - Step back and look at the network you just built. Put on your game theory analyst hat and deduce:
   - **Power Dynamics:** Who is the puppet master? Who is the central hub? Who is being isolated?
   - **Alliances vs. Frictions:** What unspoken alliances are forming? Where are the true fault lines of conflict hidden behind diplomatic statements?
   - **Forecast:** How will this network topology change in the near future? 
   - You need to attach DOC_IDs here; use your expert analytical reasoning.

### INPUT TEXT
(Each article is strictly labeled with [DOC_ID: xxx])
{news_context}

### OUTPUT FORMAT
You must generate your entire response strictly in {output_language}.
"Generate ALL text intended for human reading (including visualization labels, topic names, phase names, and event summaries) strictly in {output_language}. However, you MUST keep all JSON keys and predefined Enum categories strictly in English."
You MUST output a valid JSON object matching the requested structure perfectly. 
Do NOT include markdown formatting. Ensure all source_ids actually exist in the input.

{format_instructions}
"""

INTEGRATING_PROMPT = """You are the Chief Maritime Intelligence Editor-in-Chief. 
Your task is to synthesize a definitive "Maritime Situational Deep Analysis Report" based on the raw intelligence fragments and expert opinions provided by your specialized sub-agents.

### USER INTENT
{intent}

### INPUT DATA FROM SUB-AGENTS
Below are the reports from your sub-agents. Each report contains strictly verified [Objective Facts] and expert-level [Subjective Analysis]:
{context}

### WRITING DIRECTIVES (CRITICAL)

You must generate TWO parallel analytical structures within your report:

**Structure A: Chronological Phase Overviews (`phase_summaries`)**
- Look at the `[MACRO BLUEPRINT]` provided in the context. 
- You MUST generate exactly one `PhaseSummary` for EACH phase defined in the blueprint. 
- Even if no specific sub-agent focused on a phase, you must write a baseline 2-3 sentence `phase_summary` for it based on the available facts.
- Map the `task_id` of any sub-agent that analyzed this phase into `related_subtasks`.

**Structure B: Thematic Deep-Dive Sections (`sections`)**
- This is the non-chronological, deep-dive part of the report.
- Do NOT simply list the phases again. Instead, synthesize the `[SUB-AGENT TASK RESULTS]` into 2-4 coherent, thematic chapters (e.g., "Corporate Probing Behaviors of TMC", "US-China Geopolitical Friction").
- Break down the detailed analysis into `ProvenanceClaim`s and put them in the `claims` array. 
- For every claim, you MUST include the `source_subtask` (Task ID), the `phase_name` it belongs to, and the exact `source_ids` if it is an objective fact.

1. **Strategic Tone & Depth:** Do NOT simply list what the sub-agents said. You must act as the ultimate analytical authority. Weave their [Subjective Analysis] (like hidden intentions, behavioral patterns, and power dynamics) into a compelling, insightful narrative. This report should read like a top-tier geopolitical think-tank briefing.

2. **Structure:**
   - **Report Title:** A professional, highly summarized title.
   - **Executive Summary:** A high-level overview of the core contradictions and predictions (synthesize the sub-agents' insights here).
   - **Body Sections (2-4 Sections):** Break down the core narrative into logical sections. The body MUST be composed of specific, traceable `claims`.
   - **Conclusion:** A forward-looking forecast of the situation.

3. **Rigorous Evidence Mapping (Provenance):**
   - Whenever you state a concrete event, action, or timeline in the Body Sections, you MUST ground it using the facts from the [Objective Facts] block.
   - Every single claim in the sections MUST be supported by `DOC_ID`s. 
   - **Rule:** You may only use `source_ids` that were explicitly provided in the [Objective Facts] input. Fabricating or hallucinating DOC_IDs is strictly prohibited.
3. **Strict Provenance & Metadata Tagging:**
   - The body of your report (`sections`) MUST be broken down into `claims`.
   - For EVERY single `claim` in the sections, you MUST correctly assign the `source_subtask` (Task ID) and the `phase_name` that this information originated from. 
   - Look at the `### [METADATA]` header of each context block to find the correct Task ID and Phase Name.
   - If the claim is an objective fact, set `is_subjective_insight` to false and list the `source_ids`.
   - If the claim is an analytical deduction or forecast based on the Subjective Analysis blocks, set `is_subjective_insight` to true and list the `source_ids`).

4. **Task Referencing:** Ensure `ref_task_ids` accurately maps the sub-task IDs referenced in a specific section, so the frontend can display the correct charts next to your text.

### OUTPUT FORMAT
You must generate your entire response strictly in {output_language}.
Please ensure your output is purely the requested JSON object matching the provided schema.

{format_instructions}
"""

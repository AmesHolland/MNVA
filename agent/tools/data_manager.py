import hashlib
import json
import os
import uuid
import sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# =========== 处理项目根目录路径，确保能找到 agent 模块 ===========
current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(current_dir, "../../"))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

try:
    from agent.config.llm_config import llm_qw_quick 
except ImportError as e:
    print(f"导包失败，请检查 PYTHONPATH。错误详情: {e}")
    llm_qw_quick = None 

# 配置数据仓库目录
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data_warehouse')
REGISTRY_FILE = os.path.join(DATA_DIR, 'registry.json')

def init_warehouse():
    """初始化数据仓库目录和注册表"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    if not os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE, 'w', encoding='utf-8') as f:
            json.dump([], f)

def load_registry():
    init_warehouse()
    with open(REGISTRY_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_to_registry(dataset_info):
    registry = load_registry()
    registry.append(dataset_info)
    with open(REGISTRY_FILE, 'w', encoding='utf-8') as f:
        json.dump(registry, f, ensure_ascii=False, indent=4)

# --- 阶段一：LLM 骨架抽取 ---

def extract_skeleton_with_llm(content):
    if not llm_qw_quick or not content or content == "N/A":
        return {}
    
    prompt = ChatPromptTemplate.from_template("""
    你是一个专业的深海态势分析专家。请从以下新闻正文中抽取结构化骨架。
    必须严格返回 JSON 格式。
    
    JSON结构要求：
    {{
        "summary": "核心摘要",
        "entities": [{{ "name": "实体名", "type": "类型" }}],
        "locations": [{{ "name": "地名", "lat": null, "lng": null }}],
        "temporal_refs": [],
        "keywords": [],
        "event_type": "分类"
    }}
    
    新闻正文：{content}
    """)
    
    # 重点 1：先不用 JsonOutputParser 强压，拿原始字符串
    chain = prompt | llm_qw_quick 
    
    try:
        response = chain.invoke({"content": str(content)[:3000]})
        # 如果返回的是消息对象，取 content
        raw_text = response.content if hasattr(response, 'content') else str(response)
        
        # 重点 2：清理 Markdown 标签
        raw_text = raw_text.replace("```json", "").replace("```", "").strip()
        
        # 重点 3：使用更加宽容的 json.loads
        import json
        return json.loads(raw_text)
        
    except Exception as e:
        # 如果还是失败，尝试用你的 base 里的 safe_parse_json (如果有的话)
        from agent.tools.base import safe_parse_json
        print(f"⚠️ 结构化解析尝试修复...")
        return safe_parse_json(raw_text) if 'raw_text' in locals() else {}

# --- 阶段二：全局实体归一化逻辑 ---

def generate_canonical_mapping(all_entities):
    """
    调用LLM将收集到的所有异名实体进行对齐（如：美国/USA -> United States）
    """
    if not all_entities or not llm_qw_quick:
        return {}

    # 去重并过滤掉太短或无效的词
    unique_entities = sorted(list(set([str(e) for e in all_entities if len(str(e)) > 1])))
    if not unique_entities:
        return {}
    
    prompt = ChatPromptTemplate.from_template("""
    你是一个深海战略情报专家。以下是从一批新闻中提取出的原始实体列表。
    请识别出指代同一个对象的不同表述，并提供一个归一化映射表。
    
    归一化原则：
    1. 缩写与全称对齐（如：NOAA -> National Oceanic and Atmospheric Administration）。
    2. 中英文对齐（如：美国 -> United States）。
    3. 别名对齐（如：华盛顿方面 -> United States）。
    4. 标准名请优先使用【正式英文缩写】或【官方全称】。
    
    请严格返回JSON格式：{{"原始词": "标准词", "原始词2": "标准词"}}
    
    原始实体列表：
    {entity_list}
    """)
    
    chain = prompt | llm_qw_quick | JsonOutputParser()
    try:
        print(f"🧠 正在启动全局实体对齐 (Entity Alignment)，处理 {len(unique_entities)} 个潜在实体...")
        return chain.invoke({"entity_list": json.dumps(unique_entities, ensure_ascii=False)})
    except Exception as e:
        print(f"⚠️ 实体归一化任务调用失败: {e}")
        return {}

# --- 阶段三：核心加工流程 ---

def process_uploaded_file(file_path, original_filename, dataset_name=None):
    """核心加工引擎：读取 -> 抽取骨架 -> 全局归一化 -> 持久化"""
    init_warehouse()
    ext = original_filename.split('.')[-1].lower()
    try:
        if ext == 'csv': df = pd.read_csv(file_path)
        elif ext in ['xls', 'xlsx']: df = pd.read_excel(file_path)
        elif ext == 'json': df = pd.read_json(file_path)
        else: return {"error": f"不支持格式: {ext}"}
    except Exception as e:
        return {"error": f"文件读取失败: {str(e)}"}

    # 字段映射与清洗
    column_mapping = {
        '标题': 'title', 'Title': 'title',
        '内容': 'content', '正文': 'content', 'Content': 'content',
        '日期': 'publish_date', '发布时间': 'publish_date', 'Date': 'publish_date',
        '来源': 'source', 'Source': 'source',
        '链接': 'url', 'Url': 'url'
    }
    df.rename(columns=column_mapping, inplace=True, errors='ignore')
    for col in ['title', 'content', 'publish_date', 'source']:
        if col not in df.columns: df[col] = "N/A"
    df = df.fillna("N/A")

    records = df.to_dict(orient='records')
    
    # 1. 并行抽取初始骨架
    print(f"🚀 正在启动 LLM 骨架抽取流水线，共 {len(records)} 条新闻...")
    with ThreadPoolExecutor(max_workers=5) as executor:
        contents = [r.get('content', '') for r in records]
        skeletons = list(executor.map(extract_skeleton_with_llm, contents))

    # 2. 收集所有实体进行归一化处理
    all_raw_entities = []
    for sk in skeletons:
        if sk and 'entities' in sk:
            all_raw_entities.extend([e['name'] for e in sk['entities'] if 'name' in e])
    
    mapping_table = generate_canonical_mapping(all_raw_entities)

    # 3. 注入骨架并应用归一化映射
    processed_data = []
    for i, row in enumerate(records):
        row['DOC_ID'] = "news_" + hashlib.md5(f"{row['title']}_{i}".encode()).hexdigest()[:8]
        row['publish_date'] = str(row['publish_date']).split(' ')[0] if row['publish_date'] != "N/A" else ""
        
        # 处理当前条目的骨架
        current_sk = skeletons[i]
        if current_sk and 'entities' in current_sk:
            for ent in current_sk['entities']:
                original_name = ent.get('name')
                # 如果在映射表中找到了标准名称，则替换它
                if original_name in mapping_table:
                    ent['name'] = mapping_table[original_name]
        
        row['evidence_skeleton'] = current_sk
        processed_data.append(row)

    # 4. 持久化到数据仓库
    dataset_id = f"ds_{uuid.uuid4().hex[:8]}"
    save_path = os.path.join(DATA_DIR, f"{dataset_id}.json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)

    # 5. 注册数据集信息
    dataset_info = {
        "dataset_id": dataset_id,
        "dataset_name": dataset_name or original_filename.rsplit('.', 1)[0],
        "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "row_count": len(processed_data),
        "file_path": save_path,
        "has_normalized": True
    }
    save_to_registry(dataset_info)
    

    return {"success": True, "dataset": dataset_info}

def load_dataset(dataset_id=None):
    """加载指定数据集的内容"""
    if not dataset_id:
        registry = load_registry()
        if not registry: return []
        registry.sort(key=lambda x: x['upload_time'], reverse=True)
        dataset_id = registry[0]['dataset_id']
    
    filepath = os.path.join(DATA_DIR, f"{dataset_id}.json")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"读取数据集失败: {e}")
        return []


def get_dataset_entities(dataset_id=None):
    """从标准数据集中提取所有已归一化的实体名称"""
    data = load_dataset(dataset_id)
    entities = set()
    for item in data:
        for ent in item.get('evidence_skeleton', {}).get('entities', []):
            entities.add(ent['name'])
    return sorted(list(entities))


def retrieve_news(time_range_start=None, time_range_end=None, entities=None, locations=None, event_types=None, keywords=None, top_k=None, dataset_id=None):
    """
    最稳兼容版：保持原有参数顺序不变，新增参数放在最后
    """
    # 1. 确定最终要使用的数据集 ID
    target_ds_id = dataset_id
    
    if not target_ds_id:
        # 如果调用者没传 dataset_id（比如旧的 nodes.py），则自动去注册表拿最新的
        registry = load_registry()
        if registry:
            target_ds_id = registry[-1]["dataset_id"]
            # print(f"💡 [兼容模式] 自动挂载最新数据集: {target_ds_id}")
        else:
            target_ds_id = "default_pool"

    # 2. 加载数据
    all_data = load_dataset(target_ds_id)
    if not all_data:
        return []

    # 3. 解析时间 (此时 time_range_start 对应的就是旧代码传进来的 date_from)
    start_dt, end_dt = None, None
    try:
        if time_range_start and isinstance(time_range_start, str):
            start_dt = datetime.strptime(time_range_start.split(' ')[0], "%Y-%m-%d")
        if time_range_end and isinstance(time_range_end, str):
            end_dt = datetime.strptime(time_range_end.split(' ')[0], "%Y-%m-%d")
    except Exception as e:
        # print(f"⏰ 时间解析跳过: {e}")
        pass

    scored_results = []

    for item in all_data:
        skeleton = item.get("evidence_skeleton", {})
        if not skeleton: continue
        
        # --- 过滤与评分逻辑 ---
        # A. 时间硬过滤
        pub_date_str = item.get("publish_date", "")
        if start_dt or end_dt:
            try:
                dt = datetime.strptime(pub_date_str.split(' ')[0], "%Y-%m-%d")
                if start_dt and dt < start_dt: continue
                if end_dt and dt > end_dt: continue
            except: continue

        score = 0
        matched_dims = []

        # B. 实体匹配 (增强鲁棒性，防止 skeleton 数据格式意外)
        if entities and isinstance(skeleton.get('entities'), list):
            item_entities = [e.get('name', '').lower() for e in skeleton['entities'] if isinstance(e, dict)]
            hits = sum(1 for e in entities if str(e).lower() in item_entities)
            if hits > 0:
                score += hits * 2
                matched_dims.append("entity")

        # C. 地点匹配
        if locations and isinstance(skeleton.get('locations'), list):
            item_locs = [l.get('name', '').lower() for l in skeleton['locations'] if isinstance(l, dict)]
            hits = sum(1 for loc in locations if str(loc).lower() in item_locs)
            if hits > 0:
                score += hits * 2
                matched_dims.append("location")

        # D. 事件类型与关键词
        if event_types and skeleton.get("event_type") in event_types:
            score += 1
            matched_dims.append("event_type")

        if keywords:
            search_text = (str(skeleton.get("summary", "")) + " " + " ".join(skeleton.get("keywords", []))).lower()
            hits = sum(1 for kw in keywords if str(kw).lower() in search_text)
            if hits > 0:
                score += hits
                matched_dims.append("keyword")

        # 结果装载
        has_query_params = any([entities, locations, event_types, keywords])
        if not has_query_params or score > 0:
            item["_retrieval_score"] = score
            item["_matched_dimensions"] = matched_dims
            scored_results.append(item)

    # 排序：分数高者在前，时间新者在前
    scored_results.sort(key=lambda x: (x.get("_retrieval_score", 0), x.get("publish_date", "")), reverse=True)
    
    return scored_results[:top_k] if top_k else scored_results

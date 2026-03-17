import hashlib
import json
import os
import uuid
from datetime import datetime

import pandas as pd

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
    """读取当前系统内可用的所有数据集"""
    init_warehouse()
    with open(REGISTRY_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_to_registry(dataset_info):
    """将新数据集追加到注册表中"""
    registry = load_registry()
    registry.append(dataset_info)
    with open(REGISTRY_FILE, 'w', encoding='utf-8') as f:
        json.dump(registry, f, ensure_ascii=False, indent=4)


def generate_doc_id(row, index):
    """基于行数据生成稳定的 DOC_ID"""
    title = str(row.get('title', ''))
    date = str(row.get('publish_date', ''))
    content_snippet = str(row.get('content', ''))[:20]
    unique_str = f"{title}_{date}_{content_snippet}_{index}"
    return "news_" + hashlib.md5(unique_str.encode('utf-8')).hexdigest()[:8]


def process_uploaded_file(file_path, original_filename, dataset_name=None):
    """
    核心转换引擎：将用户上传的五花八门的文件，洗成标准 JSON。
    :param file_path: 临时保存的上传文件路径
    :param original_filename: 原始文件名
    :param dataset_name: 用户自定义的数据集名称，默认用文件名
    """
    init_warehouse()

    # 1. 智能读取：根据后缀名调用 pandas 对应的方法
    ext = original_filename.split('.')[-1].lower()
    try:
        if ext == 'csv':
            df = pd.read_csv(file_path)
        elif ext in ['xls', 'xlsx']:
            df = pd.read_excel(file_path)
        elif ext == 'json':
            df = pd.read_json(file_path)
        else:
            return {"error": f"不支持的文件格式: {ext}"}
    except Exception as e:
        return {"error": f"文件解析失败: {str(e)}"}

    # 2. 字段映射与容错 (把中文表头洗成系统认的英文 key)
    # 你可以根据实际情况扩展这个字典
    column_mapping = {
        '标题': 'title', 'Title': 'title',
        '内容': 'content', '正文': 'content', 'Content': 'content', 'text': 'content',
        '日期': 'publish_date', '发布时间': 'publish_date', 'Date': 'publish_date', 'date': 'publish_date',
        '来源': 'source', 'Source': 'source',
        '链接': 'url', 'Url': 'url'
    }
    df.rename(columns=column_mapping, inplace=True)

    # 确保必填字段存在，防止下游图表/Agent 报错
    for required_col in ['title', 'content', 'publish_date', 'source']:
        if required_col not in df.columns:
            df[required_col] = "N/A"  # 缺失字段补全

    # 处理 NaN 和 NaT 等 pandas 特殊空值
    df = df.fillna("N/A")

    # 3. 转换为字典列表，并注入 DOC_ID
    records = df.to_dict(orient='records')
    processed_data = []

    for i, row in enumerate(records):
        # 如果原始数据里没有 DOC_ID，我们就给它算一个
        if 'DOC_ID' not in row or row['DOC_ID'] == "N/A":
            row['DOC_ID'] = generate_doc_id(row, i)

        # 将 publish_date 强转为字符串，防止 datetime 对象 JSON 序列化失败
        row['publish_date'] = str(row['publish_date']).split(' ')[0] if row['publish_date'] != "N/A" else ""
        processed_data.append(row)

    # 4. 生成唯一的数据集 ID，并持久化到文件系统
    dataset_id = f"ds_{uuid.uuid4().hex[:8]}"
    dataset_name = dataset_name or original_filename.rsplit('.', 1)[0]

    save_path = os.path.join(DATA_DIR, f"{dataset_id}.json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)

    # 5. 更新系统注册表
    dataset_info = {
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "original_filename": original_filename,
        "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "row_count": len(processed_data),
        "file_path": save_path
    }
    save_to_registry(dataset_info)

    # 清理掉原始的临时文件（可选，保持服务器干净）
    try:
        os.remove(file_path)
    except:
        pass

    return {
        "success": True,
        "message": f"成功导入 {len(processed_data)} 条新闻",
        "dataset": dataset_info
    }


def load_dataset(dataset_id=None):
    """
    根据 dataset_id 动态加载标准化的 JSON 数据
    """
    if not dataset_id:
        # 如果没有传 dataset_id，为了不让系统崩溃，我们可以默认读取注册表里的最新一个数据集
        registry = load_registry()
        if not registry:
            print("⚠️ 数据仓库为空，请先上传数据！")
            return []
        # 默认取最新上传的
        registry.sort(key=lambda x: x['upload_time'], reverse=True)
        dataset_id = registry[0]['dataset_id']

    filepath = os.path.join(DATA_DIR, f"{dataset_id}.json")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ 找不到数据集文件: {filepath}")
        return []
    except Exception as e:
        print(f"❌ 加载数据集失败: {e}")
        return []


def retrieve_news(start_date=None, end_date=None, keywords=None, dataset_id=None):
    """
    执行带时间和关键词约束的本地检索，支持动态数据集
    :param dataset_id: 前端传来的目标数据集ID (例如 "ds_a1b2c3")
    """
    # 🌟 核心变化：动态加载数据
    all_data = load_dataset(dataset_id)
    filtered_data = []

    start_dt = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
    end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None

    for item in all_data:
        # 1. 时间范围约束
        item_date_str = item.get("publish_date", "")
        if start_dt or end_dt:
            if not item_date_str or item_date_str == "N/A":
                continue
            try:
                item_dt = datetime.strptime(item_date_str, "%Y-%m-%d")
                if start_dt and item_dt < start_dt:
                    continue
                if end_dt and item_dt > end_dt:
                    continue
            except ValueError:
                continue

        # 2. 关键词匹配约束
        if keywords and isinstance(keywords, list) and len(keywords) > 0:
            search_pool = (
                    str(item.get("title", "")) +
                    str(item.get("content", "")) +
                    str(item.get("source", "")) +
                    " ".join(item.get("keywords", []))
            ).lower()

            match = any(str(kw).lower() in search_pool for kw in keywords)
            if not match:
                continue

        filtered_data.append(item)

    # 🌟 新增：按publish_date排序
    def get_sort_key(item):
        """排序辅助函数：处理日期转换异常，无效日期返回极小值（排在最前）"""
        date_str = item.get("publish_date", "")
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except (ValueError, TypeError):
            # 无效日期（空/格式错误/N/A）返回极小值，会排在最前面
            return datetime.min

    # 按publish_date升序排序（最新的在后面），如需降序可加 reverse=True
    sorted_data = sorted(filtered_data, key=get_sort_key)

    return sorted_data
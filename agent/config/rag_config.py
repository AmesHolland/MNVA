

class RAGConfig:
    """RAG 系统配置"""
    # 模型配置（使用全局 model）
    temperature: float = 0.1

    # 分块配置
    chunk_size: int = 500
    chunk_overlap: int = 100

    # 检索配置
    top_k: int = 5
    search_type: str = "similarity"  # similarity, mmr

    # 生成配置
    max_tokens: int = 1000

    # 新闻ID前缀（最终ID格式：news_001、news_002...）
    news_id_prefix = "news_"

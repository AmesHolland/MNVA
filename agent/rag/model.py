from langchain.chat_models import init_chat_model
from agent.config.rag_config import SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL, SILICONFLOW_EMBEDDING_URL, SILICONFLOW_CHAT_MODEL, SILICONFLOW_EMBEDDING_MODEL
from langchain_openai import OpenAIEmbeddings, ChatOpenAI


def get_embeddings():
    """
    获取硅基流动的嵌入模型实例（替换你原有返回None的版本）
    返沪的实例兼容LangChain标准接口，可直接传给Weaviate入库的ingest方法
    """
    # 校验API Key
    if not SILICONFLOW_API_KEY:
        raise ValueError("请设置SILICONFLOW_API_KEY环境变量（硅基流动平台获取）")
    
    return OpenAIEmbeddings(
        api_key=SILICONFLOW_API_KEY,  # 硅基流动API Key
        base_url=SILICONFLOW_EMBEDDING_URL,  # 补全嵌入接口完整路径
        model=SILICONFLOW_EMBEDDING_MODEL,  # 硅基流动嵌入模型名
        # normalize_embeddings=True,  # 归一化向量（RAG检索推荐开启，提升相似度计算精度）
        max_retries=3,  # 请求失败重试次数
        timeout=30  # 超时时间
    )


def get_rerankings():
    """
    获取硅基流动的嵌入模型实例（替换你原有返回None的版本）
    返沪的实例兼容LangChain标准接口，可直接传给Weaviate入库的ingest方法
    """
    # 校验API Key
    if not SILICONFLOW_API_KEY:
        raise ValueError("请设置SILICONFLOW_API_KEY环境变量（硅基流动平台获取）")

    return OpenAIEmbeddings(
        api_key=SILICONFLOW_API_KEY,  # 硅基流动API Key
        base_url=SILICONFLOW_EMBEDDING_URL,  # 补全嵌入接口完整路径
        model=SILICONFLOW_EMBEDDING_MODEL,  # 硅基流动嵌入模型名
        # normalize_embeddings=True,  # 归一化向量（RAG检索推荐开启，提升相似度计算精度）
        max_retries=3,  # 请求失败重试次数
        timeout=30  # 超时时间
    )

def get_chat_model(
    temperature: float = 0.1,
    top_p: float = 0.9
):
    """
    获取硅基流动的对话模型实例（RAG的生成环节使用）
    :param temperature: 随机性，0~1，值越低越严谨
    :param max_tokens: 单次生成最大令牌数
    :param top_p: 采样阈值，0~1，推荐0.9
    :return: LangChain ChatModel实例，支持invoke/stream等标准方法
    """
    if not SILICONFLOW_API_KEY:
        raise ValueError("请设置SILICONFLOW_API_KEY环境变量（硅基流动平台获取）")
    
    return ChatOpenAI(
        api_key=SILICONFLOW_API_KEY,  # 硅基流动API Key
        base_url=SILICONFLOW_BASE_URL,  # 对话接口完整URL（你提供的无需补全）
        model=SILICONFLOW_CHAT_MODEL,  # 硅基流动对话模型名
        temperature=temperature,  # 随机性
        top_p=top_p,  # 采样策略
        max_retries=3,  # 重试次数
        timeout=60,  # 超时时间
        # streaming=True  # 支持流式输出（如需关闭设为False）
    )

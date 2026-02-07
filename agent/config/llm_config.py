from langchain_openai import ChatOpenAI
from zhipuai import ZhipuAI
import os

from dotenv import load_dotenv

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL")
SILICONFLOW_EMBEDDING_URL = os.getenv("SILICONFLOW_EMBEDDING_URL")
SILICONFLOW_EMBEDDING_MODEL = os.getenv("SILICONFLOW_EMBEDDING_MODEL")
SILICONFLOW_CHAT_MODEL = os.getenv("SILICONFLOW_CHAT_MODEL")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
DEEPSEEK_CHAT_MODEL = os.getenv("DEEPSEEK_CHAT_MODEL")

QWEN_API_KEY = os.getenv("QWEN_API_KEY")
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL")

llm = ChatOpenAI(
        api_key=SILICONFLOW_API_KEY,  # 硅基流动API Key
        base_url=SILICONFLOW_BASE_URL,  # 对话接口完整URL（你提供的无需补全）
        model=SILICONFLOW_CHAT_MODEL,  # 硅基流动对话模型名
        temperature=0.0,  # 随机性
        max_retries=3,  # 重试次数
        timeout=60,  # 超时时间
    )

llm_qw_quick = ChatOpenAI(
        api_key=QWEN_API_KEY,  # 硅基流动API Key
        base_url=QWEN_BASE_URL,  # 对话接口完整URL（你提供的无需补全）
        model="qwen-plus",  # 硅基流动对话模型名
        temperature=0.0,  # 随机性
        max_retries=3,  # 重试次数
        # timeout=60,  # 超时时间
    )

llm_ds = ChatOpenAI(
        api_key=DEEPSEEK_API_KEY,  # 硅基流动API Key
        base_url=DEEPSEEK_BASE_URL,  # 对话接口完整URL（你提供的无需补全）
        model=DEEPSEEK_CHAT_MODEL,  # 硅基流动对话模型名
        temperature=0.0,  # 随机性
        max_retries=3,  # 重试次数
        timeout=60,  # 超时时间
)
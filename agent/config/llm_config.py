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
SILICONFLOW_CHAT_MODEL = os.getenv("SILICONFLOW_CHAT_MODEL")
SILICONFLOW_EMBEDDING_MODEL = os.getenv("SILICONFLOW_EMBEDDING_MODEL")

llm = ChatOpenAI(
    model="qwen3-max",
    temperature=0.7,
    api_key=SILICONFLOW_API_KEY,
    base_url=SILICONFLOW_CHAT_MODEL
)

# zhipuai_client = ZhipuAI(api_key=ZHIPU_API_KEY)
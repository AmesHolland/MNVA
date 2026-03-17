from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

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

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
DEEPSEEK_CHAT_MODEL = os.getenv("DEEPSEEK_CHAT_MODEL")

QWEN_API_KEY = os.getenv("QWEN_API_KEY")
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL")

llm = ChatOpenAI(
        api_key=SILICONFLOW_API_KEY,  # 硅基流动API Key
        base_url=SILICONFLOW_BASE_URL,  # 对话接口完整URL（你提供的无需补全）
        model=SILICONFLOW_CHAT_MODEL,  # 硅基流动对话模型名
        temperature=0.0,  # 随机性
        max_retries=3,  # 重试次数
        timeout=60,  # 超时时间
    )
model_kwargs = {
                "extra_body": {
                    "enable_thinking": False  # 显式关闭思考模式
                }
            }
llm_qw_quick = ChatOpenAI(
        api_key=QWEN_API_KEY,  # 硅基流动API Key
        base_url=QWEN_BASE_URL,  # 对话接口完整URL（你提供的无需补全）
        model="qwen3.5-flash-2026-02-23",  # 硅基流动对话模型名
        temperature=0.2,  # 随机性
        max_retries=3,  # 重试次数
        # timeout=60,  # 超时时间
        model_kwargs=model_kwargs
    )

llm_qw_thinking = ChatOpenAI(
        api_key=QWEN_API_KEY,  # 硅基流动API Key
        base_url=QWEN_BASE_URL,  # 对话接口完整URL（你提供的无需补全）
        model="qwen3.5-flash-2026-02-23",  # 硅基流动对话模型名 qwen3.5-flash-2026-02-23
        temperature=0.7,  # 随机性
        max_retries=3,  # 重试次数
        # timeout=60,  # 超时时间
        model_kwargs=model_kwargs
    )

# 快速模型 - Haiku 4.5，速度快，适合简单任务
llm_claude_quick = ChatAnthropic(
    base_url=ANTHROPIC_BASE_URL,
    api_key=ANTHROPIC_API_KEY,
    model="claude-haiku-4-5-20251001",
    temperature=0.3,
    max_retries=3,
)

# 思考模型 - Sonnet 4.5，开启 extended thinking，适合复杂推理
llm_claude_thinking = ChatAnthropic(
    api_key=ANTHROPIC_API_KEY,
    base_url=ANTHROPIC_BASE_URL,
    model="claude-sonnet-4-5-20250929",
    temperature=1,  # 开启 thinking 时，temperature 必须设为 1
    max_retries=3,
    thinking={
        "type": "enabled",
    },
)


llm_ds = ChatOpenAI(
        api_key=DEEPSEEK_API_KEY,  # 硅基流动API Key
        base_url=DEEPSEEK_BASE_URL,  # 对话接口完整URL（你提供的无需补全）
        model=DEEPSEEK_CHAT_MODEL,  # 硅基流动对话模型名
        temperature=0.0,  # 随机性
        max_retries=3,  # 重试次数
        timeout=60,  # 超时时间
)
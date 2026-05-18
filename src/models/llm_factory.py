import os
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel

def get_llm(model_name: str = None, temperature: float = 0.7) -> BaseChatModel:
    """
    获取大模型实例。
    默认使用 OpenAI 兼容 API。由于目前绝大多数大模型厂商（DeepSeek, 通义千问, 智谱等）
    以及本地部署工具（Ollama, vLLM）都兼容 OpenAI API 格式，使用 ChatOpenAI 是最通用的选择。
    
    只需设置以下环境变量即可切换模型：
    - OPENAI_API_KEY
    - OPENAI_API_BASE (如需要)
    - DEEPTHINK_MODEL_NAME (默认模型名称，如 'gpt-3.5-turbo', 'deepseek-chat' 等)
    """
    if model_name is None:
        model_name = os.environ.get("DEEPTHINK_MODEL_NAME", "gpt-3.5-turbo")
        
    base_url = os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL")
    
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        base_url=base_url,
        # max_retries 等参数可以在这里统一配置
    )

import os
import re
import time
import json
import logging
from typing import Optional, Type, Any
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)


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
    )


def _extract_usage(response) -> dict:
    """从 LangChain 响应中提取 token 使用统计。"""
    metadata = getattr(response, "response_metadata", {}) or {}
    usage = metadata.get("token_usage", {}) or {}
    return {
        "prompt_tokens": usage.get("prompt_tokens", 0) or 0,
        "completion_tokens": usage.get("completion_tokens", 0) or 0,
        "total_tokens": usage.get("total_tokens", 0) or 0,
    }


def _accumulate_usage(current: Optional[dict], delta: dict) -> dict:
    """累积 token 使用量。"""
    if current is None:
        current = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return {
        "prompt_tokens": current.get("prompt_tokens", 0) + delta.get("prompt_tokens", 0),
        "completion_tokens": current.get("completion_tokens", 0) + delta.get("completion_tokens", 0),
        "total_tokens": current.get("total_tokens", 0) + delta.get("total_tokens", 0),
    }


def _normalize_structured_data(data: dict, schema: Type[BaseModel]) -> dict:
    """
    对常见模型输出格式做容错映射，使其尽量匹配 Pydantic schema。
    例如把 clarifying_questions / status 映射到 response / is_clear。
    """
    normalized = dict(data)
    lower_keys = {k.lower(): k for k in normalized}

    # ClarifierOutput 兼容
    if "is_clear" not in normalized:
        if "status" in lower_keys:
            status_value = normalized[lower_keys["status"]]
            if isinstance(status_value, str):
                normalized["is_clear"] = status_value.lower() in ("clear", "cleared", "ok", "confirmed", "true")
            else:
                normalized["is_clear"] = bool(status_value)

    if "response" not in normalized:
        # 尝试把 clarifying_questions / questions / message / answer 等映射为 response
        for alt_key in ["clarifying_questions", "questions", "message", "answer", "reconstructed_question", "clarified_question"]:
            if alt_key in normalized:
                value = normalized[alt_key]
                if isinstance(value, list):
                    normalized["response"] = "\n".join(f"{i+1}. {q}" for i, q in enumerate(value))
                else:
                    normalized["response"] = str(value)
                break

    # ReviewFeedback 兼容
    if "score" in normalized and isinstance(normalized["score"], (int, float)):
        normalized["score"] = int(normalized["score"])
    if "passed" not in normalized and "score" in normalized:
        normalized["passed"] = normalized["score"] >= 8
    if "comments" not in normalized:
        for alt_key in ["feedback", "suggestions", "review", "comment", "advice"]:
            if alt_key in normalized:
                normalized["comments"] = str(normalized[alt_key])
                break

    # JudgeOutput 兼容
    if "strengths" in normalized and not isinstance(normalized["strengths"], list):
        normalized["strengths"] = [str(normalized["strengths"])]
    if "weaknesses" in normalized and not isinstance(normalized["weaknesses"], list):
        normalized["weaknesses"] = [str(normalized["weaknesses"])]

    return normalized


def _safe_json_parse(text: str, schema: Type[BaseModel]) -> Optional[BaseModel]:
    """
    尝试从文本中提取并解析 JSON，然后反序列化为 Pydantic 模型。
    兼容 ```json 代码块和普通 JSON 对象，并对常见字段别名做容错映射。
    """
    text = text.strip()
    # 先尝试整块解析
    candidates = [text]
    # 提取 ```json ... ``` 内容
    fenced = re.findall(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    candidates.extend(fenced)
    # 提取最外层 { ... }
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        candidates.append(brace_match.group(0))

    for candidate in candidates:
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                data = _normalize_structured_data(data, schema)
            return schema.model_validate(data)
        except Exception:
            continue
    return None


def invoke_llm(
    messages: list[BaseMessage],
    state: Optional[dict] = None,
    temperature: float = 0.7,
    max_retries: int = 2,
) -> tuple[str, dict, Any]:
    """
    统一调用大模型，自动处理重试和 token 统计。

    Returns:
        content: 模型生成的文本
        token_usage: 累积后的 token 使用量
        response: 原始 LangChain 响应对象
    """
    llm = get_llm(temperature=temperature)
    current_usage = state.get("token_usage") if state else None

    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            response = llm.invoke(messages)
            usage = _extract_usage(response)
            new_usage = _accumulate_usage(current_usage, usage)
            return response.content, new_usage, response
        except Exception as e:
            last_exception = e
            logger.warning(f"LLM 调用失败 (attempt {attempt + 1}/{max_retries + 1}): {e}")
            if attempt < max_retries:
                time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"LLM 调用在 {max_retries + 1} 次尝试后仍然失败: {last_exception}")


def invoke_structured_llm(
    messages: list[BaseMessage],
    schema: Type[BaseModel],
    state: Optional[dict] = None,
    temperature: float = 0.7,
    max_retries: int = 2,
) -> tuple[BaseModel, dict, Any]:
    """
    统一调用大模型并请求结构化输出。

    优先使用 ChatOpenAI.with_structured_output；如果后端不支持函数调用/结构化输出，
    则自动降级为普通文本生成 + JSON 解析，并在解析失败时重试。

    如果模型/后端已知不支持结构化输出，可设置环境变量 DEEPTHINK_FORCE_JSON_FALLBACK=1
    直接走 JSON 解析降级路径，避免一次失败的 API 调用。
    """
    current_usage = state.get("token_usage") if state else None
    force_fallback = os.environ.get("DEEPTHINK_FORCE_JSON_FALLBACK", "").lower() in ("1", "true", "yes")
    llm = get_llm(temperature=temperature)

    # 第一次尝试：使用原生结构化输出能力
    if not force_fallback:
        try:
            structured_llm = llm.with_structured_output(schema)
            response = structured_llm.invoke(messages)
            usage = _extract_usage(response)
            new_usage = _accumulate_usage(current_usage, usage)
            return response, new_usage, response
        except Exception as e:
            logger.warning(f"结构化输出调用失败，将降级为 JSON 解析: {e}")

    # 降级方案：普通文本生成 + JSON 解析，并自动重试
    plain_messages = list(messages)
    schema_dict = schema.model_json_schema()
    # 移除 Pydantic 内部 $defs，保留字段描述
    schema_dict.pop("$defs", None)
    schema_json = json.dumps(schema_dict, ensure_ascii=False, indent=2)
    extra_instruction = (
        "\n\n你必须严格输出合法 JSON，不要包含任何 markdown 代码块或其他解释文本。"
        f"请严格符合以下 JSON Schema：\n{schema_json}"
    )
    if plain_messages:
        last_msg = plain_messages[-1]
        if hasattr(last_msg, "content") and isinstance(last_msg.content, str):
            plain_messages[-1] = last_msg.__class__(content=last_msg.content + extra_instruction)

    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            response = llm.invoke(plain_messages)
            usage = _extract_usage(response)
            new_usage = _accumulate_usage(current_usage, usage)
            parsed = _safe_json_parse(response.content, schema)
            if parsed is not None:
                return parsed, new_usage, response
            last_exception = ValueError(f"无法从模型输出中解析出 {schema.__name__}: {response.content[:200]}")
        except Exception as e:
            last_exception = e
            logger.warning(f"结构化输出降级解析失败 (attempt {attempt + 1}/{max_retries + 1}): {e}")
        if attempt < max_retries:
            time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"结构化输出在 {max_retries + 1} 次尝试后仍然失败: {last_exception}")

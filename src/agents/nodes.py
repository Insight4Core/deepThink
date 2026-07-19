import os
import yaml
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.messages import SystemMessage, HumanMessage

from src.models.llm_factory import invoke_llm, invoke_structured_llm
from src.models.schemas import ClarifierOutput, ReviewFeedback
from src.agents.state import GraphState

logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "prompts.yaml")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    PROMPTS = yaml.safe_load(f)


def _safe_format(template: str, **kwargs) -> str:
    """
    安全地替换模板中的指定占位符，同时把模板中其它 { / } 转义为 {{ / }}，
    避免 .format() 遇到用户内容中的花括号时抛出 KeyError。
    """
    # 先把真实占位符换成临时标记，避免被转义
    temp_marker = "\x00{key}\x00"
    intermediate = template
    placeholder_to_marker = {}
    for key, value in kwargs.items():
        placeholder = "{" + key + "}"
        marker = temp_marker.format(key=key)
        placeholder_to_marker[marker] = str(value)
        intermediate = intermediate.replace(placeholder, marker)

    # 转义剩余花括号
    intermediate = intermediate.replace("{", "{{").replace("}", "}}")

    # 还原真实占位符
    for marker, value in placeholder_to_marker.items():
        intermediate = intermediate.replace(marker, value)

    return intermediate.format(**kwargs)


def clarifier_node(state: GraphState) -> dict:
    original_question = state["original_question"]
    history = state.get("clarification_history", [])

    history_str = "无"
    if history:
        history_str = "\n".join([f"{item['role']}: {item['content']}" for item in history])

    rounds = state.get("clarification_rounds", 0)
    instruction_extra = ""
    if rounds >= 3:
        instruction_extra = (
            "【最高优先级警告】追问次数已达上限！你本次必须设置 is_clear=true，"
            "并且 response 必须是结合了所有历史记录重构后的【最终清晰问题】，绝对不能再向用户追问！"
        )

    prompt = _safe_format(
        PROMPTS["clarifier"]["system"],
        original_question=original_question,
        history_str=history_str,
        instruction_extra=instruction_extra,
    )

    parsed, token_usage, _ = invoke_structured_llm(
        [HumanMessage(content=prompt)],
        ClarifierOutput,
        state=state,
        temperature=0.3,
    )

    # 强制熔断：超过 3 轮必须进入生成阶段
    if rounds >= 3:
        parsed.is_clear = True

    if not parsed.is_clear:
        return {
            "needs_clarification": True,
            "clarification_message": parsed.response,
            "clarification_rounds": rounds + 1,
            "token_usage": token_usage,
        }
    else:
        return {
            "needs_clarification": False,
            "clarified_question": parsed.response,
            "token_usage": token_usage,
        }


def ask_human_node(state: GraphState):
    """
    占位节点：在进入此节点前图会被挂起 (interrupt)。
    主程序通过 update_state 作为此节点输入人类的回答。
    """
    pass


def generator_node(state: GraphState) -> dict:
    question = state.get("clarified_question", state["original_question"])

    system_text = PROMPTS["generator"]["system"]
    messages = [
        SystemMessage(content=system_text),
        HumanMessage(content=f"用户问题：\n{question}"),
    ]

    content, token_usage, _ = invoke_llm(messages, state=state, temperature=0.7)

    return {
        "current_draft": content,
        "iteration_count": state.get("iteration_count", 0),
        "token_usage": token_usage,
    }


def _review_single_role(role_name: str, role_prompt: str, question: str, draft: str, state: GraphState) -> tuple:
    """单个 Reviewer 的并行调用单元。"""
    system_text = (
        f"你现在的角色是：{role_name}。{role_prompt}\n\n"
        "请对下面的原问题和当前草稿回答进行审查，并给出结构化评分。"
    )
    human_text = (
        f"【原问题】\n{question}\n\n"
        f"【当前草稿回答】\n{draft}"
    )
    messages = [
        SystemMessage(content=system_text),
        HumanMessage(content=human_text),
    ]

    parsed, token_usage, _ = invoke_structured_llm(
        messages,
        ReviewFeedback,
        state=state,
        temperature=0.3,
    )
    return role_name, parsed, token_usage


def reviewer_node(state: GraphState) -> dict:
    question = state.get("clarified_question", state["original_question"])
    draft = state["current_draft"]
    roles = PROMPTS.get("reviewer_roles", {})

    if not roles:
        logger.warning("未配置任何 reviewer 角色，跳过审查直接通过")
        return {
            "review_feedback": {},
            "is_approved": True,
            "token_usage": state.get("token_usage"),
        }

    review_feedback = {}
    total_usage = state.get("token_usage") or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    all_passed = True

    # 并行调用所有 Reviewer，减少总体等待时间
    with ThreadPoolExecutor(max_workers=min(len(roles), 4)) as executor:
        futures = {
            executor.submit(_review_single_role, role_name, role_prompt, question, draft, state): role_name
            for role_name, role_prompt in roles.items()
        }
        for future in as_completed(futures):
            role_name, parsed, usage = future.result()
            review_feedback[role_name] = parsed
            all_passed = all_passed and parsed.passed
            for key in total_usage:
                total_usage[key] += usage.get(key, 0)

    return {
        "review_feedback": review_feedback,
        "is_approved": all_passed,
        "token_usage": total_usage,
    }


def reviser_node(state: GraphState) -> dict:
    question = state.get("clarified_question", state["original_question"])
    draft = state["current_draft"]
    feedback = state.get("review_feedback", {})

    feedback_lines = []
    for role, fb in feedback.items():
        if isinstance(fb, ReviewFeedback):
            feedback_lines.append(f"【{role}】评分: {fb.score}/10\n意见:\n{fb.comments}")
        else:
            feedback_lines.append(f"【{role}】\n{fb}")
    feedback_str = "\n\n".join(feedback_lines)

    system_text = PROMPTS["reviser"]["system"]
    human_text = (
        f"【原问题】\n{question}\n\n"
        f"【当前草稿】\n{draft}\n\n"
        f"【审查意见】\n{feedback_str}"
    )
    messages = [
        SystemMessage(content=system_text),
        HumanMessage(content=human_text),
    ]

    content, token_usage, _ = invoke_llm(messages, state=state, temperature=0.7)

    return {
        "current_draft": content,
        "iteration_count": state.get("iteration_count", 0) + 1,
        "token_usage": token_usage,
    }

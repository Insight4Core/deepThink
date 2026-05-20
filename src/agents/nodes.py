import os
import yaml
from langchain_core.messages import SystemMessage, HumanMessage
from src.models.llm_factory import get_llm
from src.agents.state import GraphState

# 加载外部 Prompt 配置文件
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "prompts.yaml")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    PROMPTS = yaml.safe_load(f)

def _accumulate_tokens(state: GraphState, response) -> dict:
    current_usage = state.get("token_usage")
    if not current_usage:
        current_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
    metadata = getattr(response, "response_metadata", {})
    usage = metadata.get("token_usage", {})
    if usage:
        return {
            "prompt_tokens": current_usage.get("prompt_tokens", 0) + usage.get("prompt_tokens", 0),
            "completion_tokens": current_usage.get("completion_tokens", 0) + usage.get("completion_tokens", 0),
            "total_tokens": current_usage.get("total_tokens", 0) + usage.get("total_tokens", 0),
        }
    return current_usage

def clarifier_node(state: GraphState) -> GraphState:
    """
    意图澄清节点：分析用户输入，如果不清晰则进行扩充和澄清。
    目前采取自动扩写模式（Autonomous Refinement）。
    """
    llm = get_llm()
    original_question = state["original_question"]
    
    # 从配置文件读取 Prompt 模板并格式化
    prompt_template = PROMPTS["clarifier"]["system"]
    prompt = prompt_template.format(original_question=original_question)
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "clarified_question": response.content,
        "token_usage": _accumulate_tokens(state, response)
    }

def generator_node(state: GraphState) -> GraphState:
    """
    生成初始草稿节点。
    """
    llm = get_llm()
    question = state.get("clarified_question", state["original_question"])
    
    # 从配置文件读取 Prompt 模板并格式化
    prompt_template = PROMPTS["generator"]["system"]
    prompt = prompt_template.format(question=question)
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "current_draft": response.content,
        "iteration_count": state.get("iteration_count", 0),
        "token_usage": _accumulate_tokens(state, response)
    }

def reviewer_node(state: GraphState) -> GraphState:
    """
    多角色审查节点。
    模拟多个不同视角的 Reviewer 对当前草稿进行审查。
    """
    llm = get_llm()
    question = state.get("clarified_question", state["original_question"])
    draft = state["current_draft"]
    
    # 从配置文件动态读取所有的角色和基础模板
    roles = PROMPTS.get("reviewers", {})
    reviewer_base = PROMPTS["reviewer_base"]
    
    review_feedback = {}
    all_passed = True
    
    current_state_usage = state.get("token_usage")
    if not current_state_usage:
        current_state_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    for role_name, role_prompt in roles.items():
        # 格式化各个角色的具体 Prompt
        prompt = reviewer_base.format(
            role_name=role_name,
            role_prompt=role_prompt,
            question=question,
            draft=draft
        )
        
        response = llm.invoke([HumanMessage(content=prompt)])
        feedback = response.content
        review_feedback[role_name] = feedback
        
        if "PASS" not in feedback.upper():
            all_passed = False
            
        # 累加 Token
        metadata = getattr(response, "response_metadata", {})
        usage = metadata.get("token_usage", {})
        if usage:
            current_state_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
            current_state_usage["completion_tokens"] += usage.get("completion_tokens", 0)
            current_state_usage["total_tokens"] += usage.get("total_tokens", 0)
            
    return {
        "review_feedback": review_feedback,
        "is_approved": all_passed,
        "token_usage": current_state_usage
    }

def reviser_node(state: GraphState) -> GraphState:
    """
    根据审查意见修改节点。
    """
    llm = get_llm()
    question = state.get("clarified_question", state["original_question"])
    draft = state["current_draft"]
    feedback = state.get("review_feedback", {})
    
    feedback_str = "\n".join([f"【{role}】的意见:\n{fb}" for role, fb in feedback.items()])
    
    # 从配置文件读取 Prompt 模板并格式化
    prompt_template = PROMPTS["reviser"]["system"]
    prompt = prompt_template.format(
        question=question,
        draft=draft,
        feedback=feedback_str
    )
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "current_draft": response.content,
        "iteration_count": state.get("iteration_count", 0) + 1,
        "token_usage": _accumulate_tokens(state, response)
    }

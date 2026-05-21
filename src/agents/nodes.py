import os
import yaml
import json
from langchain_core.messages import SystemMessage, HumanMessage
from src.models.llm_factory import get_llm
from src.agents.state import GraphState

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
    llm = get_llm()
    original_question = state["original_question"]
    history = state.get("clarification_history", [])
    
    history_str = "无"
    if history:
        history_str = "\n".join([f"{item['role']}: {item['content']}" for item in history])
        
    rounds = state.get("clarification_rounds", 0)
    instruction_extra = ""
    if rounds >= 3:
        instruction_extra = "【最高优先级警告】追问次数已达上限！你本次必须设置 \"is_clear\": true，并且 response 必须是结合了所有历史记录重构后的【最终清晰问题】，绝对不能再向用户追问！"
    
    prompt_template = PROMPTS["clarifier"]["system"]
    prompt = prompt_template.format(
        original_question=original_question, 
        history_str=history_str,
        instruction_extra=instruction_extra
    )
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    # 尝试解析 JSON
    try:
        content = response.content.strip()
        import re
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            content = match.group(0)
            
        data = json.loads(content)
        is_clear = data.get("is_clear", True)
        msg = data.get("response", response.content)
    except Exception as e:
        # 如果解析失败，默认当作清楚了，直接输出
        is_clear = True
        msg = response.content

    # 强制熔断机制：即使大模型没有听话，超过3轮也强制跳出
    if rounds >= 3:
        is_clear = True
        
    # 如果不清楚并且没有超过3轮
    if not is_clear:
        return {
            "needs_clarification": True,
            "clarification_message": msg,
            "clarification_rounds": rounds + 1,
            "token_usage": _accumulate_tokens(state, response)
        }
    else:
        return {
            "needs_clarification": False,
            "clarified_question": msg,
            "token_usage": _accumulate_tokens(state, response)
        }

def ask_human_node(state: GraphState):
    """
    占位节点：在进入此节点前图会被挂起 (interrupt)。
    主程序通过 update_state 作为此节点输入人类的回答。
    """
    pass

def generator_node(state: GraphState) -> GraphState:
    llm = get_llm()
    question = state.get("clarified_question", state["original_question"])
    
    prompt_template = PROMPTS["generator"]["system"]
    prompt = prompt_template.format(question=question)
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "current_draft": response.content,
        "iteration_count": state.get("iteration_count", 0),
        "token_usage": _accumulate_tokens(state, response)
    }

def reviewer_node(state: GraphState) -> GraphState:
    llm = get_llm()
    question = state.get("clarified_question", state["original_question"])
    draft = state["current_draft"]
    
    roles = PROMPTS.get("reviewers", {})
    reviewer_base = PROMPTS["reviewer_base"]
    
    review_feedback = {}
    all_passed = True
    
    current_state_usage = state.get("token_usage")
    if not current_state_usage:
        current_state_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    for role_name, role_prompt in roles.items():
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
    llm = get_llm()
    question = state.get("clarified_question", state["original_question"])
    draft = state["current_draft"]
    feedback = state.get("review_feedback", {})
    
    feedback_str = "\n".join([f"【{role}】的意见:\n{fb}" for role, fb in feedback.items()])
    
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

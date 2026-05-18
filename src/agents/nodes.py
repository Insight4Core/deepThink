from langchain_core.messages import SystemMessage, HumanMessage
from src.models.llm_factory import get_llm
from src.agents.state import GraphState

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
    
    prompt = f"""
    你是一个问题分析与澄清专家。
    用户的原始问题是："{original_question}"
    
    请分析这个问题是否清晰、明确、包含足够的上下文。
    如果问题太短或模糊，请将其重构、扩写为一个清晰、具体、结构化且包含合理假设的问题，以帮助下游 AI 更好地回答。
    如果原问题已经很清晰，请只对它做轻微润色。
    
    只需输出重构后的问题，不要输出其他废话。
    """
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
    
    prompt = f"""
    你是一个专家级助手。请尽可能详尽、准确地回答以下问题：
    
    问题：{question}
    """
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
    
    roles = {
        "逻辑审查员": "请重点关注回答的逻辑是否严密，推理是否有漏洞，结构是否清晰。",
        "全面性审查员": "请重点关注回答是否全面，有没有遗漏问题中隐含的关键点，或者可以补充的更深层次的视角。"
    }
    
    review_feedback = {}
    all_passed = True
    
    current_state_usage = state.get("token_usage")
    if not current_state_usage:
        current_state_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    for role_name, role_prompt in roles.items():
        prompt = f"""
        你现在的角色是：{role_name}。{role_prompt}
        
        原问题：{question}
        当前的草稿回答：
        {draft}
        
        请给出你的审查意见。指出具体需要改进的地方。
        如果你认为当前的回答在这个视角下已经非常完美，没有任何需要修改的地方，请在你的回复最后另起一行输出 "PASS"。
        否则，请给出详细的修改建议。
        """
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
    
    prompt = f"""
    你是一个优化专家。你需要根据多位审查员的意见，对当前的回答草稿进行修改和完善。
    
    原问题：{question}
    
    当前草稿：
    {draft}
    
    审查意见：
    {feedback_str}
    
    请严格综合上述审查意见，重新写出一份完美的回答。直接输出新的回答内容即可。
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "current_draft": response.content,
        "iteration_count": state.get("iteration_count", 0) + 1,
        "token_usage": _accumulate_tokens(state, response)
    }

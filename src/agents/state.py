from typing import TypedDict, Annotated, List, Dict, Any
import operator
from langchain_core.messages import BaseMessage

class GraphState(TypedDict):
    """
    LangGraph 状态定义。
    它在图的各个节点之间传递。
    """
    # 用户的原始问题
    original_question: str
    
    # 澄清后的问题（如果原始问题不清楚，这里存放被补充或扩写后的问题）
    clarified_question: str
    
    # 对话历史，或者用于节点间传递消息（必须使用 Annotated 和 operator.add 保证消息被追加而不是覆盖）
    messages: Annotated[List[BaseMessage], operator.add]
    
    # [新增] 多轮澄清历史
    clarification_history: Annotated[List[Dict[str, str]], operator.add]
    
    # [新增] 澄清控制字段
    needs_clarification: bool
    clarification_message: str
    clarification_rounds: int
    
    # 当前草稿 / 答案
    current_draft: str
    
    # Reviewer 们的反馈意见 (每个reviewer的反馈保存在这里)
    # 例如: {"logic_reviewer": "...", "accuracy_reviewer": "..."}
    review_feedback: Dict[str, str]
    
    # 当前迭代次数
    iteration_count: int
    
    # 最大迭代次数，避免无限死循环
    max_iterations: int
    
    # 是否通过了所有的审查
    is_approved: bool
    
    # 统计 token 消耗
    token_usage: Dict[str, int]

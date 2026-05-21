from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from src.agents.state import GraphState
from src.agents.nodes import clarifier_node, ask_human_node, generator_node, reviewer_node, reviser_node

def should_continue(state: GraphState) -> str:
    """
    路由逻辑：决定是继续修改还是结束。
    """
    if state.get("is_approved", False):
        return END
    
    if state.get("iteration_count", 0) >= state.get("max_iterations", 3):
        return END
        
    return "reviser"

def should_clarify(state: GraphState) -> str:
    """
    路由逻辑：判断是需要询问人类还是继续生成
    """
    if state.get("needs_clarification"):
        return "ask_human"
    return "generator"

def build_graph() -> StateGraph:
    """
    构建整个图工作流
    """
    workflow = StateGraph(GraphState)
    
    # 添加节点
    workflow.add_node("clarifier", clarifier_node)
    workflow.add_node("ask_human", ask_human_node)
    workflow.add_node("generator", generator_node)
    workflow.add_node("reviewer", reviewer_node)
    workflow.add_node("reviser", reviser_node)
    
    # 定义边 (数据流向)
    workflow.add_edge(START, "clarifier")
    
    # 根据 clarifier 结果分流
    workflow.add_conditional_edges(
        "clarifier",
        should_clarify,
        {
            "ask_human": "ask_human",
            "generator": "generator"
        }
    )
    
    # 人类补充完毕后，回到 clarifier 重新判断
    workflow.add_edge("ask_human", "clarifier")
    
    workflow.add_edge("generator", "reviewer")
    
    # 条件路由：从 reviewer 节点出发
    workflow.add_conditional_edges(
        "reviewer",
        should_continue,
        {
            "reviser": "reviser",
            END: END
        }
    )
    
    # 闭环：修改完毕后再次送审
    workflow.add_edge("reviser", "reviewer")
    
    # 编译图，配置记忆点并设置挂起点 (interrupt_before)
    memory = MemorySaver()
    app = workflow.compile(
        checkpointer=memory,
        interrupt_before=["ask_human"]
    )
    
    return app

from langgraph.graph import StateGraph, START, END
from src.agents.state import GraphState
from src.agents.nodes import clarifier_node, generator_node, reviewer_node, reviser_node

def should_continue(state: GraphState) -> str:
    """
    路由逻辑：决定是继续修改还是结束。
    """
    if state.get("is_approved", False):
        return END
    
    if state.get("iteration_count", 0) >= state.get("max_iterations", 3):
        return END
        
    return "reviser"

def build_graph() -> StateGraph:
    """
    构建整个图工作流
    """
    workflow = StateGraph(GraphState)
    
    # 添加节点
    workflow.add_node("clarifier", clarifier_node)
    workflow.add_node("generator", generator_node)
    workflow.add_node("reviewer", reviewer_node)
    workflow.add_node("reviser", reviser_node)
    
    # 定义边 (数据流向)
    workflow.add_edge(START, "clarifier")
    workflow.add_edge("clarifier", "generator")
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
    
    # 编译图
    app = workflow.compile()
    
    return app

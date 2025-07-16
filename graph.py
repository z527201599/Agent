from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from Agent_BadJokes import AgentState, our_agent, tools, should_continue_edge

def build_graph():
    graph = StateGraph(AgentState)  # 创建状态图
    graph.add_node("agent", our_agent)  # 添加模型调用节点
    graph.add_node("tools", ToolNode(tools=tools))  # 添加工具节点

    graph.set_entry_point("agent")  # 设置入口点
    
    graph.add_edge("agent", "tools")  # 添加边
    graph.add_conditional_edges(
        "tools",
        should_continue_edge,
        {
            "continue": "agent",
            "end": END
        },
    )
    return graph.compile()

from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
load_dotenv()

# Document全局变量
document_content = ""

class AgentState(TypedDict):  #State框架
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def update(content: str) -> str:  # LLM传content给tool
    """用户需要生成内容，或者修改时调用,根据传入更新文档内容"""
    global document_content
    document_content = content
    return(f"Document updated, current content is: {document_content}")

@tool
def save(filename: str) -> str:
    """用户要保存时调用，保存文档内容到指定文件(text)，生成中文文件名
    Args:
        filename: 文件名
    """
    global document_content
    if not filename.endswith('.txt'):
        filename += '.txt'
    
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(document_content)
        print(f"Document saved to {filename}.")
        return f"Document has been saved to {filename}."
    
    except Exception as e:
        return f"Error saving document: {str(e)}"

@tool
def reset() -> str:
    """重置全局变量内容为空"""
    global document_content
    document_content = ""
    return "Document content has been reset."

tools = [update, save, reset]

model = ChatOllama(
    model="qwen3:1.7b",
    base_url="http://localhost:11434"
).bind_tools(tools)

def our_agent(state: AgentState) -> AgentState:
    """处理输入的请求"""
    #实例化一个SystemMessage
    system_prompt = SystemMessage(content=f"""
                                  <role>
                                  你是一个专业冷笑话写手，帮助用户创作冷笑话.
                                  <instructions>
                                  - 如果用户需要修改就调用 'update' tool.
                                  - 如果用户完成创作需要保存就调用 'save' tool.
                                  - 如果用户需要重置就调用 'reset' tool.
                                  - 在每次修改之后都要展示当前的 document state.
                                  - 当前的 document content 是: {document_content}
                                  - 用中文进行创作和文件保存.
                                  <subject>
                                  - 短篇冷笑话,字数在100以内.
                                  <preset>
                                  - 冷笑话的本质在于刻意制造一个微小的、合乎表面逻辑的认知失调，和让人感到意外的结尾
                                  - 笑话 = 铺垫 + 笑点，铺垫制造预期，笑点揭示意外，需要两条故事线的转换
                                  - 例子：一根火柴走在路上，觉得头痒，就挠了挠，结果着火了。
                                  """)
    if not state["messages"]:
        user_input = input("我可以帮你创作冷笑话，你想创作的主题是什么？")
        user_message = HumanMessage(content=user_input)

    else:
        user_input = input("\n你想要如何修改?")
        print(f"\n User: {user_input}")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state["messages"]) + [user_message]
    response = model.invoke(all_messages)

    print(f"\n AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"\n Tool Calls: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": list(state["messages"]) + [user_message, response]}

def should_continue_edge(state: AgentState):
    """判断是否继续"""

    messages = state["messages"]

    if not messages:
        return "continue"  # 如果没有消息，继续
    
    # 检视最近的tool message, 并且判断关键字
    for message in reversed(messages):
        if (isinstance(message, ToolMessage) and
            "saved" in message.content.lower() and
            "document" in message.content.lower()):

            return "end" # 如果包含保存文档的消息，指向结束节点
        
    return "continue"  # 注意是遍历结束,则继续

def print_message(messages):
    """可视化"""
    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"Tool Call: {message.content}")

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
app = graph.compile()

def run_agent():
    print("开始运行!")

    state = {"messages": []}

    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_message(step["messages"])
    
    print("\n运行结束.")

if __name__ == "__main__":
    run_agent()
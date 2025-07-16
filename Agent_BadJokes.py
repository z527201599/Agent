
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from config import OLLAMA_MODEL, OLLAMA_BASE_URL
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
import asyncio


load_dotenv()

# Document全局变量
document_content = ""

class AgentState(TypedDict):  #State框架
    messages: Annotated[Sequence[BaseMessage], add_messages]  # 上下文记忆

@tool
def update(content: str) -> str:  # LLM传content给tool
    """用户需要生成内容，或者修改时调用,根据传入更新文档内容"""
    global document_content
    document_content = content
    return(f"文件已更新, 当前内容是: {document_content}")

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
        print(f"文件储存为 {filename}.")
        return f"文件已储存为 {filename}."
    
    except Exception as e:
        return f"Error saving document: {str(e)}"

@tool
def reset() -> str:
    """重置全局变量内容为空"""
    global document_content
    document_content = ""
    return "草稿内容已重置，重新创作."

tools = [update, save, reset]

model = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL
).bind_tools(tools)

async def our_agent(state: AgentState) -> AgentState:
    """处理输入的请求"""
    try:
        loop = asyncio.get_event_loop()  #可以用executor
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
            user_input = await loop.run_in_executor(None, input, "我可以帮你创作冷笑话，你想创作的主题是什么？")
            user_message = HumanMessage(content=user_input)
        else:
            user_input = await loop.run_in_executor(None, input, "\n你想要如何修改, 或者储存?")
            print(f"\n-- User: {user_input}")
            user_message = HumanMessage(content=user_input)

        all_messages = [system_prompt] + list(state["messages"]) + [user_message]
        # 可用run_in_executor包装
        try:
            response = await loop.run_in_executor(None, model.invoke, all_messages)
        except Exception as e:
            print(f"模型调用异常: {e}")
            response = AIMessage(content=f"模型调用失败: {e}")

        print(f"\n-- AI回答: {response.content}")  #不调用工具时的回复内容
        if hasattr(response, "tool_calls") and response.tool_calls:
            print(f"\n-- Tool Calls选择: {[tc['name'] for tc in response.tool_calls]}")

        return {"messages": list(state["messages"]) + [user_message, response]}
    except Exception as e:
        print(f"Agent异常: {e}")
        return {"messages": list(state["messages"]) + [AIMessage(content=f"Agent异常: {e}")]}

def should_continue_edge(state: AgentState):
    """判断是否继续"""

    messages = state["messages"]

    if not messages:
        return "continue"  # 如果没有消息，继续
    
    # 检视最近的tool message, 并且判断关键字
    for message in reversed(messages):
        if (isinstance(message, ToolMessage) and
            "储存" in message.content.lower() and
            "文件" in message.content.lower()):

            return "end" # 如果包含保存文档的消息，指向结束节点
        
    return "continue"  # 注意是遍历结束,则继续

def print_message(messages):
    """可视化调用Tool"""
    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"-- Tool Call: {message.content}")


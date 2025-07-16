
# 需要改进，用interrupt节点构建graph，需要添加人工反馈，invoke和resume分开
# #

from fastapi import FastAPI, Body
from pydantic import BaseModel
import asyncio
from graph import build_graph
from Agent_BadJokes import print_message
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage

app = FastAPI()

class Request(BaseModel):
    theme: str

# class ModifyRequest(BaseModel):
#     messages: list
#     new_content: str

@app.post("/generate")
async def generate (reqB: Request):
    """生成"""
    graph_api = build_graph()

    state = {"messages": [HumanMessage(content=reqB.theme)]}  # Request(theme="动物")，用 reqB.theme 访问
        
    async for step in graph_api.astream(state, stream_mode="values"):

        for message in step["messages"][-3:]:

            if isinstance(message, ToolMessage):

                print(step["messages"][-1])

                return(f"-- Content: {message.content}")

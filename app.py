import asyncio
from graph import build_graph
from Agent_BadJokes import print_message

async def run_agent():
    print("===开始运行!===")
    app = build_graph()
    state = {"messages": []}
    try:
        async for step in app.astream(state, stream_mode="values"):  #stream方法只取values部分
            if "messages" in step:
                #print(step["messages"])
                print_message(step["messages"])

    except Exception as e:
        print(f"主流程异常: {e}")
    print("\n===运行结束.===")

if __name__ == "__main__":
    asyncio.run(run_agent())
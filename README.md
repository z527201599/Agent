# LangG_Agent 

本项目是一个基于 LangChain、LangGraph 和本地/云大模型（如 OpenAI、Ollama）的冷笑话自动生成与保存工具。你可以通过自然语言与 Agent 互动，生成、修改、保存短篇冷笑话。

## 主要功能
- 支持用户输入主题，自动生成短篇冷笑话（100字以内，中文）
- 支持修改、重置冷笑话内容
- 支持将冷笑话保存为本地 txt 文件，文件名可用中文
- 多轮对话，自动管理对话历史和工具调用

## 依赖环境
- Python 3.8+
- 主要依赖包见 requirements.txt，包括：
  - langchain
  - langgraph
  - langchain-openai
  - langchain-ollama
  - python-dotenv
  - openai

## 快速开始
1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
2. 配置模型参数（如 OLLAMA_MODEL、OLLAMA_BASE_URL），可在 config.py 或 .env 文件中设置。
3. 运行主程序：
   ```bash
   python app.py
   ```
4. 按提示输入冷笑话主题，或进行修改、保存等操作。

## 说明
- 生成的冷笑话会自动保存到指定的 txt 文件，支持中文文件名。
- 支持本地 Ollama 模型或 OpenAI 云端模型。
- 适合中文冷笑话创作、文本生成等场景。



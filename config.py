import os
from dotenv import load_dotenv

load_dotenv()

# Ollama模型配置
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:1.7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# 其他可扩展配置
# Token限制
MAX_INPUT_TOKENS = 15000
MAX_PROMPT_TOKENS = 28000

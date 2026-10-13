import requests
import os
from dotenv import load_dotenv

# 載入 .env
load_dotenv()

invoke_url = "https://openrouter.ai/api/v1/chat/completions"
api_key = os.getenv("openroute_KEY") or os.getenv("OPENROUTER_API_KEY")

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://github.com/project-shiori",
    "X-Title": "Project Shiori Light Novel RAG"
}

payload = {
    "model": "openai/gpt-6-luna",
    "messages": [{"role": "user", "content": "請回覆一句有哲理的短句"}],
    "temperature": 0.2,
    "max_tokens": 1024
}

print("--- 正在透過 requests 測試 OpenRouter (模型: openai/gpt-6-luna) ---")
try:
    response = requests.post(invoke_url, headers=headers, json=payload, timeout=25)
    response.raise_for_status()
    print("回傳內容:", response.json()["choices"][0]["message"].get("content"))
    print("--- 測試成功 ---")
except Exception as e:
    print(f"❌ 錯誤: {e}")

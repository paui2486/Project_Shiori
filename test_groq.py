import os
import requests
from dotenv import load_dotenv

# 載入 .env
load_dotenv()

api_key = os.getenv("openroute_KEY") or os.getenv("OPENROUTER_API_KEY")
model_name = "openai/gpt-6-luna"

print(f"--- 正在測試 OpenRouter API (模型: {model_name}) ---")

try:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/project-shiori",
        "X-Title": "Project Shiori Light Novel RAG"
    }
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": "請簡短介紹輕小說《GATE 奇幻自衛隊》的魅力（50字以內）"
            }
        ],
        "temperature": 0.2,
        "max_tokens": 1024
    }
    resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=25)
    resp.raise_for_status()
    res_data = resp.json()
    content = res_data["choices"][0]["message"].get("content")
    print(f"回答: {content}")
    print("\n--- 測試完成 ---")

except Exception as e:
    print(f"\n❌ 發生錯誤: {e}")

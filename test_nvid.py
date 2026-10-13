from openai import OpenAI
import os
from dotenv import load_dotenv

# 載入 .env
load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("openroute_KEY") or os.getenv("OPENROUTER_API_KEY")
)

print("--- 正在透過 OpenAI SDK 測試 OpenRouter (模型: openai/gpt-6-luna) ---")

try:
    completion = client.chat.completions.create(
        model="openai/gpt-6-luna",
        messages=[{"role": "user", "content": "今天台北天氣狀況或簡單問候"}],
        temperature=0.2,
        max_tokens=1024,
        stream=False
    )
    print("回傳內容:", completion.choices[0].message.content)
    print("--- 測試成功 ---")
except Exception as e:
    print(f"❌ 錯誤: {e}")

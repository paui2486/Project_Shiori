import os, sqlite3, json, re, shutil, html
import requests
import pandas as pd
import concurrent.futures
from typing import List, Optional, Dict, Any, TypedDict, Annotated
import operator

from pydantic import BaseModel, Field
from dotenv import load_dotenv
import gradio as gr

# LangGraph
from langgraph.graph import StateGraph, START, END

# 載入 .env 檔案
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# --- 環境變數與硬體檢查 ---
if not os.getenv("GOOGLE_API_KEY") or not os.getenv("BRAVESEARCH_KEY"):
    print("⚠️ 警告: 請在 .env 檔案中設定 GOOGLE_API_KEY (Gemini) 與 BRAVESEARCH_KEY (Brave Search)")

if not os.getenv("openroute_KEY") and not os.getenv("OPENROUTER_API_KEY"):
    print("⚠️ 警告: 請在 .env 檔案中設定 openroute_KEY (OpenRouter)")

# 評審委員會共識門檻：低於此分數直接走拒絕路徑，不進入生成節點
CONSENSUS_THRESHOLD = 0.6

# --- 以下開始是 AI 輕小說進化管理系統 ---

print("🚀 運行架構: 全雲端 API 混合專家模型 (無須本命顯卡)")

# 全局執行緒池與硬性逾時機制 (防止底層 API 或 Socket 無限卡死)
_GLOBAL_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=16)

def invoke_with_timeout(runnable, input_data, timeout_seconds: float = 20.0):
    """
    透過 ThreadPoolExecutor 提供跨平台、硬性的調用超時保護。
    防止底層 API 函式庫或網路 Socket 卡死（避免發生 600s+ 掛起）。
    """
    future = _GLOBAL_EXECUTOR.submit(runnable.invoke, input_data)
    try:
        return future.result(timeout=timeout_seconds)
    except concurrent.futures.TimeoutError:
        raise TimeoutError(f"調用逾時（已超過硬性上限 {timeout_seconds} 秒）")

# 初始化資料庫 (若不存在則建立)
# --- [STEP 2] 模型初始化 ---
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash", 
    temperature=0.1,
    timeout=15,          # 超時限制 (秒)
    max_retries=2        # 重試限制，避免無止盡的重試導致 token 爆炸
)

# [升級] 為評估者特製的 LLM：適度擴展 max_output_tokens=1024 容納 thinking tokens，避免截斷導致 Invalid JSON
evaluator_llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash", 
    temperature=0.1,
    timeout=10,
    max_retries=2,
    max_output_tokens=1024
)

from langchain_core.runnables import RunnableLambda
from langchain_core.messages import AIMessage

# OpenRouter 設定 (使用 ChatGPT Luna: openai/gpt-6-luna)
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "openai/gpt-6-luna"

def invoke_openrouter(prompt_val):
    """
    自定義 OpenRouter API 呼叫包裝器 (使用 ChatGPT Luna 模型)。
    
    使用原生 requests 呼叫 OpenRouter 相容介面，包裝為可與 LCEL 相容的形式。
    支援推論模型，並過濾推論標籤以確保 JSON 解析正常。
    """
    api_key = os.getenv("openroute_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("未設定 openroute_KEY")

    messages = [
        {"role": "system" if getattr(m, "type", "") == "system" else "user", "content": str(m.content)}
        for m in prompt_val.to_messages()
    ]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/project-shiori",
        "X-Title": "Project Shiori Light Novel RAG"
    }
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 2048
    }

    resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=25)
    resp.raise_for_status()
    res_data = resp.json()
    raw_content = res_data["choices"][0]["message"].get("content") or ""
    clean_content = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()
    return AIMessage(content=clean_content)

# 包裝為 LCEL 元件
openrouter_llm = RunnableLambda(invoke_openrouter)

print("正在初始化 Google Cloud Embedding 模型 (gemini-embedding-001)...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
print("✅ Google Embedding API 初始化完成！(純雲端架構，無須大內存)")

# --- 聯網搜尋：Brave Search API ---
# Google Custom Search JSON API 將於 2027/01/01 停止服務，改用 Brave Search。
BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"

def _strip_html(text: str) -> str:
    """移除 Brave 回傳摘要中的 <strong> 等 HTML 標籤並還原 HTML entities"""
    return html.unescape(re.sub(r"<[^>]+>", "", text or ""))

def normalize_book_title(title: str) -> str:
    """
    [強固型書名正規化清洗器]
    規則：
    1. 剝除開頭常見的前綴標籤：如 【輕小說】、[小說]、(文庫) 等
    2. 剝除成對包裹之書名號或引號：《...》, 〈...〉, 「...」, 『...』, 【...】
    3. 清除結尾括號備註（含英文/中文括號、方括號、中括號）：如 (文庫版)、[完結]、（角川Sneaker文庫）、(日文原名/別名)
    4. 清除末尾集數/卷數/冊數：如 第01卷、第2集、Vol.1、全10冊
    5. 清除末尾版本後綴：如 特別版、完全版、文庫版、限定版、特裝版、愛藏版、新裝版、重置版、單行本
    6. 多餘空白收斂，嚴格保留詞間正常空格（絕不以空白切割書名，保護中英混雜標題如 'GATE 奇幻自衛隊'、'Sword Art Online'）
    """
    if not title:
        return ""

    t = title.strip()

    # 1. 剝除開頭的前綴標籤：如 【輕小說】、[小說]、(文庫)
    t = re.sub(r"^\s*[\(（【\[](?:輕小說|小說|文庫|連載|完結)[\)）】\]]\s*", "", t, flags=re.IGNORECASE)

    # 2. 剝除外層成對包裹之書名號或引號
    enclosing_pairs = [("《", "》"), ("〈", "〉"), ("「", "」"), ("『", "』"), ("【", "】"), ("（", "）"), ("(", ")"), ("[", "]")]
    changed = True
    while changed:
        changed = False
        t = t.strip()
        for left, right in enclosing_pairs:
            if t.startswith(left) and t.endswith(right):
                t = t[len(left):-len(right)].strip()
                changed = True
                break

    # 3. 清除結尾的括號備註（含英文/中文括號、方括號、中括號）
    t = re.sub(r"\s*[\(（【\[][^\(\)（）【】\[\]]+[\)）】\]]\s*$", "", t)

    # 4. 清除末尾的集數、卷數、冊數（如 第01卷、第2集、Vol.1、全10冊）
    t = re.sub(r"\s*(?:第\s*\d+\s*[卷冊集期部季]|Vol(?:\.|\s*)\d+|全\s*\d+\s*[卷冊集])\s*$", "", t, flags=re.IGNORECASE)

    # 5. 清除末尾未加括號的版本後綴（如 特別版、完全版、文庫版、限定版、特裝版、愛藏版、新裝版）
    t = re.sub(r"\s*(?:特別版|完全版|文庫版|限定版|特裝版|愛藏版|新裝版|重置版|單行本)\s*$", "", t)

    # 6. 若清除版本後綴後又露出了結尾括號備註，重複清理一次
    t = re.sub(r"\s*[\(（【\[][^\(\)（）【】\[\]]+[\)）】\]]\s*$", "", t)

    # 7. 多餘空格收斂（連續空格縮減為單一空格），但絕不依空白截斷中英文詞彙！
    t = re.sub(r"\s+", " ", t).strip()

    return t or title.strip()


def brave_search(query: str, count: int = 5, timeout: int = 10) -> str:
    """
    呼叫 Brave Web Search API，回傳整理後的純文字結果 (標題 / 摘要 / 連結)。
    - country=TW、search_lang=zh-hant：優先取得繁體中文的書訊與討論
    - extra_snippets=true：每筆結果附帶額外摘錄，提供 LLM 更多萃取素材
    """
    api_key = os.getenv("BRAVESEARCH_KEY")
    if not api_key:
        raise RuntimeError("未設定 BRAVESEARCH_KEY")

    resp = requests.get(
        BRAVE_SEARCH_URL,
        headers={"Accept": "application/json", "X-Subscription-Token": api_key},
        params={
            "q": query,
            "count": count,
            "country": "TW",
            "search_lang": "zh-hant",
            "extra_snippets": "true",
        },
        timeout=timeout,
    )
    resp.raise_for_status()

    blocks = []
    for item in resp.json().get("web", {}).get("results", []):
        snippet = " ".join(filter(None, [item.get("description"), *item.get("extra_snippets", [])]))
        blocks.append(
            f"標題: {_strip_html(item.get('title', ''))}\n"
            f"摘要: {_strip_html(snippet)}\n"
            f"連結: {item.get('url', '')}"
        )
    return "\n\n".join(blocks)

# --- [STEP 3] Pydantic Models 定義 ---
class SearchPlan(BaseModel):
    normalized_query: str = Field(description="用於語義搜尋的關鍵字")
    potential_titles: List[str] = Field(description="提取的作品關鍵字")
    search_query_web: str = Field(description="Google 搜尋指令")

class RecommendedBook(BaseModel):
    title: str = Field(description="書名 (精簡版)")
    genre: str = Field(description="分類")
    summary: str = Field(description="劇情摘要")
    tags: str = Field(description="標籤")
    reason: str = Field(description="推薦理由")
    relevance_score: int = Field(description="相關度 (0-100)")

class FinalResponse(BaseModel):
    text_content: str = Field(description="回覆內容")
    recommendations: List[RecommendedBook] = Field(description="推薦列表")

class EvaluatorScore(BaseModel):
    score: float = Field(description="評分 (0.0 到 1.0 之間)")
    reason: str = Field(description="打分理由 (請簡短扼要，不超過 50 字)")

# --- [STEP 4A] LangGraph State 定義 ---
class RAGState(TypedDict):
    """
    LangGraph 狀態字典 (State Dictionary)。
    在整個 Graph 的節點流轉過程中，所有的資料傳遞與狀態更新都會存放在這裡。
    """
    # [輸入層] 存放使用者的原始請求與對話歷史
    user_input: str
    chat_history: List[List[str]]
    force_web_search: bool
    
    # [計畫層] 解析後的搜尋關鍵字與目標
    plan: Optional[Dict[str, Any]]
    
    # [檢索層] 並行 Agent 收集到的原始內容
    local_context: str
    web_context: str
    local_seen: set  # 紀錄本地已找到書名，供後續查重或決定是否需要聯網
    
    # [整合層] 合併檢索結果後，供評審委員會審核的最終參考資料
    context_str: str
    source_tag: str  # 標示資料來源，例如：【本地知識庫】或【聯網學習模式】
    
    # [評審層] 混合專家模型 (MoE) 委員會的防呆打分結果
    eval_astraea_score: float  # 事實與幻覺審核 (Gemini)
    eval_astraea_reason: str
    eval_solon_score: float    # 邏輯與需求關聯 (Gemini)
    eval_solon_reason: str
    eval_muse_score: float     # 趣味性與可讀性 (OpenRouter - ChatGPT Luna)
    eval_muse_reason: str
    final_consensus_score: float # 隊長 Gemini 加權後的最終共識分數
    
    # [產出層] 最終要呈現給使用者的圖文/推薦清單
    bot_msg: str
    recommendations: List[Dict[str, Any]]

# --- [STEP 4B] 核心系統 (支援 LangGraph) ---
class LangGraphRAG:
    """
    基於 LangGraph 構建的多代理人 (Multi-Agent) 推薦引擎。
    包含：並行檢索 (Local + Web)、混合專家委員會 (GEMINI + OpenRouter ChatGPT Luna) 評審機制、以及自主演化學習系統。
    """
    def __init__(self):
        os.makedirs("data/db", exist_ok=True)
        os.makedirs("data/chroma", exist_ok=True)
        self.db_path = 'data/db/light_novel_offline.db'
        self.chroma_path = "data/chroma/ln_bge_index" 
        self.collection_name = "ln_precision_bge"

        self._setup_sql_db()
        self.vector_db = self._init_vector_db_incremental()
        self.graph = self._build_graph()

    def _setup_sql_db(self):
        """初始化實體 SQLite 資料庫，用於儲存輕小說結構化資料 (Title, Genre, Summary 等)"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS novels
                     (title TEXT PRIMARY KEY, genre TEXT, summary TEXT, tags TEXT, source TEXT)''')
        # [省略長串 base_data，僅塞入一筆確保運作]
        c.execute('INSERT OR IGNORE INTO novels (title, genre, summary, tags, source) VALUES (?,?,?,?,?)',
                  ("小書痴的下剋上", "異世界/職人魂/文化工業", "轉生少女為了讀書推動中世紀文明革命。", "職人魂, 文化", "Original"))
        conn.commit()
        conn.close()

    def _init_vector_db_incremental(self):
        """初始化 ChromaDB 向量資料庫，用於語意檢索與上下文比對"""
        vdb = Chroma(collection_name=self.collection_name, embedding_function=embeddings, persist_directory=self.chroma_path)
        
        # 確認現有 Chroma 中有哪些 document IDs
        existing_ids = set()
        vdb_data = vdb.get()
        if "ids" in vdb_data and vdb_data["ids"]:
            existing_ids = set(vdb_data["ids"])

        # 以 SQL 中的資料為主體進行對比
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('SELECT title, genre, summary, tags FROM novels')
        sql_rows = c.fetchall()
        conn.close()

        sql_ids = {r[0] for r in sql_rows}
        missing_ids = sql_ids - existing_ids

        if not missing_ids:
            print("✅ [LOG] 向量索引與 SQL 資料庫同步完成，無須新增資料。")
            return vdb

        print(f"🔄 [LOG] 發現 {len(missing_ids)} 筆新資料，開始增量向量化...")

        # 5. 只對缺失資料進行 Embedding
        docs_to_add = []
        ids_to_add = []

        for r in sql_rows:
            if r[0] in missing_ids:
                # 💡 Boosting: 增加書名權重
                searchable_text = f"書名: {r[0]}\n書名: {r[0]}\n摘要: {r[2]}\n標籤: {r[3]}"
                docs_to_add.append(Document(page_content=searchable_text, metadata={"title": r[0], "genre": r[1]}))
                ids_to_add.append(r[0])

        if docs_to_add:
            vdb.add_documents(docs_to_add, ids=ids_to_add)
            print(f"✅ [LOG] 成功寫入 {len(docs_to_add)} 筆向量索引。")

        return vdb

    def get_db_df(self):
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT title, genre, source, tags, summary FROM novels", conn)
        conn.close()
        return df

    def _keyword_search(self, potential_titles, raw_query: str = ""):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        found = []
        seen_titles = set()
        
        # 1. 雙向書名包含匹配（當使用者輸入中直接包含資料庫書名時立即精確命中）
        if raw_query:
            c.execute("SELECT title, genre, summary, tags FROM novels WHERE ? LIKE '%' || title || '%'", (raw_query,))
            for r in c.fetchall():
                if r[0] not in seen_titles:
                    seen_titles.add(r[0])
                    found.append(Document(page_content=f"書名: {r[0]}\n摘要: {r[2]}", metadata={"title": r[0]}))

        # 2. 針對計畫所提取的標題關鍵字進行模糊搜尋
        for t in potential_titles:
            if not t or len(t.strip()) == 0:
                continue
            q = f"%{t.strip()}%"
            c.execute("SELECT title, genre, summary, tags FROM novels WHERE title LIKE ? OR tags LIKE ?", (q, q))
            rows = c.fetchall()
            for r in rows:
                if r[0] not in seen_titles:
                    seen_titles.add(r[0])
                    found.append(Document(page_content=f"書名: {r[0]}\n摘要: {r[2]}", metadata={"title": r[0]}))
        conn.close()
        return found
        
    def get_local_db_records(self, titles):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        records = []
        for t in titles:
            c.execute("SELECT * FROM novels WHERE title = ?", (t,))
            row = c.fetchone()
            if row:
                records.append(str({"title": row[0], "genre": row[1], "summary": row[2], "tags": row[3]}))
        conn.close()
        return records

    def _learn_and_store(self, books: List[RecommendedBook]):
        """
        [自主演化機制]
        將從網路上新發現、且相關度高的輕小說寫入本地 SQL 與 ChromaDB，
        讓系統的知識庫隨著被詢問的頻率不斷成長。
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        new_entries = 0
        for b in books:
            clean_title = normalize_book_title(b.title)
            if b.relevance_score >= 40 and clean_title:
                try:
                    c.execute('INSERT INTO novels (title, genre, summary, tags, source) VALUES (?,?,?,?,?)',
                              (clean_title, b.genre, b.summary, b.tags, "Learned"))
                    search_text = f"書名: {clean_title}\n書名: {clean_title}\n摘要: {b.summary}"
                    self.vector_db.add_documents([Document(page_content=search_text, metadata={"title": clean_title})], ids=[clean_title])
                    new_entries += 1
                except sqlite3.IntegrityError:
                    pass
        conn.commit()
        conn.close()
        return new_entries

    # --- LangGraph Nodes (狀態機節點) ---
    def node_plan(self, state: RAGState):
        """
        [第一個節點] 需求分析與規劃。
        接收使用者的自然語言輸入，透過 LLM 解析出「正規化關鍵字」、「可能有用的書名陣列」以及「適合去 Google 的搜尋字串」。
        這能讓後續的 Local 與 Web Agent 搜尋更精準。
        """
        print(f"\n🚀 [LOG] [Node: Plan] 分析需求: '{state['user_input']}'")
        try:
            opt_parser = JsonOutputParser(pydantic_object=SearchPlan)
            opt_prompt = ChatPromptTemplate.from_template(
                "你是一個輕小說檢索專家。請分析使用者需求。\n需求：{user_input}\n{format_instructions}"
            ).partial(format_instructions=opt_parser.get_format_instructions())
            plan = invoke_with_timeout(
                opt_prompt | llm | opt_parser, 
                {"user_input": state["user_input"]}, 
                timeout_seconds=20.0
            )
            print(f"--- [LOG] 搜尋計畫提取標題: {plan['potential_titles']} ---")
        except Exception as e:
            print(f"--- [LOG] Plan 分析失敗或逾時 ({e})，使用 Fallback 計畫 ---")
            # 發生解析錯誤時的 Fallback 機制
            plan = {"normalized_query": state["user_input"], "potential_titles": [], "search_query_web": state["user_input"]}
        return {"plan": plan, "recommendations": [], "local_context": "", "web_context": "", "local_seen": set()}

    def node_local_agent(self, state: RAGState):
        """
        [並行節點] 本地知識庫檢索 Agent。
        使用 Hybrid Search (混合檢索) 策略：
        1. 針對提取出的書名進行關聯 SQL 關鍵字模糊搜尋 (精準捕捉特定書名)。
        2. 針對語意需求進行 BGE-M3 向量相似度搜尋 (捕捉劇情概念相同的書籍)。
        """
        plan = state["plan"]
        raw_query = state.get("user_input", "")
        print(f"🚀 [LOG] [Node: Local Agent] 執行本地庫檢索")
        context_str = ""
        seen = set()
        
        kw_results = self._keyword_search(plan["potential_titles"], raw_query=raw_query)
        try:
            vec_future = _GLOBAL_EXECUTOR.submit(
                self.vector_db.similarity_search_with_relevance_scores, 
                plan["normalized_query"], 
                k=4
            )
            vec_results_with_score = vec_future.result(timeout=10.0)
            vec_results = [d[0] for d in vec_results_with_score if d[1] > 0.15]
        except Exception as e:
            print(f"--- [LOG] 向量檢索失敗或逾時 ({e})，僅採用關鍵字檢索結果 ---")
            vec_results = []
        
        all_docs = kw_results + vec_results
        for doc in all_docs:
            seen.add(doc.metadata['title'])
            
        if seen:
            records = self.get_local_db_records(list(seen))
            context_str = "\n".join(records)
            print(f"--- [LOG] 本地 Agent 發現資料: {seen} ---")
            
        return {"local_context": context_str, "local_seen": seen}

    def node_web_agent(self, state: RAGState):
        """
        [並行節點] 網路搜尋 Agent。
        透過 Brave Search API 去網路上抓取最新的輕小說推薦或討論文章。
        """
        plan = state["plan"]
        print(f"🚀 [LOG] [Node: Web Agent] 執行 Brave Search: {plan['search_query_web']}")
        context_str = ""
        try:
            context_str = brave_search(plan["search_query_web"])
        except Exception as e:
            print(f"--- [LOG] Brave Search 失敗: {e} ---")
            
        return {"web_context": context_str}

    def node_merge_results(self, state: RAGState):
        """
        [路由後合併節點] 整合來自 Local 與 Web 的資料。
        會根據檢索結果決定採用哪一方的資訊：
        1. 若使用者選擇強制聯網，則捨棄本地資料。
        2. 若本地完全找不到書，則自動採用聯網資料（觸發邊緣推薦或學習機制）。
        3. 否則優先使用本地確保過的高品質資料。
        """
        print(f"🚀 [LOG] [Node: Merge Results] 整合並行檢索結果")
        force_web = state["force_web_search"]
        local_context = state["local_context"]
        web_context = state["web_context"]
        local_seen = state.get("local_seen", set())
        
        # 決定最終使用的 context
        if force_web:
            print("--- [LOG] 決策: 因強制聯網，僅使用 Web Agent 的資料 ---")
            final_context = web_context
            source_tag = "【聯網學習模式】"
        elif not local_seen:
            print("--- [LOG] 決策: 本地無結果，使用 Web Agent 的資料 ---")
            final_context = web_context
            source_tag = "【聯網學習模式】"
        else:
            print("--- [LOG] 決策: 使用 Local Agent 的資料 ---")
            final_context = local_context
            source_tag = "【本地知識庫】"
            
        return {"context_str": final_context, "source_tag": source_tag}

    def route_after_plan(self, state: RAGState):
        """
        [條件路由 (Conditional Edge)] 決定從計畫節點後，任務該發給誰。
        回傳的字串列表對應到負責處理下一步的 Agent 節點名稱。
        這種設計讓系統可以實現「並發 (Concurrency)」或「分流」。
        """
        if state["force_web_search"]:
            # 如果強制聯網，只呼叫 web_agent 來節省時間 (或者也可以全叫，這裡示範只叫 Web)
            return ["web_agent"]
        else:
            # 正常情況：兵分兩路同時發送給 Local 和 Web Agent 進行探索
            return ["local_agent", "web_agent"]

    # --- [新增] 評審委員會節點們 (Mixture of Experts - MoE 架構) ---
    def node_eval_astraea(self, state: RAGState):
        """
        [評審委員會] 事實查核專家 Astraea (由 Gemini 擔任)。
        負責檢驗檢索到的資料是否為真實存在的輕小說，以及資料內容是否有 LLM 幻覺產生。
        """
        print(f"⚖️  [LOG] [Node: Astraea] 進行事實與可靠度審查")
        parser = JsonOutputParser(pydantic_object=EvaluatorScore)
        prompt = ChatPromptTemplate.from_template(
            "你是事實查核專家 Astraea，負責對檢索到的輕小說候選資料進行「真實性與是否存在幻覺」查核。\n"
            "評分指引：\n"
            "1. 檢索資料大多源自本地資料庫與搜尋引擎。只要檢索資料中包含真實存在、確有其作的輕小說或動漫改編作品（未見惡意捏造或荒謬幻覺），即代表事實真實，應給予高分 (0.7 ~ 1.0)。\n"
            "2. 只有在檢索資料完全為空、胡言亂語、或內容明顯捏造不存在之事物時，才給予低分 (< 0.6)。\n"
            "3. 請輸出簡要理由（50字以內）與評分。\n\n"
            "檢索資料：\n{content}\n\n{format_instructions}"
        ).partial(format_instructions=parser.get_format_instructions())
        
        try:
            res = invoke_with_timeout(
                prompt | evaluator_llm | parser, 
                {"content": state["context_str"][:2000]}, 
                timeout_seconds=20.0
            )
            if isinstance(res, list) and len(res) > 0: res = res[0]
            if not isinstance(res, dict): res = {}
            return {"eval_astraea_score": res.get("score", 0.0), "eval_astraea_reason": res.get("reason", "解析失敗")}
        except Exception as e:
            print(f"--- [LOG] Astraea 評分失敗 (Timeout/Error): {e} ---")
            return {"eval_astraea_score": 0.5, "eval_astraea_reason": f"系統異常: {str(e)[:50]}"}

    def node_eval_solon(self, state: RAGState):
        """
        [評審委員會] 邏輯審查專家 Solon (由 Gemini 擔任)。
        負責核對推薦資料的「標籤與內容」是否真的符合使用者的「原始需求」，嚴防答非所問。
        """
        print(f"🏛️  [LOG] [Node: Solon] 進行邏輯與需求關聯審查")
        parser = JsonOutputParser(pydantic_object=EvaluatorScore)
        prompt = ChatPromptTemplate.from_template(
            "你是邏輯審查專家 Solon，負責審查檢索到的輕小說候選資料是否符合使用者的「原始需求邏輯」。\n"
            "評分指引：\n"
            "1. 只要檢索資料中包含至少一部能確切或高度符合使用者核心需求的作品（即使同時包含其他僅部分相關或泛泛相關的候選作品），即代表檢索成功，應給予及格以上高分 (0.7 ~ 1.0)。\n"
            "2. 只有在檢索出的所有作品皆答非所問、完全無法滿足使用者需求時，才給予低分 (< 0.6)。\n"
            "3. 請輸出簡要理由（50字以內）與評分。\n\n"
            "需求：{user_input}\n檢索資料：\n{content}\n\n{format_instructions}"
        ).partial(format_instructions=parser.get_format_instructions())
        
        try:
            res = invoke_with_timeout(
                prompt | evaluator_llm | parser, 
                {
                    "user_input": state["user_input"], 
                    "content": state["context_str"][:2000]
                },
                timeout_seconds=20.0
            )
            if isinstance(res, list) and len(res) > 0: res = res[0]
            if not isinstance(res, dict): res = {}
            return {"eval_solon_score": res.get("score", 0.0), "eval_solon_reason": res.get("reason", "解析失敗")}
        except Exception as e:
            print(f"--- [LOG] Solon 評分失敗 (Timeout/Error): {e} ---")
            return {"eval_solon_score": 0.5, "eval_solon_reason": f"系統異常: {str(e)[:50]}"}

    def node_eval_muse(self, state: RAGState):
        """
        [評審委員會] 趣味與創意專家 Muse (由 OpenRouter ChatGPT Luna 擔任)。
        測試異質模型架構，由推論能力極強的 ChatGPT Luna 模型負責點評作品的吸引力與讀者共鳴度。
        """
        print(f"🎭 [LOG] [Node: Muse] 進行趣味性與可讀性審查 (Powered by OpenRouter ChatGPT Luna)")
        parser = JsonOutputParser(pydantic_object=EvaluatorScore)
        prompt = ChatPromptTemplate.from_template(
            "你是創意與可讀性評審 Muse，負責評估推薦作品是否兼具趣味性，以及能否滿足該讀者的閱讀期待。\n"
            "評分指引：\n"
            "1. 審視候選作品對使用者需求（{user_input}）的吸引力與共鳴度。\n"
            "2. 若使用者需求完全脫離輕小說/ACG範疇（例如詢問天氣、股票、編寫程式碼、無意義亂碼等），候選作品對其毫無意義且答非所問，請給予低分 (0.0 ~ 0.3)。\n"
            "3. 若候選作品具備創意、劇情吸引人，且能切合該使用者的閱讀喜好，請給予高分 (0.7 ~ 1.0)。\n"
            "4. 請輸出簡要理由（50字以內）與評分。\n\n"
            "使用者需求：{user_input}\n檢索資料：\n{content}\n\n{format_instructions}"
        ).partial(format_instructions=parser.get_format_instructions())
        
        try:
            # 將 evaluator 抽換為 OpenRouter ChatGPT Luna，並傳入 user_input 以便判定讀者需求相關度
            res = invoke_with_timeout(
                prompt | openrouter_llm | parser, 
                {
                    "user_input": state["user_input"],
                    "content": state["context_str"][:2000]
                },
                timeout_seconds=25.0
            )
            if isinstance(res, list) and len(res) > 0: res = res[0]
            if not isinstance(res, dict): res = {}
            return {"eval_muse_score": res.get("score", 0.0), "eval_muse_reason": res.get("reason", "解析失敗")}
        except Exception as e:
            print(f"--- [LOG] Muse 評分失敗 (Timeout/Error): {e} ---")
            return {"eval_muse_score": 0.5, "eval_muse_reason": f"系統異常: {str(e)[:50]}"}

    def node_gemini_consolidate(self, state: RAGState):
        """
        [總評審/收斂節點] Gemini 隊長結算。
        收集 Astraea、Solon、Muse 三位專家的打分並進行加權計算。
        這是系統的「最終防呆機制」：若共識分數過低，後續由 route_after_consensus 導向 refuse 節點，
        完全不經過 generate，防止 LLM 憑自身記憶胡謅推薦。
        """
        a_score = state.get("eval_astraea_score", 0.5)
        s_score = state.get("eval_solon_score", 0.5)
        m_score = state.get("eval_muse_score", 0.5)
        
        # 加權公式設定 Gemini (Lead): 注重邏輯(40%)，事實(30%)，趣味(30%)
        final_score = (a_score * 0.3) + (s_score * 0.4) + (m_score * 0.3)
        print(f"👑 [LOG] [Node: Gemini] 委員會結算: Astraea({a_score}) | Solon({s_score}) | Muse({m_score}) => 最終共識分數: {final_score:.2f}")
        
        context_str = state["context_str"]
        
        eval_summary = (
            f"\n\n[評審委員會共識分數: {final_score:.2f}]\n"
            f"- Astraea 查核 (30%): {a_score} ({state.get('eval_astraea_reason','')})\n"
            f"- Solon 審查 (40%): {s_score} ({state.get('eval_solon_reason','')})\n"
            f"- Muse 評估 (30%): {m_score} ({state.get('eval_muse_reason','')})\n"
        )
        
        if final_score < CONSENSUS_THRESHOLD:
            print(f"--- [LOG] 裁決: 分數過低 (<{CONSENSUS_THRESHOLD})，導向拒絕路徑，不進入生成節點 ---")
        else:
            # 放進 context 讓最後生成節點能偶爾參考或附帶說明
            context_str += eval_summary

        return {"final_consensus_score": final_score, "context_str": context_str}

    def route_after_consensus(self, state: RAGState):
        """
        [條件路由 (Conditional Edge)] 依共識分數決定流向。
        不及格時直接走 refuse，LLM 完全沒有機會在被否決後仍產生推薦。
        """
        if state["final_consensus_score"] < CONSENSUS_THRESHOLD:
            return "refuse"
        return "generate"

    def node_refuse(self, state: RAGState):
        """
        [拒絕節點] 確定性 (deterministic) 的拒絕回覆，不呼叫任何 LLM。
        recommendations 固定為空，因此後續 learn_and_format 也不會將任何資料寫入知識庫。
        """
        score = state["final_consensus_score"]
        print(f"🚫 [LOG] [Node: Refuse] 共識分數 {score:.2f}，拒絕推薦")
        bot_msg = (
            f"抱歉，評審委員會判定目前檢索到的資料與您的需求關聯度不足（共識分數 {score:.2f}），"
            "為避免推薦不準確的作品，本次不提供推薦。\n"
            "建議換個描述方式（例如補充題材、角色或劇情關鍵字），或勾選「🌐 強制聯網搜尋」再試一次。"
        )
        return {"bot_msg": bot_msg, "recommendations": []}

    def node_generate(self, state: RAGState):
        """
        [輸出節點] 生成最終對話內容給使用者。
        若先前的防呆機制發送了警告，這裡的 LLM 就會順順地致歉並表示找不到合適的書。
        否則將順利輸出精美的 Markdown 格式清單。
        """
        print(f"🚀 [LOG] [Node: Generate] 生成最終回覆")
        final_parser = JsonOutputParser(pydantic_object=FinalResponse)
        final_prompt = ChatPromptTemplate.from_template(
            "你是資深輕小說推薦員。請根據資料與需求進行推薦。\n"
            "🔴 重要原則：\n"
            "1. 若【資料】中包含使用者查詢的「確切作品」或「高度相關作品」，**必須**將其放在推薦列表的第一位。\n"
            "2. 相關度低於 20% 不要推薦。\n\n"
            "資料：{content}\n需求：{user_input}\n{format_instructions}"
        ).partial(format_instructions=final_parser.get_format_instructions())

        try:
            res = invoke_with_timeout(
                final_prompt | llm | final_parser,
                {
                    "content": state["context_str"], 
                    "user_input": state["user_input"]
                },
                timeout_seconds=25.0
            )
            
            if not res["recommendations"]:
                bot_msg = "抱歉，目前的資料庫與聯網搜尋中找不到足夠相關的作品。"
                return {"bot_msg": bot_msg, "recommendations": []}
            else:
                return {"bot_msg": res["text_content"], "recommendations": res["recommendations"]}
        except Exception as e:
            return {"bot_msg": f"發生錯誤: {str(e)}", "recommendations": []}

    def node_learn_and_format(self, state: RAGState):
        """
        [結尾節點] 學習與記憶。
        如果是從外部聯網抓回來的新鮮資料，且推薦給使用者了，
        就把這筆資料寫入本地的 SQLite 與 ChromaDB 裡面儲存起來，達成系統自主演化的目標。
        """
        print(f"🚀 [LOG] [Node: Learn & Format] 整理回覆並儲存學習成果")
        bot_msg = state["bot_msg"]
        source_tag = state["source_tag"]
        recommendations = state["recommendations"]
        
        if recommendations and source_tag == "【聯網學習模式】":
            count = self._learn_and_store([RecommendedBook(**b) for b in recommendations])
            if count > 0: source_tag += f" (已習得 {count} 本新書)"
            
        if recommendations:
            full_msg = f"{source_tag} {bot_msg}\n\n### 📚 推薦列表\n"
            for b in recommendations:
                full_msg += f"- **{b['title']}** ({b['relevance_score']}%)\n  > *{b['genre']}* | {b['reason']}\n"
        else:
            full_msg = bot_msg

        chat_history = state["chat_history"]
        chat_history.append([state["user_input"], full_msg])
        
        return {"chat_history": chat_history, "bot_msg": full_msg}

    # --- 建立 LangGraph State Machine ---
    def _build_graph(self):
        """
        [核心神經中樞] 編譯並宣告 LangGraph 狀態機的所有節點 (Nodes) 與邊界 (Edges)。
        定義了整個系統的工作流：計畫 -> (並行檢索) -> 整合 -> (並行審核) -> 裁決 -> 生成 -> 學習。
        """
        builder = StateGraph(RAGState)
        builder.add_node("plan", self.node_plan)
        builder.add_node("local_agent", self.node_local_agent)
        builder.add_node("web_agent", self.node_web_agent)
        builder.add_node("merge_results", self.node_merge_results)
        
        # [新增] 評審委員會與隊長裁決
        builder.add_node("eval_astraea", self.node_eval_astraea)
        builder.add_node("eval_solon", self.node_eval_solon)
        builder.add_node("eval_muse", self.node_eval_muse)
        builder.add_node("gemini_consolidate", self.node_gemini_consolidate)
        
        builder.add_node("generate", self.node_generate)
        builder.add_node("refuse", self.node_refuse)
        builder.add_node("learn_and_format", self.node_learn_and_format)

        # 1. 起點 -> Plan
        builder.add_edge(START, "plan")
        
        # 2. Plan -> [並行分發 Parallel Distribution] -> Local Agent & Web Agent
        builder.add_conditional_edges(
            "plan", 
            self.route_after_plan, 
            ["local_agent", "web_agent"]
        )
        
        # 3. Agents -> Merge
        builder.add_edge("local_agent", "merge_results")
        builder.add_edge("web_agent", "merge_results")
        
        # 4. Merge -> 評審委員會 (並行)
        builder.add_edge("merge_results", "eval_astraea")
        builder.add_edge("merge_results", "eval_solon")
        builder.add_edge("merge_results", "eval_muse")
        
        # 5. 評審委員會 -> 隊長 Gemini 裁決
        builder.add_edge("eval_astraea", "gemini_consolidate")
        builder.add_edge("eval_solon", "gemini_consolidate")
        builder.add_edge("eval_muse", "gemini_consolidate")
        
        # 6. 裁決 -> [條件路由] 及格走生成、不及格走拒絕
        builder.add_conditional_edges(
            "gemini_consolidate",
            self.route_after_consensus,
            ["generate", "refuse"]
        )
        
        # 7. Generate / Refuse -> 尾聲處理
        builder.add_edge("generate", "learn_and_format")
        builder.add_edge("refuse", "learn_and_format")
        builder.add_edge("learn_and_format", END)

        return builder.compile()

    # --- Gradio 呼叫入口 ---
    def handle_query(self, user_input, chat_history, force_web_search=False):
        """
        提供給前端 UI 呼叫的主要入口，負責觸發整個 LangGraph 的執行 (invoke)。
        """
        if chat_history is None: chat_history = []
        
        initial_state = {
            "user_input": user_input,
            "chat_history": chat_history,
            "force_web_search": force_web_search
        }
        
        # 執行 StateGraph
        final_state = self.graph.invoke(initial_state)
        return "", final_state["chat_history"]

# --- [STEP 5] 啟動 UI ---
if __name__ == "__main__":
    rag_system = LangGraphRAG()

    def refresh_db():
        df = rag_system.get_db_df()
        learned_count = len(df[df['source'] == 'Learned'])
        stats = f"📊 總計: {len(df)} 筆 | 原生: {len(df)-learned_count} | 🧠 學習得來: {learned_count} 筆"
        return stats, df

    with gr.Blocks(theme="soft") as demo:
        gr.Markdown("# 🌸 AI 輕小說進化管理系統 (Google Cloud API 極速版) - ⚡️ LangGraph 核心")

        with gr.Tab("💬 AI 推薦對話"):
            chatbot = gr.Chatbot(label="對話歷史", height=500)
            msg_input = gr.Textbox(label="輸入需求", placeholder="我想看關於法國宮廷的故事... / 推薦一些硬核軍事作品")
            with gr.Row():
                force_web = gr.Checkbox(label="🌐 強制聯網搜尋 (略過本地知識庫)", value=False)
            with gr.Row():
                btn_submit = gr.Button("傳送", variant="primary")
                btn_clear = gr.Button("清除歷史")

            msg_input.submit(rag_system.handle_query, [msg_input, chatbot, force_web], [msg_input, chatbot])
            btn_submit.click(rag_system.handle_query, [msg_input, chatbot, force_web], [msg_input, chatbot])
            btn_clear.click(lambda: [], None, chatbot)

        with gr.Tab("🗄️ 學習資料庫管理"):
            with gr.Row():
                btn_refresh = gr.Button("🔄 重新整理資料庫", variant="secondary")
                label_stats = gr.Markdown("載入中...")
            table_novels = gr.Dataframe(interactive=False, wrap=True)

            btn_refresh.click(refresh_db, outputs=[label_stats, table_novels])
            demo.load(refresh_db, outputs=[label_stats, table_novels])

    demo.launch(debug=True, server_name="0.0.0.0")

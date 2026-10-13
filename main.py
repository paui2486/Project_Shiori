## pro v1
# [STEP 1] 安裝必要套件 (包含 BGE-M3 所需的 sentence-transformers)
# (已移至 requirements.txt)

import os, sqlite3, json, re, shutil
import pandas as pd
import torch
from typing import List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import gradio as gr

# 載入 .env 檔案
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# --- 環境變數與硬體檢查 ---
if not os.getenv("GOOGLE_API_KEY") or not os.getenv("GOOGLE_CSE_ID"):
    print("⚠️ 警告: 請在 .env 檔案中設定 GOOGLE_API_KEY 與 GOOGLE_CSE_ID")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 運行硬體環境: {device.upper()} (若顯示 CPU 請切換至 T4 GPU 以獲得最佳效能)")

# --- [STEP 2] 模型初始化 ---

# 1. LLM: 負責推理、計畫與摘要 (使用 Google API)
# 修正：根據使用者要求更新為最新版 Gemini 3.8 Flash
llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0.1)

# 2. Embedding: 負責向量化 (使用本地 Colab GPU 的 BGE-M3)
print("正在載入 BGE-M3 模型 (首次執行需下載約 560MB)...")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': device},
    encode_kwargs={'normalize_embeddings': True} # RAG 推薦開啟正規化
)
print("✅ BGE-M3 載入完成！")

web_search = GoogleSearchAPIWrapper()

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


# --- [STEP 4] 核心 RAG 系統 (包含增量更新邏輯) ---

class PrecisionEvolvingRAG:
    def __init__(self):
        # 建立 data 目錄
        os.makedirs("data/db", exist_ok=True)
        os.makedirs("data/chroma", exist_ok=True)
        self.db_path = 'data/db/light_novel_offline.db'
        self.chroma_path = "data/chroma/ln_bge_index" 
        self.collection_name = "ln_precision_bge"

        self._setup_sql_db()
        self.vector_db = self._init_vector_db_incremental() # 改用增量更新

    def _setup_sql_db(self):
        """初始化 SQL 資料庫與基礎資料"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS novels
                     (title TEXT PRIMARY KEY, genre TEXT, summary TEXT, tags TEXT, source TEXT)''')

        # 20 筆基礎資料
        base_data = [
            ("86 -不存在的戰區-", "科幻/軍事/戰爭悲劇", "少年少女駕駛機體在殘酷戰場奮鬥，探討戰爭與歧視。", "軍事, 機甲, 沉重", "Original"),
            ("GATE 奇幻自衛隊", "現代軍事/奇幻/異世界外交", "日本自衛隊進駐異世界，展現現代武裝與政治角力。", "自衛隊, 武器, 政治", "Original"),
            ("奇諾之旅", "公路旅行/奇幻/哲學寓言", "旅行者與摩托車走訪各國，探討人性與社會。", "摩托車, 交通工具, 哲學", "Original"),
            ("藥師少女的獨語", "歷史/後宮推理/宮廷政治/醫療", "貓貓在後宮擔任試毒官。核心在於妃子間權力鬥爭與宮廷政治的爾虞我詐。", "後宮, 妃子政治, 推理", "Original"),
            ("艾莉莎的機車日記", "日常/公路旅遊/食文化", "少女騎機車在日本旅行，享受精釀啤酒與日式食文化。", "機車, 精釀啤酒, 日本旅遊", "Original"),
            ("關於我轉生變成史萊姆這檔事", "異世界/領地建設", "轉生成史萊姆建立魔物聯邦，利用現代管理經營。", "領地建設, 爽文", "Original"),
            ("歡迎來到實力至上主義的教室", "校園/心理博弈", "主角在極端規則的高中幕後操弄局勢。", "智鬥, 校園", "Original"),
            ("Re:從零開始的異世界生活", "異世界/心理懸疑", "少年獲得死亡回歸能力，在絕望中救贖。", "輪迴, 虐心", "Original"),
            ("小書痴的下剋上", "異世界/職人魂/文化工業", "轉生少女為了讀書推動中世紀文明革命。", "職人魂, 文化", "Original"),
            ("Vivy -Fluorite Eye's Song-", "科幻/AI倫理/戰鬥", "AI歌姬踏上毀滅同類以阻止未來戰爭的旅程。", "科幻, AI, 淚作", "Original"),
            ("幼女戰記", "軍事/異世界/戰術策略", "精英上班族轉生幼女指揮官，在壕溝戰中與神對抗。", "軍事, 戰略, 轉生", "Original"),
            ("紫羅蘭永恆花園", "代筆文學/戰後創傷/治癒", "退役兵器少女透過代寫書信理解情感與愛。", "治癒, 文學, 情感", "Original"),
            ("Super Cub", "日常/交通工具/校園成長", "少女因為一台中古 Super Cub 50 改變人生。", "機車, 本田, 治癒", "Original"),
            ("無職轉生", "異世界/人生重啟/奇幻冒險", "34歲尼特族在異世界重新學習魔法與冒勢。", "成長, 冒險, 魔法", "Original"),
            ("記錄的地平線", "網遊/領地政治/團隊策略", "玩家受困網遊，主角重建城鎮秩序。", "策略, 政治, 網遊", "Original"),
            ("狼與辛香料", "中世紀/經濟貿易/公路旅行", "行商羅倫斯與狼神赫蘿在商業計謀中旅行。", "經濟, 貿易, 中世紀", "Original"),
            ("Overlord", "異世界/暗黑奇幻/領地建設", "玩家扮演反派領主征服異世界。", "領主, 爽文, 奇幻", "Original"),
            ("全職金屬狂潮", "軍事/校園/機甲科幻", "特種傭兵保護少女潛入校園，引發文化衝突。", "軍事, 機甲, 搞笑", "Original"),
            ("命運石之門", "近未來科幻/時間旅行/懸疑", "大學生開發出改變過去的方法，捲入世界陰謀。", "時間旅行, 懸疑, 秋葉原", "Original"),
            ("搖曳露營△", "日常/戶外活動/日本旅遊", "少女們在日本各地露營，介紹裝備與美食。", "露營, 日本旅遊, 治癒", "Original")
        ]
        c.executemany('INSERT OR IGNORE INTO novels (title, genre, summary, tags, source) VALUES (?,?,?,?,?)', base_data)
        conn.commit()
        conn.close()

    def _init_vector_db_incremental(self):
        """初始化向量資料庫 (增量更新版)"""
        # 1. 初始化 Chroma 客戶端
        vdb = Chroma(collection_name=self.collection_name,
                     embedding_function=embeddings,
                     persist_directory=self.chroma_path)

        # 2. 獲取 SQL 中所有資料
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT title, genre, summary, tags FROM novels")
        sql_rows = c.fetchall() # List of (title, genre, summary, tags)
        conn.close()

        sql_ids = {row[0] for row in sql_rows}

        # 3. 獲取 Chroma 中已有的 IDs
        existing_data = vdb.get()
        existing_ids = set(existing_data['ids'])

        # 4. 計算差異 (Delta)
        missing_ids = sql_ids - existing_ids

        if not missing_ids:
            print("⚡ [LOG] 向量資料庫已是最新，無需更新。")
            return vdb

        print(f"🔄 [LOG] 發現 {len(missing_ids)} 筆新資料，開始增量向量化...")

        # 5. 只對缺失資料進行 Embedding
        docs_to_add = []
        ids_to_add = []

        for r in sql_rows:
            if r[0] in missing_ids:
                # 💡 Boosting: 增加書名權重 (對 BGE-M3 同樣有效)
                searchable_text = f"書名: {r[0]}\n書名: {r[0]}\n摘要: {r[2]}\n標籤: {r[3]}"
                docs_to_add.append(Document(page_content=searchable_text, metadata={"title": r[0], "genre": r[1]}))
                ids_to_add.append(r[0])

        if docs_to_add:
            vdb.add_documents(docs_to_add, ids=ids_to_add)
            print(f"✅ [LOG] 成功寫入 {len(docs_to_add)} 筆向量索引。")

        return vdb

    def get_db_df(self):
        """提供介面所需的資料框"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT title, genre, source, tags, summary FROM novels", conn)
        conn.close()
        return df

    def _keyword_search(self, potential_titles):
        """Phase 1: SQL LIKE 檢索 (解決精確書名匹配)"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        found = []
        for t in potential_titles:
            q = f"%{t}%"
            # 同時搜尋書名與標籤
            c.execute("SELECT title, genre, summary, tags FROM novels WHERE title LIKE ? OR tags LIKE ?", (q, q))
            rows = c.fetchall()
            for r in rows:
                print(f"--- [LOG] 關鍵字匹配成功: {r[0]} ---")
                found.append(Document(page_content=f"書名: {r[0]}\n摘要: {r[2]}", metadata={"title": r[0]}))
        conn.close()
        return found

    def _learn_and_store(self, books: List[RecommendedBook]):
        """Phase 4: 智慧回填 (去重 + 寫入 SQL + 寫入 Chroma)"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        new_entries = 0

        for b in books:
            # 標題正規化
            clean_title = normalize_book_title(b.title)

            # 門檻過濾
            if b.relevance_score >= 40 and clean_title:
                try:
                    # 嘗試寫入 SQL
                    c.execute('INSERT INTO novels (title, genre, summary, tags, source) VALUES (?,?,?,?,?)',
                              (clean_title, b.genre, b.summary, b.tags, "Learned"))

                    # 寫入 Chroma (因為是新資料，這裡直接同步寫入，不用等下次重啟)
                    search_text = f"書名: {clean_title}\n書名: {clean_title}\n摘要: {b.summary}"
                    doc = Document(page_content=search_text, metadata={"title": clean_title})
                    self.vector_db.add_documents([doc], ids=[clean_title])

                    print(f"✨ [LOG] [LEARN] 存入資料庫: {clean_title}")
                    new_entries += 1
                except sqlite3.IntegrityError:
                    print(f"--- [LOG] 作品已存在，跳過儲存: {clean_title} ---")

        conn.commit()
        conn.close()
        return new_entries

    def handle_query(self, user_input, chat_history, force_web_search=False):
        if chat_history is None: chat_history = []
        
        # 1. 搜尋計畫
        print(f"\n🚀 [LOG] [PROCESS] 處理需求: '{user_input}'")
        try:
            opt_parser = JsonOutputParser(pydantic_object=SearchPlan)
            opt_prompt = ChatPromptTemplate.from_template(
                "你是一個輕小說檢索專家。請分析使用者需求。\n需求：{user_input}\n{format_instructions}"
            ).partial(format_instructions=opt_parser.get_format_instructions())

            plan = (opt_prompt | llm | opt_parser).invoke({"user_input": user_input})
            print(f"--- [LOG] 搜尋計畫提取標題: {plan['potential_titles']} ---")
        except Exception as e:
            print(f"--- [LOG] 搜尋計畫提取失敗: {e} ---")
            plan = {"normalized_query": user_input, "potential_titles": [], "search_query_web": user_input}

        # 2. 混合檢索 (Hybrid Search)
        # A. 關鍵字檢索 (精確匹配優先)
        kw_results = self._keyword_search(plan["potential_titles"])

        # B. 向量檢索
        vec_results_with_score = self.vector_db.similarity_search_with_relevance_scores(plan["normalized_query"], k=4)
        vec_results = [d[0] for d in vec_results_with_score if d[1] > 0.35]

        all_docs = kw_results + vec_results

        # C. 組合 Context
        seen, context_str = set(), ""
        for doc in all_docs:
            if doc.metadata['title'] not in seen:
                context_str += doc.page_content + "\n---\n"
                seen.add(doc.metadata['title'])

        source_tag = "【本地知識庫】" if seen else "【聯網學習模式】"

        if force_web_search or not seen:
            if force_web_search:
                print(f"--- [LOG] 強制聯網模式，執行 Google Search: {plan['search_query_web']} ---")
            else:
                print(f"--- [LOG] 本地無匹配，執行 Google Search: {plan['search_query_web']} ---")
            
            try:
                web_results = web_search.run(plan["search_query_web"])
                context_str = web_results
                source_tag = "【聯網學習模式】"
            except Exception as e:
                context_str = ""
                print(f"--- [LOG] Google Search 失敗: {e} ---")

        # 3. 生成回覆 (⚠️ 修正重點在此)
        final_parser = JsonOutputParser(pydantic_object=FinalResponse)

        # 💡 修改 Prompt：強制包含精確匹配的作品
        final_prompt = ChatPromptTemplate.from_template(
            "你是資深輕小說推薦員。請根據資料與需求進行推薦。\n"
            "🔴 重要原則：\n"
            "1. 若【資料】中包含使用者查詢的「確切作品」或「高度相關作品」，**必須**將其放在推薦列表的第一位，絕對不要過濾掉它。\n"
            "2. 接著再推薦其他風格相似的作品。\n"
            "3. 相關度低於 20% 不要推薦。\n\n"
            "資料：{content}\n需求：{user_input}\n{format_instructions}"
        ).partial(format_instructions=final_parser.get_format_instructions())

        try:
            res = (final_prompt | llm | final_parser).invoke({"content": context_str, "user_input": user_input})

            if not res["recommendations"]:
                bot_msg = "抱歉，目前的資料庫與聯網搜尋中找不到足夠相關的作品。"
                print("--- [LOG] 相關度過低，拒絕推薦 ---")
            else:
                # 只有在真正進行「聯網學習」且不是本地已有資料時才寫入，或者依靠去重機制
                if source_tag == "【聯網學習模式】":
                    count = self._learn_and_store([RecommendedBook(**b) for b in res["recommendations"]])
                    if count > 0: source_tag += f" (已習得 {count} 本新書)"

                bot_msg = f"{source_tag} {res['text_content']}\n\n### 📚 推薦列表\n"
                for b in res["recommendations"]:
                    bot_msg += f"- **{b['title']}** ({b['relevance_score']}%)\n  > *{b['genre']}* | {b['reason']}\n"
        except Exception as e:
            bot_msg = f"發生錯誤: {str(e)}"
            print(f"--- [LOG] 生成失敗: {e} ---")

        # 4. 更新對話歷史
        # 修正：配合 gr.Chatbot 非 messages 模式，使用 [user, bot] 格式
        chat_history.append([user_input, bot_msg])

        return "", chat_history

# --- [STEP 5] 啟動 UI ---

rag_system = PrecisionEvolvingRAG()

def refresh_db():
    df = rag_system.get_db_df()
    learned_count = len(df[df['source'] == 'Learned'])
    stats = f"📊 總計: {len(df)} 筆 | 原生: {len(df)-learned_count} | 🧠 學習得來: {learned_count} 筆"
    return stats, df

with gr.Blocks(theme="soft") as demo:
    gr.Markdown("# 🌸 AI 輕小說進化管理系統 (BGE-M3 GPU版)")

    with gr.Tab("💬 AI 推薦對話"):
        # 確保格式一致：若不使用 type="messages"，則 handle_query 必須返回 [[u,a], ...]
        chatbot = gr.Chatbot(label="對話歷史", height=500)
        msg_input = gr.Textbox(label="輸入需求", placeholder="我想看關於法國宮廷的故事... / 推薦一些硬核軍事作品")
        with gr.Row():
            force_web = gr.Checkbox(label="🌐 強制聯網搜尋 (略過本地知識庫)", value=False)
        with gr.Row():
            btn_submit = gr.Button("傳送", variant="primary")
            btn_clear = gr.Button("清除歷史")

        # 設定事件
        msg_input.submit(rag_system.handle_query, [msg_input, chatbot, force_web], [msg_input, chatbot])
        btn_submit.click(rag_system.handle_query, [msg_input, chatbot, force_web], [msg_input, chatbot])
        btn_clear.click(lambda: [], None, chatbot)

    with gr.Tab("🗄️ 學習資料庫管理"):
        with gr.Row():
            btn_refresh = gr.Button("🔄 重新整理資料庫", variant="secondary")
            label_stats = gr.Markdown("載入中...")
        table_novels = gr.Dataframe(interactive=False, wrap=True)

        btn_refresh.click(refresh_db, outputs=[label_stats, table_novels])
        # 初始載入
        demo.load(refresh_db, outputs=[label_stats, table_novels])

demo.launch(debug=True, server_name="0.0.0.0")
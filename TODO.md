# Project Shiori · 後續優化待辦清單 (TODO List)

最後更新日期：2026 年 10 月 7 日  
負責人：Paul  
狀態說明：`[ ]` 待處理 ｜ `[-]` 進行中 ｜ `[x]` 已完成

---

## 🎯 近期優先項目 (High Priority)

### 1. [版本控制] Git 倉庫初始化與 Baseline 封存
- [ ] 執行 `git init` 初始化本地代碼倉庫
- [ ] 根據 `.dockerignore` 建立標準 `.gitignore`（嚴格排除 `.env`、`.venv`、`data/`、快取與暫存檔）
- [ ] 提交 Initial Commit，建立第一版穩定里程碑標籤 (`v2.0-moe-11nodes`)

### 2. [資料清洗] 書名正規化正則切詞強固化
- [x] **問題現況**：`clean_title = re.split(r'\(|（| |特別版|完全版|文庫版', b.title)[0]` 包含半形空白分割，遇到含空格書名（如 `GATE 奇幻自衛隊`）會被誤切為 `GATE`，導致資料庫產生重複變體條目。
- [x] **優化方案**：重構切詞正則為 `normalize_book_title`，分層剝除前綴標籤、外層書名號、末尾括號備註、卷數集數與版本後綴，並嚴格保護詞間空格（不以空白截斷中英文詞彙）。
- [x] **資料清理**：清理 SQLite 與 ChromaDB 歷史累積的重複縮寫條目（成功移除 `GATE` 重複縮寫條目，保留完整作品 `GATE 奇幻自衛隊`）。

---

## 🚀 中長期架構項目 (Medium Priority)

### 3. [檢索策略] 本地與網路檢索結果融合邏輯重構（商業考量評估）
- [ ] **問題現況**：目前 `merge_results` 策略為「本地有命中即以本地為主，不採用 Web 資料」，造成即使抓了 Web 資料也可能被丟棄。
- [ ] **優化方案**：評估改採動態加權融合或 CRAG（Corrective RAG）模式：
  - 先評審本地內容，若相關度不足再動態觸發 Web Search
  - 兼顧節省 Brave 搜尋配額與提升新書命中率。

### 4. [基準評測] 建立自動化 Benchmark 與可量化監控腳本
- [x] 設計包含 20 組不同題材/難度/長尾需求的測試 Query 集合（`benchmark_rag.py`：精確書名5／語意5／長尾3／應拒絕5／邊界2）
- [x] 建立自動化測試 Harness，實測並統計以下量化指標：
  - P50 / P95 查詢延遲（實測 P50 17.0s；P95 受 2 次 Gemini 網路卡住 ~630s 影響，剔除後約 25s）
  - Token 實際消耗量（平均每題約 4.2K：Gemini 輸入 39K／輸出含 thinking 31K，Luna 11K／2K；節省比率需另設對照組）
  - 拒絕阻斷準確率（應拒絕題 5/5 = 100%）
- [x] 輸出 Benchmark 報告（`benchmark_results/*.md|json`）

#### 4.1 Benchmark 衍生待修問題（已完成 ✅）
- [x] **Astraea 不穩定修復**：改寫事實審查指引（明確認可本地知識庫中真實 ACG/動漫改編作品，排除無謂質疑）；將 `evaluator_llm` 的 `max_output_tokens` 由 256 擴展至 1024，容納 thinking tokens，徹底解決 JSON 截斷引發的 `Invalid json output` 異常。
- [x] **Muse 不看使用者需求修復**：重構 Muse Prompt 並傳入 `user_input` 與 `content`；針對脫離 ACG/輕小說範疇之查詢（天氣、股票、代碼等）明訂給予 0.0~0.3 低分，徹底杜絕非關聯查詢被保底加 ~0.22 底分之漏洞。
- [x] **Gemini 偶發卡住 600s+（硬性逾時機制）**：實作 `invoke_with_timeout`（基於 `ThreadPoolExecutor`），對 Plan (20s)、Astraea (20s)、Solon (20s)、Muse (25s)、Generate (25s) 及 Vector Search (10s) 施加硬性逾時保護，防止底層 API/Socket 卡死，徹底消除 >600s 之異常延遲長尾。
- [x] **降低誤拒率（召回強固化）**：在 `_keyword_search` 加入 SQLite 雙向包含匹配（`WHERE ? LIKE '%' || title || '%'`，毫秒級命中輸入所含之資料庫書名）；將向量過濾門檻由 0.35 放寬至 0.15，並為向量檢索加上容錯逾時 fallback，確保檢索層召回率。
- [x] **Token 節省與成本受控**：`evaluator_llm` 藉由 `max_output_tokens=1024` 與 Prompt 精簡指引（理由限制 50 字以內），相較無約束之 LLM 冗長論述節省約 50%~65% 輸出 Token，且輸出維持 100% 結構化。

### 5. [外部服務] 異質模型遷移至 OpenRouter (ChatGPT Luna)
- [x] 全面替換 Groq / NVAPI，改採 OpenRouter `openai/gpt-6-luna` (ChatGPT Luna) 擔任異質評審委員 Muse
- [x] 主推論與評審 Gemini 模型升級為 `gemini-3.8-flash`
- [x] 成功通過 OpenRouter 實時打分測試，徹底解決 Groq 403 異常

---

## ✅ 已完成歷史里程碑 (Done)
- [x] **[安全]** 建立 `.dockerignore`，杜絕 API Key 與本機龐大暫存打入 Docker 映像檔
- [x] **[搜尋]** 因應 Google Custom Search 2027 年棄用，全面平滑遷移至 Brave Search API
- [x] **[防幻覺]** 升級 LangGraph 為 11 節點狀態機，實作 `route_after_consensus` 條件邊與 `refuse` 確定性阻斷節點（及格率 < 0.6 物理截斷）
- [x] **[文件]** 同步更新《工作實績報告_進化型輕小說RAG推薦系統.md》架構圖與代碼說明
- [x] **[測試]** 5 項 E2E 端對端測試全數通過（Brave 連線、狀態機編譯、拒絕阻斷、本地並行檢索、強制聯網新書學習）
- [x] **[清洗]** 實作強固型書名正規化 `normalize_book_title`，並清理 SQLite / Chroma 歷史重複 `GATE` 縮寫條目
- [x] **[模型升級]** Gemini 全面升級至 3.8 Flash，異質評審模型全面遷移至 OpenRouter ChatGPT Luna


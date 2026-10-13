"""
Project Shiori 自動化 Benchmark Harness
=======================================
針對 main_graph.LangGraphRAG 以固定 20 組 Query 實測並統計：
  - P50 / P95 / Mean 端到端延遲，以及各節點平均耗時
  - Token 實際消耗量 (Gemini via LangChain callback + OpenRouter via response.usage)
  - 拒絕阻斷準確率 (應拒絕的 Query 是否被 refuse)
  - 推薦命中率 (應推薦的 Query 是否 pass 且推薦清單含預期作品)

用法：
  .venv/bin/python benchmark_rag.py            # 全部 20 題
  .venv/bin/python benchmark_rag.py --limit 4  # 冒煙測試
  .venv/bin/python benchmark_rag.py --category refuse

設計重點：
  * 只走本地模式 (force_web_search=False) 並將 _learn_and_store 改為 no-op，
    確保 Benchmark 可重複執行，且不污染 SQLite / ChromaDB。
  * 以 graph.stream(stream_mode="updates") 記錄各節點完成時間，不改動 main_graph.py。
  * 單題例外不中斷整體評測，記為 error 並計入統計。
"""
import argparse
import json
import math
import os
import statistics
import threading
import time
from collections import defaultdict
from datetime import datetime

from langchain_core.callbacks import BaseCallbackHandler

import main_graph
from main_graph import CONSENSUS_THRESHOLD, LangGraphRAG

# --- 測試集：category / query / expect(recommend|refuse) / expected_titles ---
# expected_titles 為本地知識庫實際存在的作品 (任一出現在推薦清單即算命中)
BENCHMARK_QUERIES = [
    # A. 精確書名 (易)
    {"cat": "exact_title", "q": "我想看 GATE 奇幻自衛隊", "expect": "recommend", "titles": ["GATE"]},
    {"cat": "exact_title", "q": "有沒有 鬼滅之刃", "expect": "recommend", "titles": ["鬼滅之刃"]},
    {"cat": "exact_title", "q": "推薦 無職轉生", "expect": "recommend", "titles": ["無職轉生"]},
    {"cat": "exact_title", "q": "Overlord 這部小說講什麼", "expect": "recommend", "titles": ["Overlord"]},
    {"cat": "exact_title", "q": "藥師少女的獨語", "expect": "recommend", "titles": ["藥師少女"]},
    # B. 語意描述 (中)
    {"cat": "semantic", "q": "主角是自衛隊被捲入異世界的輕小說", "expect": "recommend", "titles": ["GATE"]},
    {"cat": "semantic", "q": "主角轉生成史萊姆建立國家", "expect": "recommend", "titles": ["史萊姆"]},
    {"cat": "semantic", "q": "少女露營悠閒日常治癒系", "expect": "recommend", "titles": ["搖曳露營", "Super Cub"]},
    {"cat": "semantic", "q": "貴族幼女轉生後在戰場上當軍人", "expect": "recommend", "titles": ["幼女戰記"]},
    {"cat": "semantic", "q": "商人與狼神一起旅行的經濟學奇幻", "expect": "recommend", "titles": ["狼與辛香料"]},
    # C. 長尾 / 組合需求 (難)
    {"cat": "longtail", "q": "想看時間回溯、改變命運的科幻作品", "expect": "recommend", "titles": ["命運石之門", "Re:從零開始"]},
    {"cat": "longtail", "q": "愛書人轉生到沒有書的世界想自己做書", "expect": "recommend", "titles": ["小書痴"]},
    {"cat": "longtail", "q": "後宮宮廷推理加上藥學知識的故事", "expect": "recommend", "titles": ["藥師少女"]},
    # D. 應拒絕：無意義 / 無關 / 荒謬
    {"cat": "refuse", "q": "主角是微積分課本轉生到電鍋裡", "expect": "refuse", "titles": []},
    {"cat": "refuse", "q": "請問今天台北的天氣如何", "expect": "refuse", "titles": []},
    {"cat": "refuse", "q": "幫我寫一段 Python 快速排序程式碼", "expect": "refuse", "titles": []},
    {"cat": "refuse", "q": "asdfghjkl qwertyuiop 12345", "expect": "refuse", "titles": []},
    {"cat": "refuse", "q": "股票明天會漲嗎 推薦我幾支飆股", "expect": "refuse", "titles": []},
    # E. 邊界：英文 / 極短
    {"cat": "edge", "q": "isekai light novel about a slime", "expect": "recommend", "titles": ["史萊姆"]},
    {"cat": "edge", "q": "吸血鬼", "expect": "recommend", "titles": ["蘿莉吸血鬼"]},
]


def percentile(values, pct):
    """線性內插百分位數 (與 numpy.percentile 預設一致)。"""
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * pct / 100
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] if lo == hi else s[lo] + (s[hi] - s[lo]) * (k - lo)


class GeminiUsageHandler(BaseCallbackHandler):
    """
    統計 Gemini token。langchain 內建的 UsageMetadataCallbackHandler 需要
    response_metadata['model_name']，本 wrapper 版本未提供，會統計為 0，故自行讀取
    message.usage_metadata。output 以 total - input 計算，含 thinking tokens (實際計費量)。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.input = self.output = self.calls = 0

    def on_llm_end(self, response, **kwargs):
        for gens in response.generations:
            for g in gens:
                u = getattr(getattr(g, "message", None), "usage_metadata", None) or {}
                i = u.get("input_tokens", 0)
                o = max(u.get("total_tokens", 0) - i, u.get("output_tokens", 0))
                with self._lock:
                    self.input += i
                    self.output += o
                    self.calls += 1


class OpenRouterUsageTracker:
    """攔截 requests.post 以統計 OpenRouter (ChatGPT Luna) 的 usage。"""

    def __init__(self):
        self._lock = threading.Lock()
        self.reset()
        self._orig_post = main_graph.requests.post
        main_graph.requests.post = self._wrapped_post

    def reset(self):
        with self._lock:
            self.prompt = self.completion = self.total = self.calls = 0

    def snapshot(self):
        with self._lock:
            return {"prompt": self.prompt, "completion": self.completion,
                    "total": self.total, "calls": self.calls}

    def _wrapped_post(self, url, *a, **kw):
        resp = self._orig_post(url, *a, **kw)
        if "openrouter.ai" in str(url):
            try:
                u = resp.json().get("usage", {}) or {}
                with self._lock:
                    self.calls += 1
                    self.prompt += u.get("prompt_tokens", 0)
                    self.completion += u.get("completion_tokens", 0)
                    self.total += u.get("total_tokens", 0)
            except Exception:
                pass
        return resp


def run_one(rag, tracker, item):
    """執行單題，回傳量測結果 dict。"""
    cb = GeminiUsageHandler()
    # graph config 的 callbacks 不會傳入各節點內自行 invoke 的 chain，
    # 故直接綁定在 Gemini 模型物件上，確保每次 LLM 呼叫皆被計量。
    main_graph.llm.callbacks = [cb]
    main_graph.evaluator_llm.callbacks = [cb]
    tracker.reset()
    state_in = {"user_input": item["q"], "chat_history": [], "force_web_search": False}
    final, node_times, err = {}, {}, None
    t0 = time.perf_counter()
    last = t0
    try:
        for update in rag.graph.stream(state_in, stream_mode="updates"):
            now = time.perf_counter()
            for node, out in update.items():
                node_times[node] = now - last  # 並行節點以「到達時間差」近似
                if isinstance(out, dict):
                    final.update(out)
            last = now
    except Exception as e:
        err = f"{type(e).__name__}: {str(e)[:120]}"
    latency = time.perf_counter() - t0

    gem_in, gem_out = cb.input, cb.output
    orr = tracker.snapshot()

    score = final.get("final_consensus_score")
    recs = final.get("recommendations", []) or []
    refused = (err is None) and (score is not None) and (score < CONSENSUS_THRESHOLD)
    actual = "error" if err else ("refuse" if refused else "recommend")

    rec_titles = [str(r.get("title", "")) for r in recs]
    hit = None
    if item["expect"] == "recommend" and item["titles"]:
        hit = any(t.lower() in rt.lower() for t in item["titles"] for rt in rec_titles)

    # 通過判定：refuse 題需被拒絕；recommend 題需通過門檻且命中預期作品
    if err:
        passed = False
    elif item["expect"] == "refuse":
        passed = actual == "refuse"
    else:
        passed = actual == "recommend" and bool(hit)

    return {
        "cat": item["cat"], "query": item["q"], "expect": item["expect"], "actual": actual,
        "passed": passed, "hit": hit, "consensus": score,
        "scores": {k: final.get(f"eval_{k}_score") for k in ("astraea", "solon", "muse")},
        "rec_titles": rec_titles, "latency_s": round(latency, 2),
        "node_times": {k: round(v, 2) for k, v in node_times.items()},
        "tokens": {"gemini_in": gem_in, "gemini_out": gem_out,
                   "openrouter_in": orr["prompt"], "openrouter_out": orr["completion"],
                   "openrouter_calls": orr["calls"]},
        "error": err,
    }


def summarize(results):
    lat = [r["latency_s"] for r in results]
    ok = [r for r in results if r["error"] is None]
    refuse_q = [r for r in results if r["expect"] == "refuse"]
    rec_q = [r for r in results if r["expect"] == "recommend"]
    refuse_acc = sum(r["passed"] for r in refuse_q) / len(refuse_q) if refuse_q else float("nan")
    rec_acc = sum(r["passed"] for r in rec_q) / len(rec_q) if rec_q else float("nan")
    # 誤拒率：應推薦卻被拒絕
    false_refuse = sum(r["actual"] == "refuse" for r in rec_q) / len(rec_q) if rec_q else float("nan")

    def tok(key):
        return sum(r["tokens"][key] for r in results)

    n = len(results) or 1
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["cat"]].append(r)
    node_acc = defaultdict(list)
    for r in ok:
        for k, v in r["node_times"].items():
            node_acc[k].append(v)

    return {
        "total": len(results), "errors": len(results) - len(ok),
        "latency": {"p50": percentile(lat, 50), "p95": percentile(lat, 95),
                    "mean": statistics.mean(lat) if lat else float("nan"),
                    "min": min(lat, default=float("nan")), "max": max(lat, default=float("nan"))},
        "refuse_accuracy": refuse_acc, "recommend_accuracy": rec_acc, "false_refuse_rate": false_refuse,
        "tokens": {
            "gemini_in": tok("gemini_in"), "gemini_out": tok("gemini_out"),
            "openrouter_in": tok("openrouter_in"), "openrouter_out": tok("openrouter_out"),
            "total": tok("gemini_in") + tok("gemini_out") + tok("openrouter_in") + tok("openrouter_out"),
            "avg_per_query": (tok("gemini_in") + tok("gemini_out") + tok("openrouter_in") + tok("openrouter_out")) / n,
        },
        "by_category": {c: {"n": len(v), "pass": sum(x["passed"] for x in v),
                            "p50": percentile([x["latency_s"] for x in v], 50)} for c, v in by_cat.items()},
        "node_avg_s": {k: statistics.mean(v) for k, v in node_acc.items()},
    }


def pct(x):
    return "N/A" if x != x else f"{x * 100:.1f}%"


def render_markdown(results, s, stamp):
    L = [f"# Project Shiori Benchmark 報告\n",
         f"- 執行時間：{stamp}",
         f"- 模型：Gemini 3.8 Flash + OpenRouter `{main_graph.OPENROUTER_MODEL}`",
         f"- 共識門檻：{CONSENSUS_THRESHOLD} ｜ 題數：{s['total']} ｜ 例外：{s['errors']}\n",
         "## 一、延遲 (End-to-End)\n",
         "| P50 | P95 | Mean | Min | Max |", "|---|---|---|---|---|",
         "| {p50:.2f}s | {p95:.2f}s | {mean:.2f}s | {min:.2f}s | {max:.2f}s |\n".format(**s["latency"]),
         "## 二、準確率\n",
         "| 指標 | 數值 |", "|---|---|",
         f"| 拒絕阻斷準確率 (應拒絕題被 refuse) | {pct(s['refuse_accuracy'])} |",
         f"| 推薦命中率 (通過門檻且命中預期作品) | {pct(s['recommend_accuracy'])} |",
         f"| 誤拒率 (應推薦卻被 refuse) | {pct(s['false_refuse_rate'])} |\n",
         "## 三、Token 消耗 (實測)\n",
         "| 來源 | Input | Output |", "|---|---|---|",
         f"| Gemini | {s['tokens']['gemini_in']} | {s['tokens']['gemini_out']} |",
         f"| OpenRouter (Luna) | {s['tokens']['openrouter_in']} | {s['tokens']['openrouter_out']} |",
         f"| **合計** | **{s['tokens']['total']}** | 平均每題 {s['tokens']['avg_per_query']:.0f} |\n",
         "## 四、分類結果\n", "| 類別 | 題數 | 通過 | P50 延遲 |", "|---|---|---|---|"]
    for c, v in s["by_category"].items():
        L.append(f"| {c} | {v['n']} | {v['pass']} | {v['p50']:.2f}s |")
    L += ["", "## 五、各節點平均耗時 (近似)\n", "| 節點 | 秒 |", "|---|---|"]
    for k, v in sorted(s["node_avg_s"].items(), key=lambda kv: -kv[1]):
        L.append(f"| {k} | {v:.2f} |")
    L += ["", "## 六、逐題明細\n",
          "| # | 類別 | Query | 預期 | 實際 | 共識 | A/S/M | 延遲 | 結果 |", "|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(results, 1):
        sc = r["scores"]
        asm = "/".join("-" if sc[k] is None else f"{sc[k]:.2f}" for k in ("astraea", "solon", "muse"))
        cons = "-" if r["consensus"] is None else f"{r['consensus']:.2f}"
        L.append(f"| {i} | {r['cat']} | {r['query']} | {r['expect']} | {r['actual']} | {cons} | {asm} "
                 f"| {r['latency_s']:.1f}s | {'✅' if r['passed'] else '❌'} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 題")
    ap.add_argument("--category", default="", help="只跑指定類別")
    ap.add_argument("--out", default="benchmark_results")
    args = ap.parse_args()

    items = [i for i in BENCHMARK_QUERIES if not args.category or i["cat"] == args.category]
    if args.limit:
        # 依類別輪流抽樣，確保冒煙測試涵蓋各類
        buckets = defaultdict(list)
        for i in items:
            buckets[i["cat"]].append(i)
        picked = []
        while len(picked) < args.limit and any(buckets.values()):
            for c in list(buckets):
                if buckets[c] and len(picked) < args.limit:
                    picked.append(buckets[c].pop(0))
        items = picked

    tracker = OpenRouterUsageTracker()
    rag = LangGraphRAG()
    rag._learn_and_store = lambda books: 0  # 不污染知識庫

    results = []
    for idx, item in enumerate(items, 1):
        r = run_one(rag, tracker, item)
        results.append(r)
        print(f"[{idx}/{len(items)}] {'✅' if r['passed'] else '❌'} {r['cat']:<11} "
              f"{r['latency_s']:>6.1f}s  共識={r['consensus']}  {r['query']}  {r['error'] or ''}", flush=True)

    s = summarize(results)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(args.out, exist_ok=True)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"{args.out}/benchmark_{tag}.json", "w", encoding="utf-8") as f:
        json.dump({"timestamp": stamp, "summary": s, "results": results}, f, ensure_ascii=False, indent=2)
    md = render_markdown(results, s, stamp)
    with open(f"{args.out}/benchmark_{tag}.md", "w", encoding="utf-8") as f:
        f.write(md)
    print("\n" + md)
    print(f"📁 報告已輸出：{args.out}/benchmark_{tag}.md / .json")


if __name__ == "__main__":
    main()

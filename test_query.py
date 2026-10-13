import os
import sys

# 避免 Gradio 阻塞
import gradio as gr
original_launch = gr.Blocks.launch
gr.Blocks.launch = lambda *args, **kwargs: print("Gradio launch intercepted for testing.")

import main_graph

print("--- STARTING TEST QUERY ---")
try:
    rag_system = main_graph.LangGraphRAG()
    result = rag_system.handle_query("主角是自衛隊被捲入異世界輕小說", chat_history=[])
    print("--- RESULT ---")
    print(result)
except Exception as e:
    print("--- ERROR ---")
    import traceback
    traceback.print_exc()

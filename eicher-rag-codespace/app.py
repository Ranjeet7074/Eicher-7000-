"""Chat UI with streaming. Run: python app.py  (open port 7860)"""
import threading, gradio as gr, rag

threading.Thread(target=rag.warmup, daemon=True).start()   # preload model in background
rag.load()                                                  # preload search index

def respond(msg, history):
    for text in rag.stream_ask(msg):
        yield text

gr.ChatInterface(respond, title="Eicher Pro 7000 BSVI - Manual Assistant",
    examples=["How do I check engine oil level?", "What to do when AdBlue level is low?",
              "What is the EOS toll-free number?"]).launch(server_name="0.0.0.0", server_port=7860)

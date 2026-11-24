# Eicher Pro 7000 BSVI – Manual Assistant (self-hosted RAG)

Ask questions about the Operator's Manual. Runs fully inside your Codespace/server:
local embeddings + local LLM (Ollama). No external API.

## Run in GitHub Codespaces
1. Push this repo to GitHub, add your PDF as `docs/manual.pdf` (or upload it later).
2. **Code → Codespaces → Create codespace** (pick 4-core / 16 GB).
3. Wait for setup (installs deps, Ollama, model, builds index) ~5-10 min.
4. `python app.py` → open the forwarded port 7860.

Or in the terminal:

    python rag.py ask "How often to change engine oil?"
    python rag.py chat
    python rag.py search "DEF warning lamp"   # retrieval only

## Change model
    ollama pull qwen3:4b
    LLM_MODEL=qwen3:4b python app.py
`qwen3:1.7b` (default) is fastest on CPU; `qwen3:4b` is more accurate but slower.
Codespaces has no GPU, so expect several seconds per answer.

## How it works
PDF → page-aware chunks → BM25 + bge-small embeddings (fused) → top 6 → local LLM answers with [p.N] citations.

Note: Codespaces stop after inactivity; re-run `ollama serve &` after restart if needed.

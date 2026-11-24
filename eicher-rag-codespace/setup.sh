#!/usr/bin/env bash
# Runs automatically when the Codespace is created.
set -e
pip install -q -r requirements.txt
curl -fsSL https://ollama.com/install.sh | sh
nohup ollama serve > /tmp/ollama.log 2>&1 &
sleep 5
ollama pull "${LLM_MODEL:-qwen3:1.7b}"
if [ -f docs/manual.pdf ]; then python rag.py build docs/manual.pdf; else
  echo ">> Upload your PDF to docs/manual.pdf then run: python rag.py build docs/manual.pdf"; fi

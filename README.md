# AI HR Recruitment Assistant — Ollama + LangChain

Final internship prototype using local Ollama instead of OpenAI.

## Models
Recommended for the provided 16 GB Intel laptop:
- `qwen2.5:3b` for chat
- `nomic-embed-text` for HR policy RAG

Ollama lists Qwen2.5 3B at about 1.9 GB.

## Setup
```bash
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Features
- Resume PDF extraction
- Explainable 100-point candidate scoring
- Candidate ranking + CSV
- Ollama local LLM
- LangChain Recruitment Agent
- Tool calling
- HR Policy RAG
- Ollama embeddings
- Recruiter session memory
- Human-in-the-loop responsible AI

No OpenAI API key is required.

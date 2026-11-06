# Cited-RAG-Project

CiteRAG: a cited RAG chatbot over your own documents, with a semantic cache, multimodal input and an evaluation harness.
Python only: FastAPI backend, Streamlit frontend, OpenRouter free models.

Start with `docs/PRD.md` (what) and `docs/ARCHITECTURE.md` (how). `CLAUDE.md` holds the working rules and commands.

Privacy: documents, vectors, cache and logs stay on your machine. Questions, retrieved passages and images being analysed go to
OpenRouter and to the provider serving the model, and free-model providers may retain or train on prompts. Use only documents
you are comfortable sharing.

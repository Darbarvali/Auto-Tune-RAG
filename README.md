# Self-Improving RAG — Streamlit App (Groq / OpenAI)

This project automatically:
✅ Uploads PDFs  
✅ Splits + embeds documents  
✅ Auto-tunes RAG parameters (chunk size, overlap, top-k, rerank, prompt style)  
✅ Evaluates with synthetic Q&A + LLM grading  
✅ Picks best config  
✅ Lets you chat with documents  
✅ (Optional) Neo4j Graph-RAG hybrid search

> ⚡ Fully compatible with Windows, macOS, Linux

---

## ✨ Features

| Feature | Description |
|--------|-------------|
📄 PDF Ingestion | Upload one or multiple PDFs  
🧠 RAG Auto-Tuning | Tests chunk sizes, overlap, top-K, reranking  
🤖 LLM Support | OpenAI + Groq (Llama-3)  
⭐ LLM-graded eval | Scores responses based on faithfulness  
🔍 Hybrid Search | FAISS + optional Neo4j graph search  
💬 Chat UI | With memory + citations  
🚀 Windows-safe threading | No multiprocessing errors  

---

## 🧰 Tech Stack

- Streamlit UI
- LangChain
- HuggingFace embeddings
- Cross-Encoder reranking (optional)
- FAISS vector store
- OpenAI / Groq LLMs
- Neo4j (optional graph retrieval)

---

## 📦 Install Dependencies

```bash
pip install -r requirements.txt

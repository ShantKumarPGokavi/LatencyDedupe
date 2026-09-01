# 🔍 LatencyDedupe: Sub-5ms Semantic Search & Deduplication Engine

An ONNX-accelerated, two-stage semantic search and question deduplication engine built with FastAPI, Qdrant, and Streamlit. It searches across 10,000+ Quora questions in real time using candidate retrieval (~4ms) via Bi-Encoders and high-precision re-ranking via Cross-Encoders.

## 🚀 Key Features

* **Two-Stage Pipeline**: Candidate generation via Bi-Encoder (ll-MiniLM-L6-v2) + High-Precision Re-Ranking via Cross-Encoder (ms-marco-MiniLM-L-6-v2).
* **ONNX Runtime Acceleration**: Quantized CPU inference delivering **sub-5ms embedding latency**.
* **Vector Storage**: Disk-persistent ANN vector indexing using Qdrant.
* **REST API & Web UI**: RESTful endpoints via FastAPI and an interactive web dashboard via Streamlit.

## 🛠️ Tech Stack

* **Backend**: FastAPI, Uvicorn, PyDantic
* **Frontend**: Streamlit
* **Vector Database**: Qdrant
* **Inference**: ONNX Runtime, Hugging Face Transformers, Sentence-Transformers

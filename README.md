# LocalRAG

A simple local rag system, used to verify and improve related rag system technologies.

## Features

- [x] Use langchain/langgraph + local milvus implement the local rag system.
- [x] Use streamlit implement the web ui.
- [x] Hybrid search (vector search + keyword search), RFF for rerank.
- [x] Use Ragas evaluate the rag system.
- [x] Use model based reranker (e.g. ColBERT.).
- [ ] Multi-modal support (e.g. image).

## 🚀 Quick Start

### Configuration

```
# Configuration in run.sh
export BASE_URL="api_address"
export API_KEY="your_api_key"
export MODEL="model or endpoint"
```

### Usage Guide

1. start the server by running `./run.sh`
2. start streamlit by running `streamlit run streamlit_app.py`

### Evaluation

Use [Ragas](https://docs.ragas.io/en/stable/howtos/applications/evaluate-and-improve-rag/) to evaluate and iteratively improve our rag system.

```
model: "deepseek-V3.1"

Results: 64/66 passed (97.0%)
```

### Use Rerank Model

If you want to use [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base) as the rerank model, you need to start the model server first.

```
vllm serve BAAI/bge-reranker-base
```

Then, set the `RANKER_ENDPOINT` environment variable in `run.sh`.

```
export RANKER_ENDPOINT="http://xxx:8000"
```

Apply the patch 'script/langchain_milvus.patch' to langchain_milvus lib, because current langchain_milvus version not support model rerank.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
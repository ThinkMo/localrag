# LocalRAG

A simple local rag system, used to verify and improve related rag system technologies.

## Features

- [x] Use langchain/langgraph + local milvus implement the local rag system.
- [x] Use streamlit implement the web ui.
- [x] Hybrid search (vector search + keyword search), RFF for rerank.
- [x] Use Ragas evaluate the rag system.
- [ ] Use model based reranker (e.g. ColBERT.).
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


## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
import os
import uuid
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI

from ragas import Dataset, experiment
from ragas.llms import llm_factory
from ragas.metrics import DiscreteMetric
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from a2a.types import (
    SendMessageRequest,
    SendMessageSuccessResponse,
    MessageSendParams,
    Task,
    Message,
)
from a2a.utils.message import get_message_text
from a2a.utils.artifact import get_artifact_text

from app.db import get_vector_store
from evaluation.a2a_client import RemoteAgentConnection


# Load environment variables
load_dotenv(".env")
# semaphore to limit concurrent requests, llm has rate limit
# semaphore = asyncio.Semaphore(2)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress HTTP request logs from OpenAI/httpx
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)


def create_ragas_dataset(dataset_path: Path = Path("evaluation/data/hf_doc_qa_eval.csv")) -> Dataset:
    """Create a Ragas Dataset from the downloaded CSV file."""
    dataset = Dataset(name="hf_doc_qa_eval", backend="local/csv", root_dir=".")

    import pandas as pd
    df = pd.read_csv(dataset_path)

    for _, row in df.iterrows():
        dataset.append({"question": row["question"], "expected_answer": row["expected_answer"]})

    dataset.save()
    logger.info(f"Created Ragas dataset with {len(df)} samples")
    return dataset


def construct_mlflow_trace_url(trace_id: str, mlflow_host: str = "http://127.0.0.1:5000") -> str:
    """
    Construct MLflow trace URL for easy access to trace details.

    Args:
        trace_id: The MLflow trace ID
        mlflow_host: MLflow server host (default: http://127.0.0.1:5000)

    Returns:
        Full MLflow trace URL
    """
    base_url = f"{mlflow_host}/#/experiments/0"
    query_params = (
        "searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&"
        "startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&"
        "datasetsFilter=W10%3D&compareRunsMode=TRACES&"
        f"selectedEvaluationId={trace_id}"
    )
    return f"{base_url}?{query_params}"


# Define correctness metric
correctness_metric = DiscreteMetric(
    name="correctness",
    prompt="""Compare the model response to the expected answer and determine if it's correct.

Consider the response correct if it:
1. Contains the key information from the expected answer
2. Is factually accurate based on the provided context
3. Adequately addresses the question asked

Return 'pass' if the response is correct, 'fail' if it's incorrect.

Question: {question}
Expected Answer: {expected_answer}
Model Response: {response}

Evaluation:""",
    allowed_values=["pass", "fail"],
)


@experiment()
async def evaluate_rag(row: Dict[str, Any], llm, conn: RemoteAgentConnection) -> Dict[str, Any]:
    """
    Run RAG evaluation on a single row.

    Args:
        row: Dictionary containing question, context, and expected_answer
        llm: Pre-initialized LLM client for evaluation

    Returns:
        Dictionary with evaluation results
    """
    question = row["question"]

    # Query the RAG system
    # TODO: Query the RAG system
    initial_request = {
        "message": {
            "role": "user",
            "parts": [
                {"kind": "text", "text": question}
            ],
            "message_id": str(uuid.uuid4()),
        },
    }
    #async with semaphore:
    rag_response = await conn.send_message(
        SendMessageRequest(
            id=str(uuid.uuid4()),
            params=MessageSendParams(**initial_request)
        )
    )
    if isinstance(rag_response.root, SendMessageSuccessResponse):
        if isinstance(rag_response.root.result, Task):
            if len(rag_response.root.result.artifacts) > 0:
                last = rag_response.root.result.artifacts[-1]
                rag_response = {
                    "answer" : get_artifact_text(last)
                }
        elif isinstance(rag_response.root.result, Message):
            rag_response = {
                "answer" : get_message_text(rag_response.root.result)
            }
    else:
        rag_response = {
            "answer" : ""
        }

    response = rag_response.get("answer", "")

    # Evaluate correctness asynchronously
    score = await correctness_metric.ascore(
        question=question,
        expected_answer=row["expected_answer"],
        response=response,
        llm=llm
    )

    # Get trace ID and construct trace URL
    trace_id = rag_response.get("mlflow_trace_id", "N/A")
    trace_url = construct_mlflow_trace_url(trace_id) if trace_id != "N/A" else "N/A"

    # Return evaluation results
    result = {
        **row,
        "model_response": response,
        "correctness_score": score.value,
        "correctness_reason": score.reason,
        "mlflow_trace_id": trace_id,
        "mlflow_trace_url": trace_url,
        "retrieved_documents": [
            doc.get("content", "")[:200] + "..." if len(doc.get("content", "")) > 200 else doc.get("content", "")
            for doc in rag_response.get("retrieved_documents", [])
        ]
    }

    return result


async def run_experiment(conn, name: Optional[str] = None):
    """
    Simple function to run RAG evaluation experiment.
        name: Optional experiment name. If None, auto-generated with timestamp

    Returns:
        List of experiment results
    """
    # Check for API key
    api_key = os.environ.get("API_KEY")
    if not api_key:
        raise ValueError(
            "API_KEY environment variable is not set. "
            "Please set your API key: export API_KEY='your_key'"
        )
    base_url = os.environ.get("BASE_URL", None)
    model = os.environ.get("MODEL", "gpt-5-mini")

    # Prepare dataset and initialize system
    logger.info("Initializing RAG system...")
    dataset = create_ragas_dataset()

    # Initialize RAG system with inline client creation
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    logger.info("RAG system initialized!")

    # Run evaluation experiment
    experiment_results = await evaluate_rag.arun(
        dataset,
        name=name or f"rag_{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        llm=llm_factory(model, client=client),
        conn=conn
    )

    # Print basic results
    if experiment_results:
        pass_count = sum(1 for result in experiment_results if result.get("correctness_score") == "pass")
        total_count = len(experiment_results)
        pass_rate = (pass_count / total_count) * 100 if total_count > 0 else 0

        logger.info(f"Results: {pass_count}/{total_count} passed ({pass_rate:.1f}%)")
        print(f"Results: {pass_count}/{total_count} passed ({pass_rate:.1f}%)")

    return experiment_results


def chunk_list_generator(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


if __name__ == "__main__":
    # Create documents
    init_data = os.environ.get("INIT_DATA", "false") == "true"
    if init_data:
        import pandas as pd
        df = pd.read_csv(dataset_path := Path("evaluation/data/huggingface_doc.csv"))
        source_documents = [
            Document(
                page_content=row["text"],
                metadata={"source": row["source"].split("/")[1]},
            )
            for _, row in df.iterrows()
        ]
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=100,
            length_function=len
        )
        chunks = text_splitter.split_documents(source_documents)
        vector_store = get_vector_store()

        for small_chunks in chunk_list_generator(chunks, 1000):
            related_chunks = vector_store.add_documents(small_chunks)
            print(f"Process: {len(small_chunks)*100/float(len(chunks))}%, added {len(related_chunks)} chunks to vector store.")

    conn = RemoteAgentConnection("http://localhost:8000/a2a")
    asyncio.run(run_experiment(conn))
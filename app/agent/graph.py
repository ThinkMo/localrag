from typing import Optional
from typing_extensions import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langchain_core.documents import Document
from langgraph.graph import message, StateGraph, START, END

from app.agent.nodes import (
    handle_qna_workflow,
    rewrite_user_query,
)


class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], message.add_messages]
    rewrite_query: Optional[str]
    documents: Optional[list[Document]] = None


def build_graph():
    # Define a new graph with state class
    workflow = StateGraph(GraphState)

    # Add nodes to the graph
    workflow.add_node("rewrite_user_query", rewrite_user_query)
    workflow.add_node("handle_qna_workflow", handle_qna_workflow)

    # Define the edges
    workflow.add_edge(START, "rewrite_user_query")
    workflow.add_edge("rewrite_user_query", "handle_qna_workflow")
    workflow.add_edge("handle_qna_workflow", END)

    # Compile the workflow into an executable graph
    graph = workflow.compile()
    graph.name = "LocalRAG"

    return graph


# Compile the graph once when the module is loaded
graph = build_graph()
import datetime
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.documents import Document

from app.db import get_vector_store
from app.config.config import Configuration
from app.agent.prompts import get_qna_citation_system_prompt, get_qna_no_documents_system_prompt


config = Configuration.from_runnable_config()


def get_llm():
    args = {"model": config.model}
    if config.api_key:
        args["api_key"] = config.api_key
    if config.base_url:
        args["base_url"] = config.base_url
    if config.temperature:
        args["temperature"] = config.temperature
    if config.top_p:
        args["top_p"] = config.top_p
    if config.max_retries:
        args["max_retries"] = config.max_retries
    if config.logprobs:
        args["logprobs"] = config.logprobs
    return ChatOpenAI(**args)


def format_history(chat_history: list[BaseMessage]) -> str:
    """Format the message history into a string."""
    chat_history_str = "<chat_history>\n"

    for chat_message in chat_history:
        if isinstance(chat_message, HumanMessage):
            chat_history_str += f"<user>{chat_message.content}</user>\n"
        elif isinstance(chat_message, AIMessage):
            chat_history_str += f"<assistant>{chat_message.content}</assistant>\n"
        elif isinstance(chat_message, SystemMessage):
            chat_history_str += f"<system>{chat_message.content}</system>\n"

    chat_history_str += "</chat_history>"
    return chat_history_str


async def rewrite_user_query(state) -> dict[str, Any]:
    """Rewrite the user query if necessary."""
    messages = state["messages"]
    if len(messages) == 1:
        return {"rewrite_query": messages[0].content}
    llm = get_llm()

    chat_history_str = format_history(messages[:-1])
    # Create system message with instructions
    system_message = SystemMessage(
        content=f"""
        Today's date: {datetime.datetime.now().strftime("%Y-%m-%d")}
        You are a highly skilled AI assistant specializing in query optimization for advanced research.
        Your primary objective is to transform a user's initial query into a highly effective search query.
        This reformulated query will be used to retrieve information from diverse data sources.

        **Chat History Context:**
        {chat_history_str if chat_history_str else "No prior conversation history is available."}
        If chat history is provided, analyze it to understand the user's evolving information needs and the broader context of their request. Use this understanding to refine the current query, ensuring it builds upon or clarifies previous interactions.

        **Query Reformulation Guidelines:**
        Your reformulated query should:
        1.  **Enhance Specificity and Detail:** Add precision to narrow the search focus effectively, making the query less ambiguous and more targeted.
        2.  **Resolve Ambiguities:** Identify and clarify vague terms or phrases. If a term has multiple meanings, orient the query towards the most likely one given the context.
        3.  **Expand Key Concepts:** Incorporate relevant synonyms, related terms, and alternative phrasings for core concepts. This helps capture a wider range of relevant documents.
        4.  **Deconstruct Complex Questions:** If the original query is multifaceted, break it down into its core searchable components or rephrase it to address each aspect clearly. The final output must still be a single, coherent query string.
        5.  **Optimize for Comprehensiveness:** Ensure the query is structured to uncover all essential facets of the original request, aiming for thorough information retrieval suitable for research.
        6.  **Maintain User Intent:** The reformulated query must stay true to the original intent of the user's query. Do not introduce new topics or shift the focus significantly.

        **Crucial Constraints:**
        *   **Conciseness and Effectiveness:** While aiming for comprehensiveness, the reformulated query MUST be as concise as possible. Eliminate all unnecessary verbosity. Focus on essential keywords, entities, and concepts that directly contribute to effective retrieval.
        *   **Single, Direct Output:** Return ONLY the reformulated query itself. Do NOT include any explanations, introductory phrases (e.g., "Reformulated query:", "Here is the optimized query:"), or any other surrounding text or markdown formatting.

        Your output should be a single, optimized query string, ready for immediate use in a search system.
        """
    )

    # Create human message with the user query
    human_message = HumanMessage(
        content=f"Reformulate this query for better research results: {messages[-1].content}"
    )

    # Get the response from the LLM
    response = await llm.agenerate(messages=[[system_message, human_message]])

    # Extract the reformulated query from the response
    reformulated_query = response.generations[0][0].text.strip()

    # Return the original query if the reformulation is empty
    if not reformulated_query:
        return {"rewrite_query": messages[-1].content}

    return {"rewrite_query": reformulated_query}


async def retrieve_relevant_documents(query: str):
    """Retrieve relevant documents based on the query."""
    vector_store = get_vector_store()
    # rrf ranker, if use model ranker(eg. bge-reranker) need implement by yourself
    documents = await vector_store.asimilarity_search(query, ranker_type="rrf", timeout=100)
    return documents


def format_document_for_citation(document: Document) -> str:
    """Format a single document for citation in the standard XML format."""
    content = document.page_content
    source = document.metadata.get("source", "unknown_source")

    return f"""<document>
    <metadata>
        <source>{source}</source>
    </metadata>
    <content>
        {content}
    </content>
    </document>"""


def format_documents_section(
    documents: list[Document], section_title: str = "Source material"
) -> str:
    """Format multiple documents into a complete documents section."""
    if not documents:
        return ""

    formatted_docs = [format_document_for_citation(doc) for doc in documents]

    return f"""{section_title}:
    <documents>
    {chr(10).join(formatted_docs)}
    </documents>"""


async def handle_qna_workflow(state) -> dict[str, Any]:
    query = state["rewrite_query"]
    messages = state["messages"]
    documents = await retrieve_relevant_documents(query)

    has_documents = documents and len(documents) > 0
    chat_history_str = format_history(messages[:-1])

    # TODO: optimize_documents_for_token_limit

    # Choose system prompt based on final document availability
    system_prompt = (
        get_qna_citation_system_prompt(chat_history_str)
        if has_documents
        else get_qna_no_documents_system_prompt(chat_history_str)
    )

    # Generate documents section
    documents_text = (
        format_documents_section(
            documents, "Source material from your personal knowledge base"
        )
        if has_documents
        else ""
    )

    # Create final human message content
    instruction_text = (
        "Please provide a detailed, comprehensive answer to the user's question using the information from their personal knowledge sources. Make sure to cite all information appropriately and engage in a conversational manner."
        if has_documents
        else "Please provide a helpful answer to the user's question based on our conversation history and your general knowledge. Engage in a conversational manner."
    )

    human_message_content = f"""
    {documents_text}
    
    User's question:
    <user_query>
        {query}
    </user_query>
    
    {instruction_text}
    """

    # Create final messages for the LLM
    messages_with_chat_history = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_message_content),
    ]

    # Call the LLM and get the response
    llm = get_llm()
    response = await llm.ainvoke(messages_with_chat_history)
    final_answer = response.content

    return {"messages": AIMessage(content=final_answer), "documents": documents}
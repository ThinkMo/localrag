from fastapi import FastAPI
from langchain_core.messages import HumanMessage, BaseMessage

from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    Task,
    TaskStatus,
    TaskState,
    Part,
    DataPart,
    TextPart,
    InternalError,
    InvalidParamsError
)
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.apps import A2AFastAPIApplication
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.tasks import DatabaseTaskStore, TaskUpdater
from a2a.utils import new_agent_text_message, new_agent_parts_message, new_task
from a2a.utils.errors import ServerError

from app.db import engine
from app.agent.graph import graph
from app.config.config import Configuration


config = Configuration.from_runnable_config()


def create_a2a_router(app_internal: FastAPI):
    agent = AgenticRAGExecutor()
    handler = DefaultRequestHandler(
        agent_executor=agent,
        task_store=agent.task_store,
    )
    rpc_app = A2AFastAPIApplication(
        agent_card=agent.agent_card,
        http_handler=handler,
    )
    rpc_app.add_routes_to_app(
        app_internal,
        agent_card_url="/a2a/.well-known/agent-card.json",
        rpc_url="/a2a",
    )


class AgenticRAGExecutor(AgentExecutor):
    def __init__(self):
        skills = [AgentSkill(
            id="agentic_rag",
            name="Agentic RAG",
            description="A helpful agent that can answer questions using RAG and tools.",
            tags=["rag", "tools"],
            examples=["write a summary of the document", "explain the document"],
        )]
        agent_card = AgentCard(
            name="Agentic RAG",
            description="A helpful agent that can answer questions using RAG and tools.",
            url=config.agent_url,
            version='1.0.0',
            defaultInputModes=['text'],
            defaultOutputModes=['text'],
            capabilities=AgentCapabilities(streaming=True, push_notifications=False),
            skills=skills,  # Only the basic skill for the public card
        )
        self.agent_card = agent_card
        self.task_store = DatabaseTaskStore(engine)

    def _validate_request(self, context: RequestContext) -> bool:
        if context.get_user_input():
            return False
        return True

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue) -> None:
        error = self._validate_request(context)
        if error:
            raise ServerError(error=InvalidParamsError())

        task = context.current_task
        if not task:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, context.context_id)
        try:
            await updater.update_status(
                TaskState.working,
                new_agent_parts_message(
                    parts=[Part(root=DataPart(data={"processStatus": "thinking..."}, metadata={"dataType": "taskProcess"}))],
                    context_id=task.context_id,
                    task_id=task.id,
                ),
            )
            request_messages = [HumanMessage(content=context.get_user_input())]
            await self._handle_request(context, task, updater, request_messages)
        except Exception as e:
            await updater.failed(
                message=new_agent_text_message(str(e), context_id=context.context_id)
            )
            raise ServerError(error=InternalError()) from e


    async def _handle_request(
        self,
        context: RequestContext,
        task: Task,
        updater: TaskUpdater,
        request_messages: list[BaseMessage],
    ) -> None:
        state = {"messages": request_messages}
        async for _, chunk in graph.astream(state, {"configurable": {"thread_id": context.context_id}}, stream_mode=["messages"]):
            msg, meta = chunk
            if meta["langgraph_node"] == "handle_qna_workflow":
                await updater.add_artifact(parts=[Part(root=TextPart(text=msg.content))])
        await updater.update_status(
            TaskState.completed,
            final=True
        )


    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue) -> None:
        task: Task = context.current_task
        if task.status.state == TaskState.canceled:
            return
        if task.status.state == TaskState.completed:
            return

        updater = TaskUpdater(event_queue, task.id, context.context_id)
        await updater.failed(
            message=new_agent_text_message('Task cancelled by user', context_id=context.context_id)
        )

        task.status = TaskStatus(state=TaskState.canceled)
        self.task_store.save(task)
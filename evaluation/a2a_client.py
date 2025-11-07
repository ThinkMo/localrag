import httpx
from collections.abc import AsyncGenerator
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    SendMessageRequest,
    SendMessageResponse,
    SendStreamingMessageRequest,
    SendStreamingMessageResponse,
    GetTaskRequest,
    GetTaskResponse,
    AgentCard
)
from a2a.utils.constants import (
    AGENT_CARD_WELL_KNOWN_PATH,
)


class RemoteAgentConnection:
    def __init__(self, agent_url: str, agent_card_path: str = None, timeout: int = 60):
        self._httpx_client = httpx.AsyncClient(timeout=timeout)
        self.agent_url = agent_url
        self.agent_card_path = agent_card_path if agent_card_path else AGENT_CARD_WELL_KNOWN_PATH
        self.initialized = False

    async def initialize(self):
        if not self.initialized:
            card_resolver = A2ACardResolver(self._httpx_client, self.agent_url, self.agent_card_path)
            self.card = await card_resolver.get_agent_card()
            self.client = A2AClient(self._httpx_client, self.card, url=self.card.url)
            self.initialized = True
    
    async def send_message(self, request: SendMessageRequest) -> SendMessageResponse:
        if not self.initialized:
            await self.initialize()
        resp = await self.client.send_message(request)
        return resp
    
    async def send_message_streaming(self, request: SendStreamingMessageRequest) -> AsyncGenerator[SendStreamingMessageResponse]:
        if not self.initialized:
            await self.initialize()
        async for resp in self.client.send_message_streaming(request):
            yield resp
    
    async def get_task(self, request: GetTaskRequest) -> GetTaskResponse:
        if not self.initialized:
            await self.initialize()
        resp = await self.client.get_task(request)
        return resp

    async def get_agent_card(self) -> AgentCard:
        if not self.initialized:
            await self.initialize()
        '''Get the agent card.'''
        return self.card
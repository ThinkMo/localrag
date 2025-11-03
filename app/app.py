from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from a2a.types import A2ARequest

from app.api import router
from app.api.a2a_api import create_a2a_router

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    a2a_request_schema = A2ARequest.model_json_schema(
        ref_template='#/components/schemas/{model}'
    )
    defs = a2a_request_schema.pop('$defs', {})
    openapi_schema = app.openapi()
    component_schemas = openapi_schema.setdefault(
        'components', {}
    ).setdefault('schemas', {})
    component_schemas.update(defs)
    component_schemas['A2ARequest'] = a2a_request_schema

    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
create_a2a_router(app)
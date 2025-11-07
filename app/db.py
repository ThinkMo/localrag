from datetime import UTC, datetime
from enum import Enum
from typing import AsyncGenerator

from sqlalchemy import (
    ARRAY,
    JSON,
    TIMESTAMP,
    Column,
    Enum as SQLAlchemyEnum,
    Integer,
    String,
    Text,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, declared_attr
from langchain_milvus import Milvus, BM25BuiltInFunction
from langchain_huggingface import HuggingFaceEmbeddings

from app.config.config import Configuration


config = Configuration.from_runnable_config()
engine = create_async_engine(
    config.database_url, connect_args={"autocommit": False}
)
async_session_maker = async_sessionmaker(engine, expire_on_commit=False)


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with async_session_maker() as session:
        yield session


def get_vector_store():
    # local mode only support SPARSE_INVERTED_INDEX SPARSE_WAND
    # local mode only support FLAT IVF_FLAT AUTOINDEX
    index_params=[{
        "index_type": "SPARSE_INVERTED_INDEX"
    },{
        "index_type": "SPARSE_INVERTED_INDEX"
    }]
    if not config.vector_store_uri.endswith("db"):
        index_params = [{
            "metric_type": "COSINE",
            "index_type": "HNSW",
        },{
            "metric_type": "BM25",
            "index_type": "AUTOINDEX",
        }]
    embeddings = HuggingFaceEmbeddings(
        model_name=config.embedding_model
    )
    vector_store = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": config.vector_store_uri},
        builtin_function=BM25BuiltInFunction(),
        vector_field=["dense", "sparse"],
        auto_id=True,
        drop_old=False,
        index_params=index_params,
    )
    return vector_store

class DocumentType(str, Enum):
    PDF = "pdf"
    MARKDOWN = "markdown"


class Base(DeclarativeBase):
    pass


class BaseModel(Base):
    __abstract__ = True
    __allow_unmapped__ = True

    id = Column(Integer, primary_key=True, index=True)


class TimestampMixin:
    @declared_attr
    def created_at(cls):  # noqa: N805
        return Column(
            TIMESTAMP(timezone=True),
            nullable=False,
            default=lambda: datetime.now(UTC),
            index=True,
        )


class Document(BaseModel, TimestampMixin):
    __tablename__ = "documents"

    title = Column(String, nullable=False, index=True)
    document_type = Column(SQLAlchemyEnum(DocumentType), nullable=False)
    document_metadata = Column(JSON, nullable=True)

    content = Column(Text, nullable=False)
    content_hash = Column(String, nullable=False, index=True, unique=True)
    unique_identifier_hash = Column(String, nullable=True, index=True, unique=True)

    related_chunks = Column(JSON, nullable=True)
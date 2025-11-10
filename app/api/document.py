import os
import logging

from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredMarkdownLoader
)
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import APIRouter, Depends, HTTPException, UploadFile
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.models.api import PaginatedResponse, DocumentRead
from app.db import get_async_session, Document, DocumentType, get_vector_store
from app.utils import generate_unique_identifier_hash, generate_content_hash


DEFAULT_SPLITTER = "\n\n"
router = APIRouter()


async def get_document_by_unique_identifier_hash(
    unique_identifier_hash: str,
    session: AsyncSession,
) -> Document | None:
    """Get document by unique identifier hash."""
    existing_doc_result = await session.execute(
        select(Document)
        .where(Document.unique_identifier_hash == unique_identifier_hash)
    )
    return existing_doc_result.scalars().first()


async def process_file_upload_task(file_path: str, file_name: str, session: AsyncSession):
    document_type = ""
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext == '.pdf':
        loader = PyPDFLoader(file_path)
        document_type = DocumentType.PDF
    elif file_ext in ['.md', '.markdown']:
        loader = UnstructuredMarkdownLoader(file_path)
        document_type = DocumentType.MARKDOWN
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file format: {file_ext}")
    documents = loader.load()
    unique_identifier_hash = generate_unique_identifier_hash(
        document_type=document_type,
        unique_identifier=file_name,
    )
    content = DEFAULT_SPLITTER.join([doc.page_content for doc in documents])
    content_hash = generate_content_hash(content)

    existing_document = await get_document_by_unique_identifier_hash(
        unique_identifier_hash, session
    )
    if existing_document:
        if existing_document.content_hash == content_hash:
            logging.info(f"Document {file_name} already exists.")
            return
        else:
            logging.info(f"Document {file_name} has changed.")

    # 分割文档
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=100,
        length_function=len
    )
    for doc in documents:
        doc.metadata["source"] = file_name
    chunks = text_splitter.split_documents(documents)

    vector_store = get_vector_store()
    related_chunks = await vector_store.aadd_documents(chunks)

    if existing_document:
        existing_document.content = content
        existing_document.content_hash = content_hash

        deleted_chunks = existing_document.related_chunks
        existing_document.related_chunks = related_chunks

        await session.commit()
        if deleted_chunks:
            await vector_store.adelete(ids=deleted_chunks)
    else:
        new_document = Document(
            unique_identifier_hash=unique_identifier_hash,
            document_type=document_type,
            title=file_name,
            content=content,
            content_hash=content_hash,
            related_chunks=related_chunks,
            document_metadata={
                "source": file_name,
            }
        )
        session.add(new_document)
        await session.commit()


@router.post("/documents/fileupload")
async def create_documents_file_upload(
    files: list[UploadFile],
    session: AsyncSession = Depends(get_async_session),
):
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")

        for file in files:
            try:
                # Save file to a temporary location to avoid stream issues
                import os
                import tempfile

                # Create temp file
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=os.path.splitext(file.filename)[1]
                ) as temp_file:
                    temp_path = temp_file.name

                # Write uploaded file to temp file
                content = await file.read()
                with open(temp_path, "wb") as f:
                    f.write(content)

                await process_file_upload_task(temp_path, file.filename, session)
            except Exception as e:
                raise HTTPException(
                    status_code=422,
                    detail=f"Failed to process file {file.filename}: {e!s}",
                ) from e

        await session.commit()
        return {"message": "Files uploaded for processing"}
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        raise HTTPException(
            status_code=500, detail=f"Failed to upload files: {e!s}"
        ) from e


@router.get("/documents/", response_model=PaginatedResponse[DocumentRead])
async def list_documents(
    skip: int | None = None,
    page: int | None = None,
    page_size: int = 50,
    document_types: str | None = None,
    session: AsyncSession = Depends(get_async_session),
):
    """List documents"""
    try:
        from sqlalchemy import func

        query = select(Document)
        # Filter by document_types if provided
        if document_types is not None and document_types.strip():
            type_list = [t.strip() for t in document_types.split(",") if t.strip()]
            if type_list:
                query = query.filter(Document.document_type.in_(type_list))

        # Get total count
        count_query = (select(func.count()).select_from(Document))
        if document_types is not None and document_types.strip():
            type_list = [t.strip() for t in document_types.split(",") if t.strip()]
            if type_list:
                count_query = count_query.filter(Document.document_type.in_(type_list))
        total_result = await session.execute(count_query)
        total = total_result.scalar() or 0

        # Calculate offset
        offset = 0
        if skip is not None:
            offset = skip
        elif page is not None:
            offset = page * page_size

        # Get paginated results
        if page_size == -1:
            result = await session.execute(query.offset(offset))
        else:
            result = await session.execute(query.offset(offset).limit(page_size))

        db_documents = result.scalars().all()

        # Convert database objects to API-friendly format
        api_documents = []
        for doc in db_documents:
            api_documents.append(
                DocumentRead(
                    id=doc.id,
                    title=doc.title,
                    document_type=doc.document_type,
                    document_metadata=doc.document_metadata,
                    content=doc.content,
                    created_at=doc.created_at,
                )
            )

        return PaginatedResponse(items=api_documents, total=total)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch documents: {e!s}"
        ) from e


@router.delete("/documents/{document_id}", response_model=dict)
async def delete_document(
    document_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    try:
        # Query the document directly instead of using read_document function
        result = await session.execute(
            select(Document).filter(Document.id == document_id)
        )
        document = result.scalars().first()

        if not document:
            raise HTTPException(
                status_code=404, detail=f"Document with id {document_id} not found"
            )

        deleted_chunks = document.related_chunks
        if deleted_chunks:
            vector_store = get_vector_store()
            await vector_store.adelete(ids=deleted_chunks)

        await session.delete(document)
        await session.commit()
        return {"message": "Document deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        raise HTTPException(
            status_code=500, detail=f"Failed to delete document: {e!s}"
        ) from e
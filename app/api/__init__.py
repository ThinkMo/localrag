from fastapi import APIRouter

from .document import router as document_router


router = APIRouter(prefix="/api/v1", tags=["v1"])

router.include_router(document_router)

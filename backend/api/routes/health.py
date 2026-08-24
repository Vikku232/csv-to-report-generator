# backend/api/routes/health.py
from fastapi import APIRouter
from backend.models.schemas import HealthResponse

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
def health_check():
    return {"status": "ok"}
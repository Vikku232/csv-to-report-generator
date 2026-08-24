from typing import Optional, Literal
from pydantic import BaseModel

class ReportRequestMeta(BaseModel):
    target_column: Optional[str] = None
    task_type: Optional[Literal["classification", "regression"]] = None


class HealthResponse(BaseModel):
    status: str
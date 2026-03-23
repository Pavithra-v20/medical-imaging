from pydantic import BaseModel
from typing import Any, Optional


class SessionResponse(BaseModel):
    session_id: str
    summary: str
    report: Optional[str] = None
    rev_req: bool
    prediction: dict[str, Any]

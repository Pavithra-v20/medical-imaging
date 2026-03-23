from pydantic import BaseModel
from typing import Any


class SessionResponse(BaseModel):
    session_id: str
    summary: str
    report: str
    rev_req: bool
    prediction: dict[str, Any]

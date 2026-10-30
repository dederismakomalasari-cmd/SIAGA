"""Pydantic schema untuk validasi request/response."""
from typing import Optional
from pydantic import BaseModel


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    token: str
    role: str
    crossing_id: Optional[str] = None
    name: str


class ManualTTARequest(BaseModel):
    crossing_id: str
    tta_seconds: float


class AckAlertRequest(BaseModel):
    alert_id: int
    note: Optional[str] = ""
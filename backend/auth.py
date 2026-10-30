"""Autentikasi token sederhana (username sebagai bearer)."""
from fastapi import Header, HTTPException
from .database import get_user_by_token


def require_user(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")
    token = authorization.replace("Bearer ", "").strip()
    user = get_user_by_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user


def require_role(role: str):
    def _dep(authorization: str = Header(None)):
        user = require_user(authorization)
        if user["role"] != role:
            raise HTTPException(status_code=403, detail="Forbidden")
        return user
    return _dep
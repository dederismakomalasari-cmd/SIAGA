"""Semua endpoint HTTP + WebSocket."""
import json
import cv2
import asyncio
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from ..schemas import LoginRequest, LoginResponse, ManualTTARequest, AckAlertRequest
from ..database import (
    authenticate, set_tta_override, get_alerts, ack_alert
)
from ..auth import require_user
from ..alerts.manager import alert_manager
from ..tta.tta_source import tta_source
from ..config import settings

router = APIRouter()


# ==========================================
# 1. AUTENTIKASI
# ==========================================
@router.post("/api/login", response_model=LoginResponse)
def login(req: LoginRequest):
    user = authenticate(req.username, req.password)
    if not user:
        raise HTTPException(401, "Username atau password salah")
    return LoginResponse(
        token=user["username"],
        role=user["role"],
        crossing_id=user["crossing_id"],
        name=user["name"],
    )


# ==========================================
# 2. DATA PERLINTASAN
# ==========================================
@router.get("/api/crossings")
def list_crossings(user=Depends(require_user)):
    if user["role"] == "guard":
        # Guard hanya melihat perlintasan yang dia jaga
        return [c for c in settings.crossings if c["id"] == user["crossing_id"]]
    
    # KAI melihat semua
    return [
        {"id": c["id"], "name": c["name"], "danger_zone_polygon": c["danger_zone_polygon"]}
        for c in settings.crossings
    ]


# ==========================================
# 3. TTA (TIME TO ARRIVAL)
# ==========================================
@router.post("/api/tta")
def set_tta(req: ManualTTARequest, user=Depends(require_user)):
    set_tta_override(req.crossing_id, req.tta_seconds)
    return {"ok": True, "crossing_id": req.crossing_id, "tta_seconds": req.tta_seconds}


@router.get("/api/tta/{crossing_id}")
def get_tta(crossing_id: str, user=Depends(require_user)):
    return {"crossing_id": crossing_id, "tta_seconds": tta_source.get(crossing_id)}


# ==========================================
# 4. ALERTS (LOG & KONFIRMASI)
# ==========================================
@router.get("/api/alerts")
def list_alerts(
    crossing_id: str = None, 
    limit: int = 100, 
    only_confirmed: bool = False, 
    user=Depends(require_user)
):
    # Guard selalu lihat semua alarm di perlintasannya
    if user["role"] == "guard":
        crossing_id = user["crossing_id"]
        only_confirmed = False
    
    # Ambil dari database
    rows = get_alerts(crossing_id, limit)
    
    # Filter hanya yang sudah di-ack (Terkonfirmasi) jika diminta oleh KAI
    if only_confirmed:
        rows = [r for r in rows if r["acked"] == 1]

    # Parse JSON objects
    for r in rows:
        try:
            r["objects"] = json.loads(r["objects_json"] or "[]")
        except Exception:
            r["objects"] = []
        r.pop("objects_json", None)
    
    return rows


@router.post("/api/alerts/ack")
async def ack(req: AckAlertRequest, user=Depends(require_user)):
    if user["role"] != "guard":
        raise HTTPException(403, "Hanya penjaga palang yang dapat konfirmasi")
    
    # 1. Tandai sudah di-ack di database
    ack_alert(req.alert_id, user["username"], req.note or "")
    
    # 2. Ambil data alert yang baru saja di-ack ini
    alerts = get_alerts(user["crossing_id"], limit=1)
    alert_data = None
    if alerts:
        alert_data = alerts[0]
        try:
            alert_data["objects"] = json.loads(alert_data["objects_json"] or "[]")
        except Exception:
            alert_data["objects"] = []
        alert_data.pop("objects_json", None)

    # 3. Broadcast ke WebSocket (dengan data lengkap untuk Pop-up KAI)
    await alert_manager.broadcast({
        "type": "alert_acked",
        "alert_data": alert_data
    })
    
    return {"ok": True}


# ==========================================
# 5. WEBSOCKET (REAL-TIME)
# ==========================================
@router.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    alert_manager.register(ws)
    try:
        while True:
            # Menerima ping/keep-alive dari client
            await ws.receive_text() 
    except WebSocketDisconnect:
        alert_manager.unregister(ws)
    except Exception:
        alert_manager.unregister(ws)


# ==========================================
# 6. STREAMING VIDEO (MJPEG)
# ==========================================
from ..pipeline import latest_frames  # Import di sini untuk hindari circular import

def _mjpeg(crossing_id):
    while True:
        frame = latest_frames.get(crossing_id)
        if frame is None:
            import time
            time.sleep(0.1)
            continue
            
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            continue
            
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")


@router.get("/stream/{crossing_id}")
def stream(crossing_id: str):
    return StreamingResponse(
        _mjpeg(crossing_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
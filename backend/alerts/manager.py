"""Manajer alert: cooldown, snapshot, simpan DB, push WebSocket."""
import json
import time
import cv2
from datetime import datetime
from pathlib import Path

from ..config import settings
from ..database import insert_alert


class AlertManager:
    def __init__(self):
        self._last_alert = {}  # crossing_id -> timestamp
        self.connections = []  # list[WebSocket]

    def register(self, ws):
        self.connections.append(ws)

    def unregister(self, ws):
        if ws in self.connections:
            self.connections.remove(ws)

    async def broadcast(self, message: dict):
        dead = []
        for ws in self.connections:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for d in dead:
            self.unregister(d)

    def should_alert(self, crossing_id) -> bool:
        now = time.time()
        last = self._last_alert.get(crossing_id, 0)
        if now - last < settings.alert_cooldown:
            return False
        self._last_alert[crossing_id] = now
        return True

    def save_snapshot(self, frame, crossing_id) -> str:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        fname = f"{crossing_id}_{ts}.jpg"
        path = settings.snapshot_dir / fname
        cv2.imwrite(str(path), frame)
        return fname

    async def trigger(self, crossing_id, frame, risk_level, risk_label,
                      proba, features, tta, objects):
        if not self.should_alert(crossing_id):
            return None
        snap = self.save_snapshot(frame, crossing_id)
        alert_id = insert_alert(
            crossing_id, risk_level, risk_label, tta, snap,
            json.dumps(objects, default=str),
        )
        msg = {
            "type": "risk_alert",
            "alert_id": alert_id,
            "crossing_id": crossing_id,
            "timestamp": datetime.utcnow().isoformat(),
            "risk_level": risk_level,
            "risk_label": risk_label,
            "proba": proba,
            "features": features,
            "tta_seconds": tta,
            "objects": objects,
            "snapshot_url": f"/snapshots/{snap}",
        }
        await self.broadcast(msg)
        return msg


alert_manager = AlertManager()
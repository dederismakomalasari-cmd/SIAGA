"""Loader konfigurasi global SIAGA."""
import os
import yaml
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "configs" / "crossings.yaml"


class Settings:
    def __init__(self):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        self.server = raw.get("server", {})
        self.models = raw.get("models", {})
        self.target_classes = set(raw.get("target_classes", []))
        self.crossings = raw.get("crossings", [])

        self.snapshot_dir = BASE_DIR / self.server.get("snapshot_dir", "snapshots")
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        self.alert_cooldown = float(self.server.get("alert_cooldown_seconds", 5))
        self.frame_interval = float(self.server.get("frame_interval_seconds", 0.5))

    def resolve_model_path(self, key):
        """Cari model di folder models/, fallback bila tidak ada."""
        path = BASE_DIR / self.models.get(key, "")
        if path.exists():
            return str(path)
            
        # Fallback khusus untuk yolo_path (COCO)
        if key == "yolo_path":
            fallback = self.models.get("yolo_fallback")
            if fallback:
                fb = BASE_DIR / fallback
                if fb.exists():
                    return str(fb)
                if fallback.endswith(".pt"):
                    return fallback  # Biarkan ultralytics yang mengunduh otomatis
        return None

    def get_crossing(self, crossing_id):
        for c in self.crossings:
            if c["id"] == crossing_id:
                return c
        return None

settings = Settings()
"""Sumber TTA: prioritas override manual > default crossing."""
from ..database import get_tta_override
from ..config import settings


class TTASource:
    def __init__(self):
        self._default = {c["id"]: c.get("tta_default_seconds", 30) for c in settings.crossings}

    def get(self, crossing_id) -> float:
        override = get_tta_override(crossing_id)
        if override is not None:
            return float(override)
        return float(self._default.get(crossing_id, 30))


tta_source = TTASource()
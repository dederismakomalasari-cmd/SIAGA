"""Wrapper segmentasi zona perlintasan dari model YOLO-seg hasil training."""
from ultralytics import YOLO
import numpy as np
import cv2


class SegmentationDetector:
    def __init__(self, model_path: str, conf: float = 0.25):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect_zone_mask(self, frame):
        h, w = frame.shape[:2]
        result = self.model.predict(frame, conf=self.conf, verbose=False)[0]
        mask = np.zeros((h, w), dtype=np.uint8)
        if result.masks is None or len(result.masks.xy) == 0:
            return None
        for polygon in result.masks.xy:
            pts = np.round(polygon).astype(np.int32)
            if len(pts) >= 3:
                cv2.fillPoly(mask, [pts], 1)
        return mask if mask.sum() > 0 else None
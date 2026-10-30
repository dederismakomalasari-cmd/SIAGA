"""Wrapper YOLO deteksi pekerja (vest, helmet, worker) — model custom Anda."""
from ultralytics import YOLO


class WorkerDetector:
    def __init__(self, model_path: str, conf: float = 0.3, iou: float = 0.5):
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.names = self.model.names

    def detect(self, frame):
        results = self.model(frame, conf=self.conf, iou=self.iou, verbose=False)
        boxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                cls_id = int(box.cls[0])
                boxes.append({
                    "bbox": [x1, y1, x2, y2],
                    "class_id": cls_id,
                    "class_name": self.names.get(cls_id, str(cls_id)),
                    "confidence": float(box.conf[0]),
                    "source": "worker_model",
                })
        return boxes
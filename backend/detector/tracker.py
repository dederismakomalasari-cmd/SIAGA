"""Tracker IoU sederhana untuk memberi ID konsisten antar frame."""
import itertools


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0


class SimpleTracker:
    def __init__(self, iou_threshold=0.3):
        self.iou_threshold = iou_threshold
        self._next_id = itertools.count(1)
        self._tracks = {}  # id -> {"bbox": ..., "center": ..., "history": [...]}

    def update(self, detections):
        used = set()
        new_tracks = {}
        for det in detections:
            best_id, best_iou = None, 0.0
            for tid, tr in self._tracks.items():
                if tid in used:
                    continue
                v = _iou(det["bbox"], tr["bbox"])
                if v > best_iou:
                    best_iou, best_id = v, tid
            if best_id is not None and best_iou >= self.iou_threshold:
                tid = best_id
                used.add(tid)
            else:
                tid = next(self._next_id)
            cx = (det["bbox"][0] + det["bbox"][2]) / 2
            cy = (det["bbox"][1] + det["bbox"][3]) / 2
            hist = self._tracks.get(tid, {}).get("history", [])[-9:] + [(cx, cy)]
            new_tracks[tid] = {
                "bbox": det["bbox"],
                "center": (cx, cy),
                "class_name": det["class_name"],
                "confidence": det["confidence"],
                "source": det.get("source", "coco"),  # Penting: bedakan model
                "history": hist,
            }
        self._tracks = new_tracks
        return new_tracks
"""Pipeline utama: baca kamera → deteksi (COCO + Worker) → tracking → fitur → risk → alert."""
import asyncio
import time
import cv2
import numpy as np
from collections import defaultdict

from .config import settings
from .detector.yolo_wrapper import YOLODetector
from .detector.worker_detector import WorkerDetector
from .detector.seg_wrapper import SegmentationDetector
from .detector.tracker import SimpleTracker
from .risk.feature_builder import build_features, point_in_polygon, bbox_overlap_ratio
from .risk.xgb_wrapper import RiskAssessor, LABELS
from .tta.tta_source import tta_source
from .alerts.manager import alert_manager

latest_frames = defaultdict(lambda: None)


class CrossingWorker:
    def __init__(self, crossing_cfg):
        self.cfg = crossing_cfg
        self.id = crossing_cfg["id"]
        self.polygon = crossing_cfg["danger_zone_polygon"]

        # 1. Detektor COCO (person, car, motorcycle, ...)
        self.coco_detector = YOLODetector(
            settings.resolve_model_path("yolo_path"),
            conf=settings.models.get("yolo_conf", 0.4),
            iou=settings.models.get("yolo_iou", 0.5),
        )

        # 2. Detektor worker custom Anda
        worker_path = settings.resolve_model_path("worker_path")
        if worker_path:
            try:
                self.worker_detector = WorkerDetector(
                    worker_path,
                    conf=settings.models.get("worker_conf", 0.3),
                    iou=settings.models.get("worker_iou", 0.5),
                )
                print(f"[{self.id}] Worker model aktif: {worker_path}")
            except Exception as e:
                print(f"[{self.id}] Worker model gagal load: {e}")
                self.worker_detector = None
        else:
            print(f"[{self.id}] Worker model tidak ditemukan, lewati.")
            self.worker_detector = None

        # 3. Detektor segmentasi zona perlintasan
        seg_path = settings.resolve_model_path("seg_path")
        if seg_path:
            try:
                self.seg_detector = SegmentationDetector(
                    seg_path,
                    conf=settings.models.get("seg_conf", 0.25),
                )
                print(f"[{self.id}] Segmentasi aktif: {seg_path}")
            except Exception as e:
                print(f"[{self.id}] Segmentasi gagal load: {e}")
                self.seg_detector = None
        else:
            print(f"[{self.id}] Segmentasi tidak ada, fallback ke polygon manual.")
            self.seg_detector = None

        self.tracker = SimpleTracker(iou_threshold=0.3)
        self.risk = RiskAssessor(
            settings.resolve_model_path("xgb_path"),
            fallback_mode=settings.models.get("xgb_fallback_mode", "rule"),
        )
        self.cap = None
        self._last_alert_level = None

    def _open_camera(self):
        src = self.cfg["camera_rtsp"]
        src = int(src) if str(src).isdigit() else src
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            print(f"[{self.id}] Gagal membuka kamera {src}")

    async def run(self):
        if self.cap is None:
            self._open_camera()
        loop = asyncio.get_event_loop()
        while True:
            t0 = time.time()
            ok, frame = self.cap.read()
            if not ok or frame is None:
                await asyncio.sleep(0.2)
                if not self.cap.isOpened():
                    self._open_camera()
                continue

            # === 1. Deteksi paralel: COCO + worker ===
            dets_coco = await loop.run_in_executor(None, self.coco_detector.detect, frame)
            dets_worker = []
            if self.worker_detector is not None:
                dets_worker = await loop.run_in_executor(
                    None, self.worker_detector.detect, frame
                )
            dets_all = dets_coco + dets_worker

            # === 2. Tracking ===
            tracks = self.tracker.update(dets_all)

            # === 3. Segmentasi zona ===
            zone_mask = None
            if self.seg_detector is not None:
                zone_mask = await loop.run_in_executor(
                    None, self.seg_detector.detect_zone_mask, frame
                )

            # === 4. Fitur + risk ===
            zone_input = zone_mask if zone_mask is not None else self.polygon
            tta = tta_source.get(self.id)
            feats = build_features(
                tracks, zone_input, tta, settings.target_classes
            )
            risk_level, proba = self.risk.assess(feats)

            # === 5. Gambar overlay ===
            self._draw(frame, tracks, risk_level, zone_mask)
            latest_frames[self.id] = frame.copy()

            # === 6. Trigger alert ===
            if risk_level >= 1 and (
                risk_level != self._last_alert_level or risk_level == 2
            ):
                objects = []
                for tid, tr in tracks.items():
                    if tr["class_name"] not in settings.target_classes:
                        continue
                    if zone_mask is not None:
                        in_d = bbox_overlap_ratio(tr["bbox"], zone_mask) >= 0.3
                    else:
                        in_d = point_in_polygon(tr["center"], self.polygon)
                    objects.append({
                        "track_id": tid,
                        "class": tr["class_name"],
                        "source": tr.get("source", "coco"),
                        "confidence": tr["confidence"],
                        "in_danger": in_d,
                    })
                await alert_manager.trigger(
                    self.id, frame, risk_level, LABELS[risk_level],
                    proba, feats, tta, objects,
                )
            self._last_alert_level = risk_level

            elapsed = time.time() - t0
            await asyncio.sleep(max(0, settings.frame_interval - elapsed))

    def _draw(self, frame, tracks, risk_level, zone_mask=None):
        color_zone = {0: (0, 200, 0), 1: (0, 200, 255), 2: (0, 0, 255)}[risk_level]

        # Gambar zona
        if zone_mask is not None:
            overlay = frame.copy()
            overlay[zone_mask == 1] = color_zone
            frame[:] = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)
        else:
            pts = np.array(self.polygon, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=color_zone, thickness=3)

        # Gambar objek
        for tid, tr in tracks.items():
            x1, y1, x2, y2 = [int(v) for v in tr["bbox"]]
            cls = tr["class_name"]
            if cls not in settings.target_classes:
                color = (200, 200, 200)  # Abu-abu
            else:
                if zone_mask is not None:
                    in_d = bbox_overlap_ratio(tr["bbox"], zone_mask) >= 0.3
                else:
                    in_d = point_in_polygon(tr["center"], self.polygon)
                
                # Warna beda per sumber model
                if tr.get("source") == "worker_model":
                    color = (0, 0, 255) if in_d else (0, 165, 255)   # Merah / Oranye
                else:
                    color = (0, 0, 255) if in_d else (0, 255, 0)     # Merah / Hijau

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            tag = "W" if tr.get("source") == "worker_model" else "C"
            cv2.putText(
                frame,
                f"[{tag}]#{tid} {cls} {tr['confidence']:.2f}",
                (x1, max(15, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
            )

        cv2.putText(
            frame, f"{self.id} | Risk: {LABELS[risk_level]}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_zone, 2,
        )


async def start_all_workers():
    workers = [CrossingWorker(c) for c in settings.crossings]
    await asyncio.gather(*[w.run() for w in workers])
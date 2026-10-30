"""Bangun fitur numerik untuk XGBoost dari hasil tracking + mask segmentasi."""
import math
import numpy as np


def point_in_polygon(pt, poly):
    """Ray-casting point-in-polygon (fallback bila tidak ada mask)."""
    x, y = pt
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi):
            inside = not inside
        j = i
    return inside


def polygon_centroid(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def bbox_overlap_ratio(bbox, mask):
    """Proporsi area bbox yang beririsan dengan mask (dari kode Colab Anda)."""
    if mask is None:
        return 0.0
    h, w = mask.shape
    x1, y1, x2, y2 = map(int, bbox)
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    region = mask[y1:y2, x1:x2]
    area = (x2 - x1) * (y2 - y1)
    return float(np.count_nonzero(region) / area)


def mask_centroid(mask):
    ys, xs = np.where(mask == 1)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def build_features(tracks, mask_or_polygon, tta_seconds, target_classes,
                   overlap_threshold=0.3):
    if not tracks:
        return {
            "num_objects": 0, "obj_in_danger": 0,
            "max_speed_px": 0.0, "tta": tta_seconds,
            "min_dist_to_center": 9999.0,
            "max_conf_in_danger": 0.0,
            "max_overlap_ratio": 0.0,
            "workers_in_danger": 0,
        }

    use_mask = isinstance(mask_or_polygon, np.ndarray)
    centroid = mask_centroid(mask_or_polygon) if use_mask else polygon_centroid(mask_or_polygon)
    cx_zone, cy_zone = centroid

    num_objects = 0
    obj_in_danger = 0
    max_speed = 0.0
    min_dist = 9999.0
    max_conf_in_danger = 0.0
    max_overlap = 0.0
    workers_in_danger = 0

    for tid, tr in tracks.items():
        if tr["class_name"] not in target_classes:
            continue
        num_objects += 1
        cx, cy = tr["center"]

        if use_mask:
            ratio = bbox_overlap_ratio(tr["bbox"], mask_or_polygon)
            max_overlap = max(max_overlap, ratio)
            in_danger = ratio >= overlap_threshold
        else:
            in_danger = point_in_polygon((cx, cy), mask_or_polygon)
            max_overlap = max(max_overlap, 1.0 if in_danger else 0.0)

        if in_danger:
            obj_in_danger += 1
            max_conf_in_danger = max(max_conf_in_danger, tr["confidence"])
            if tr.get("source") == "worker_model" and tr["class_name"] == "worker":
                workers_in_danger += 1

        d = math.hypot(cx - cx_zone, cy - cy_zone)
        min_dist = min(min_dist, d)

        hist = tr["history"]
        if len(hist) >= 2:
            (x0, y0), (x1, y1) = hist[0], hist[-1]
            max_speed = max(max_speed, math.hypot(x1 - x0, y1 - y0))

    return {
        "num_objects": num_objects,
        "obj_in_danger": obj_in_danger,
        "max_speed_px": round(max_speed, 2),
        "tta": tta_seconds,
        "min_dist_to_center": round(min_dist, 2),
        "max_conf_in_danger": round(max_conf_in_danger, 3),
        "max_overlap_ratio": round(max_overlap, 3),
        "workers_in_danger": workers_in_danger,
    }
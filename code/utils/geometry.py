#@title Geometry 
import numpy as np

def pixel_to_cm(depth_m, pixel_size, f):
    if depth_m is None or not np.isfinite(depth_m) or depth_m < 0.01:
        return np.nan
    return (pixel_size * depth_m / f) * 100.0

def get_mean_depth_in_box(depth_map, box):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    h, w = depth_map.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    region = depth_map[y1:y2, x1:x2]
    valid = region[np.isfinite(region) & (region > 0.01)]
    if len(valid) < 10:
        return None
    return float(np.median(valid))

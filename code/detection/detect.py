#@title Detection 
from ultralytics import YOLO

def load_yolo(model_path):
    return YOLO(model_path)

def filter_boxes(boxes, img_width, img_height, center_ratio=0.70):
    filtered = []
    margin_x = (1.0 - center_ratio) / 2.0
    margin_y = (1.0 - center_ratio) / 2.0
    x_min_allowed = img_width * margin_x
    x_max_allowed = img_width * (1.0 - margin_x)
    y_min_allowed = img_height * margin_y
    y_max_allowed = img_height * (1.0 - margin_y)

    for box in boxes:
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        if (x_min_allowed <= cx <= x_max_allowed and
                y_min_allowed <= cy <= y_max_allowed):
            filtered.append(box)
    return filtered

def detect_fish(model, image_rgb, conf=0.70, iou=0.45):
    results = model(image_rgb, conf=conf, iou=iou, verbose=False)[0]
    return results.boxes

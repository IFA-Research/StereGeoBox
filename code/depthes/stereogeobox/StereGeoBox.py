#@title StereGeoBox
import os
import sys
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

from code.detection.detect import load_yolo, detect_fish, filter_boxes
from code.matching.match import improved_match_boxes
from code.utils.geometry import pixel_to_cm

parser = argparse.ArgumentParser()
parser.add_argument("--left_dir", required=True)
parser.add_argument("--right_dir", required=True)
parser.add_argument("--yolo_model", required=True)
parser.add_argument("--regression", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--fx", type=float, required=True)
parser.add_argument("--fy", type=float, required=True)
parser.add_argument("--baseline", type=float, required=True, help="meters")
parser.add_argument("--scale_length", type=float, default=0.78)
parser.add_argument("--scale_width", type=float, default=0.88)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
vis_dir = os.path.join(args.output_dir, "visualizations")
os.makedirs(vis_dir, exist_ok=True)

yolo = load_yolo(args.yolo_model)
reg_df = pd.read_excel(args.regression).iloc[-95:][['Width', 'Length', 'Weight']].dropna()
from xgboost import XGBRegressor
xgb = XGBRegressor(random_state=42).fit(reg_df[['Width', 'Length']], reg_df['Weight'])

def draw_side_by_side(left_img, right_img, results_info, left_boxes, right_boxes):
    h, w = left_img.shape[:2]
    canvas = np.zeros((h, w * 2, 3), dtype=np.uint8)
    canvas[:, :w] = left_img
    canvas[:, w:] = right_img
    colors = [(0,255,0),(255,0,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255)]
    for idx, info in enumerate(results_info):
        color = colors[idx % len(colors)]
        x1,y1,x2,y2 = map(int, left_boxes[info["l_idx"]].xyxy[0])
        cv2.rectangle(canvas, (x1,y1), (x2,y2), color, 3)
        rx1,ry1,rx2,ry2 = map(int, right_boxes[info["r_idx"]].xyxy[0])
        cv2.rectangle(canvas, (rx1+w,ry1), (rx2+w,ry2), color, 3)
        for i,t in enumerate([f"L:{info['length']:.1f}cm", f"W:{info['width']:.1f}cm",
                              f"D:{info['depth']:.1f}cm", f"WT:{info['weight']:.1f}g"]):
            cv2.putText(canvas, t, (x2+8, y1+28+i*32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
    return canvas

left_images = sorted([f for f in os.listdir(args.left_dir) if f.lower().endswith(('.jpg','.png'))])
all_results = []

for left_name in tqdm(left_images, desc="StereGeoBox"):
    left_img = cv2.imread(os.path.join(args.left_dir, left_name))
    right_img = cv2.imread(os.path.join(args.right_dir, left_name))
    if left_img is None or right_img is None:
        continue
    h, w = left_img.shape[:2]

    left_boxes = filter_boxes(detect_fish(yolo, cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)), w, h)
    right_boxes = filter_boxes(detect_fish(yolo, cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)), w, h)
    if not left_boxes or not right_boxes:
        continue

    matches = improved_match_boxes(left_boxes, right_boxes)
    if not matches:
        continue

    results_info = []
    for mid, (l_idx, r_idx, disp) in enumerate(matches):
        if disp < 1.0:
            continue

        depth_m = np.clip(depth_m, 0.05, 2.5)

        box = left_boxes[l_idx]
        w_px = float(box.xyxy[0][2] - box.xyxy[0][0])
        h_px = float(box.xyxy[0][3] - box.xyxy[0][1])
        length_cm = pixel_to_cm(depth_m, w_px, args.fx) * args.scale_length
        width_cm  = pixel_to_cm(depth_m, h_px, args.fy) * args.scale_width

        if not (np.isfinite(length_cm) and np.isfinite(width_cm)):
            continue
        if not (5 < length_cm < 50 and 2 < width_cm < 20):
            continue

        weight_g = float(xgb.predict(pd.DataFrame([[round(width_cm,2), round(length_cm,2)]],
                                                   columns=['Width','Length']))[0])
        all_results.append({
            "Image": left_name, "Object": mid+1,
            "Length_cm": round(length_cm,2), "Width_cm": round(width_cm,2),
            "Depth_cm": round(depth_m*100,2), "Weight_g": round(weight_g,2),
            "Disparity": round(disp,1)
        })
        results_info.append({"l_idx":l_idx, "r_idx":r_idx, "length":length_cm,
                             "width":width_cm, "depth":depth_m*100, "weight":weight_g})

    if results_info:
        vis = draw_side_by_side(left_img, right_img, results_info, left_boxes, right_boxes)
        cv2.imwrite(os.path.join(vis_dir, left_name), vis)

if all_results:
    pd.DataFrame(all_results).to_excel(os.path.join(args.output_dir, "results_StereGeoBox.xlsx"), index=False)
    print(f"Saved {len(all_results)} records")
else:
    print("No valid results.")

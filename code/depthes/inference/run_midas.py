#@title MiDaS Inference
import os, sys, cv2, numpy as np, pandas as pd
from tqdm import tqdm
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
from code.detection.detect import load_yolo, detect_fish, filter_boxes
from code.matching.match import improved_match_boxes
from code.utils.geometry import pixel_to_cm, get_mean_depth_in_box

parser = argparse.ArgumentParser()
parser.add_argument("--left_dir", required=True)
parser.add_argument("--right_dir", required=True)
parser.add_argument("--depth_dir", required=True, help="Folder containing MiDaS npz files")
parser.add_argument("--yolo_model", required=True)
parser.add_argument("--regression", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--fx", type=float, required=True)
parser.add_argument("--fy", type=float, required=True)
parser.add_argument("--scale_length", type=float, default=0.78)
parser.add_argument("--scale_width", type=float, default=0.88)
parser.add_argument("--min_depth", type=float, default=0.05)
parser.add_argument("--max_depth", type=float, default=2.50)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
yolo = load_yolo(args.yolo_model)
reg_df = pd.read_excel(args.regression).iloc[-95:][['Width','Length','Weight']].dropna()
from xgboost import XGBRegressor
xgb = XGBRegressor(random_state=42).fit(reg_df[['Width','Length']], reg_df['Weight'])

left_images = sorted([f for f in os.listdir(args.left_dir) if f.lower().endswith(('.jpg','.png'))])
all_results = []

for left_name in tqdm(left_images, desc="MiDaS"):
    left_img = cv2.imread(os.path.join(args.left_dir, left_name))
    right_img = cv2.imread(os.path.join(args.right_dir, left_name))
    if left_img is None or right_img is None: continue
    h, w = left_img.shape[:2]

    left_boxes = filter_boxes(detect_fish(yolo, cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)), w, h)
    right_boxes = filter_boxes(detect_fish(yolo, cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)), w, h)
    if not left_boxes or not right_boxes: continue

    matches = improved_match_boxes(left_boxes, right_boxes)
    if not matches: continue

    base = os.path.splitext(left_name)[0]
    npz_path = os.path.join(args.depth_dir, f"{base}_relative.npz")
    if not os.path.exists(npz_path): continue
    rel = np.load(npz_path)["depth"].astype(np.float32)
    dmin, dmax = np.percentile(rel, 1), np.percentile(rel, 99)
    if dmax > dmin: rel = (rel - dmin) / (dmax - dmin)
    depth_map = rel * (args.max_depth - args.min_depth) + args.min_depth

    for mid, (l_idx, r_idx, _) in enumerate(matches):
        depth_m = get_mean_depth_in_box(depth_map, left_boxes[l_idx])
        if depth_m is None or not np.isfinite(depth_m) or depth_m <= 0.01:
            depth_m = args.max_depth
        else:
            depth_m = np.clip(depth_m, args.min_depth, args.max_depth)

        box = left_boxes[l_idx]
        w_px = float(box.xyxy[0][2] - box.xyxy[0][0])
        h_px = float(box.xyxy[0][3] - box.xyxy[0][1])
        length_cm = pixel_to_cm(depth_m, w_px, args.fx) * args.scale_length
        width_cm  = pixel_to_cm(depth_m, h_px, args.fy) * args.scale_width
        if not (np.isfinite(length_cm) and np.isfinite(width_cm)): continue
        if not (5 < length_cm < 50 and 2 < width_cm < 20): continue

        weight_g = float(xgb.predict(pd.DataFrame([[round(width_cm,2), round(length_cm,2)]], columns=['Width','Length']))[0])
        all_results.append({
            "Image": left_name, "Object": mid+1,
            "Length_cm": round(length_cm,2), "Width_cm": round(width_cm,2),
            "Depth_cm": round(depth_m*100,2), "Weight_g": round(weight_g,2)
        })

if all_results:
    pd.DataFrame(all_results).to_excel(os.path.join(args.output_dir, "results_MiDaS.xlsx"), index=False)
    print(f"Saved {len(all_results)} records")
else:
    print("No valid results.")

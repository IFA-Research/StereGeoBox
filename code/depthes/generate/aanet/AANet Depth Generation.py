#@title AANet Depth Generation
import os
import cv2
import numpy as np
import torch
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from nets.aanet import AANet
import dill
import warnings
import argparse

warnings.filterwarnings("ignore", category=UserWarning)

# ================== Argument Parser ==================
parser = argparse.ArgumentParser(description="AANet Depth Estimation")
parser.add_argument("--left_dir", type=str, required=True, help="Path to left images folder")
parser.add_argument("--right_dir", type=str, required=True, help="Path to right images folder")
parser.add_argument("--output_dir", type=str, required=True, help="Path to save depth results")
parser.add_argument("--pretrained", type=str, default="./pretrained/aanet_kitti15.pth",
                    help="Path to pretrained AANet weights")
parser.add_argument("--model_dill", type=str, default="./model.dill",
                    help="Path to model.dill file")
parser.add_argument("--fx", type=float, required=True,
                    help="Focal length (fx) in pixels - from your camera calibration")
parser.add_argument("--baseline_mm", type=float, required=True,
                    help="Baseline in millimeters - from your camera calibration")
parser.add_argument("--min_depth", type=float, required=True,
                    help="Minimum valid depth in meters (e.g. 0.05)")
parser.add_argument("--max_depth", type=float, required=True,
                    help="Maximum valid depth in meters (e.g. 1.50)")
args = parser.parse_args()

# ================== Camera & Depth Range ==================
FX = args.fx
BASELINE = args.baseline_mm / 1000.0
MIN_DEPTH = args.min_depth
MAX_DEPTH = args.max_depth

os.makedirs(args.output_dir, exist_ok=True)

print("Loading AANet model...")
with open(args.model_dill, "rb") as F:
    model = dill.load(F)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.load_state_dict(torch.load(args.pretrained, map_location=device))
model.eval()
model.to(device)
print("Model loaded successfully.")

# ================== Colormap ==================
cmap = plt.get_cmap("turbo")
norm = plt.Normalize(vmin=MIN_DEPTH, vmax=MAX_DEPTH)

# ================== Image list ==================
images = sorted(
    glob.glob(os.path.join(args.left_dir, "*.jpg")) +
    glob.glob(os.path.join(args.left_dir, "*.png"))
)
print(f"Found {len(images)} images")

INFER_H = 480
INFER_W = 1560
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ================== Process ==================
for img_path in tqdm(images, desc="AANet Depth"):
    filename = os.path.basename(img_path)
    name_no_ext = os.path.splitext(filename)[0]

    out_png = os.path.join(args.output_dir, name_no_ext + "_depth.png")
    out_npz = os.path.join(args.output_dir, name_no_ext + "_disp.npz")

    if os.path.exists(out_png) and os.path.exists(out_npz):
        continue

    right_path = os.path.join(args.right_dir, filename)
    if not os.path.exists(right_path):
        continue

    left = cv2.imread(img_path)
    right = cv2.imread(right_path)
    if left is None or right is None:
        continue

    orig_h, orig_w = left.shape[:2]

    left_r = cv2.resize(left, (INFER_W, INFER_H), interpolation=cv2.INTER_LINEAR)
    right_r = cv2.resize(right, (INFER_W, INFER_H), interpolation=cv2.INTER_LINEAR)

    left_r = left_r.astype(np.float32) / 255.0
    right_r = right_r.astype(np.float32) / 255.0
    left_r = (left_r - mean) / std
    right_r = (right_r - mean) / std

    left_t = torch.from_numpy(left_r).permute(2, 0, 1).unsqueeze(0).to(device)
    right_t = torch.from_numpy(right_r).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        disp = model(left_t, right_t)[-1].squeeze().cpu().numpy()

    disp = cv2.medianBlur(disp.astype(np.float32), 5)
    disp = cv2.bilateralFilter(disp, d=7, sigmaColor=50, sigmaSpace=50)
    disp = cv2.resize(disp, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    scale = orig_w / float(INFER_W)
    disp = disp * scale

    depth = (FX * BASELINE) / (disp + 1e-6)
    depth = np.clip(depth, MIN_DEPTH, MAX_DEPTH)
    depth[disp < 1.0] = MAX_DEPTH

    # Save numerical disparity as NPZ
    np.savez_compressed(out_npz, disp=disp.astype(np.float32))

    # Save colored depth visualization
    colored = (cmap(norm(depth))[:, :, :3] * 255).astype(np.uint8)
    colored_bgr = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_png, colored_bgr)

print("\nAll done!")
print(f"Results saved in: {args.output_dir}")

#@title SGBM Depth Generation 
import cv2
import numpy as np
import os
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

# ================== Argument Parser ==================
parser = argparse.ArgumentParser(description="SGBM Depth Estimation")
parser.add_argument("--left_dir", type=str, required=True, help="Path to left images folder")
parser.add_argument("--right_dir", type=str, required=True, help="Path to right images folder")
parser.add_argument("--output_dir", type=str, required=True, help="Path to save depth results")
parser.add_argument("--fx", type=float, required=True,
                    help="Focal length (fx) in pixels - from your camera calibration")
parser.add_argument("--baseline_mm", type=float, required=True,
                    help="Baseline in millimeters - from your camera calibration")
parser.add_argument("--min_depth", type=float, required=True,
                    help="Minimum valid depth in meters (e.g. 0.10)")
parser.add_argument("--max_depth", type=float, required=True,
                    help="Maximum valid depth in meters (e.g. 1.00)")
args = parser.parse_args()

# ================== Camera & Depth Range ==================
FX = args.fx
BASELINE = args.baseline_mm / 1000.0
MIN_DEPTH = args.min_depth
MAX_DEPTH = args.max_depth

os.makedirs(args.output_dir, exist_ok=True)

# ================== SGBM Parameters ==================
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=128,
    blockSize=100,
    P1=8 * 1 * 100 * 100,
    P2=32 * 1 * 100 * 100,
    disp12MaxDiff=20,
    uniquenessRatio=0,
    speckleWindowSize=500,
    speckleRange=2,
    preFilterCap=31,
    mode=cv2.STEREO_SGBM_MODE_HH
)

# ================== Colormap ==================
cmap = plt.get_cmap("turbo")
norm = plt.Normalize(vmin=MIN_DEPTH, vmax=MAX_DEPTH)

# ================== Image list ==================
left_paths = sorted(
    glob.glob(os.path.join(args.left_dir, "*.jpg")) +
    glob.glob(os.path.join(args.left_dir, "*.png"))
)
print(f"Found {len(left_paths)} images")
print(f"Output directory: {args.output_dir}")

# ================== Process ==================
for left_path in tqdm(left_paths, desc="SGBM Depth"):
    filename = os.path.basename(left_path)
    name_no_ext = os.path.splitext(filename)[0]
    right_path = os.path.join(args.right_dir, filename)

    out_png = os.path.join(args.output_dir, name_no_ext + "_depth.png")
    out_npz = os.path.join(args.output_dir, name_no_ext + "_disp.npz")

    if os.path.exists(out_png) and os.path.exists(out_npz):
        continue

    if not os.path.exists(right_path):
        continue

    left = cv2.imread(left_path)
    right = cv2.imread(right_path)
    if left is None or right is None:
        continue

    left_g = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_g = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    disparity = stereo.compute(left_g, right_g).astype(np.float32) / 16.0

    depth = (FX * BASELINE) / (disparity + 1e-6)
    depth = np.nan_to_num(depth, nan=MIN_DEPTH)
    depth = np.clip(depth, MIN_DEPTH, MAX_DEPTH)

    # Save numerical disparity
    np.savez_compressed(out_npz, disp=disparity.astype(np.float32))

    # Save colored depth map
    depth_normalized = norm(depth)
    depth_colored = (cmap(depth_normalized)[:, :, :3] * 255).astype(np.uint8)
    depth_bgr = cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_png, depth_bgr)

print("\nAll done!")
print(f"SGBM depth maps saved in: {args.output_dir}")

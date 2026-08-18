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
warnings.filterwarnings("ignore", category=UserWarning)

# ================== Camera Parameters ==================
FX_LEFT = 667.871914
BASELINE_MM = 59.443
BASELINE = BASELINE_MM / 1000.0          # convert to meters

MIN_DEPTH = 0.05
MAX_DEPTH = 1.50

# ================== Paths (WSL) ==================
left_dir = "/mnt/d/FDCamera/Data/2026-07-01/left_frames"
right_dir = "/mnt/d/FDCamera/Data/2026-07-01/right_frames"
output_dir = "/mnt/d/FDCamera/Data/2026-07-01/AANet_Depth"

os.makedirs(output_dir, exist_ok=True)

# ================== Load Model ==================
print("Loading AANet model...")
with open("./model.dill", "rb") as F:
    model = dill.load(F)

pretrained_path = "/mnt/e/Project/Fishfiles/Real time fish biomas/Code/aanet/pretrained/aanet_kitti15.pth"
model.load_state_dict(torch.load(pretrained_path))
model.eval()
model.cuda()
print("Model loaded successfully.")

# ================== Colormap ==================
cmap = plt.get_cmap('turbo')
norm = plt.Normalize(vmin=MIN_DEPTH, vmax=MAX_DEPTH)

# ================== Get image list ==================
images = sorted(
    glob.glob(os.path.join(left_dir, "*.jpg")) +
    glob.glob(os.path.join(left_dir, "*.png"))
)
print(f"Found {len(images)} images")

# Inference resolution (divisible by 3)
INFER_H = 480
INFER_W = 1560

# ImageNet normalization
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ================== Process images ==================
for img_path in tqdm(images, desc="AANet Depth"):
    filename = os.path.basename(img_path)
    name_no_ext = os.path.splitext(filename)[0]

    out_png = os.path.join(output_dir, name_no_ext + "_depth.png")
    out_npy = os.path.join(output_dir, name_no_ext + "_depth.npy")

    # Skip if both files already exist
    if os.path.exists(out_png) and os.path.exists(out_npy):
        continue

    right_path = os.path.join(right_dir, filename)
    if not os.path.exists(right_path):
        continue

    # Read stereo pair
    left = cv2.imread(img_path)
    right = cv2.imread(right_path)
    if left is None or right is None:
        continue

    orig_h, orig_w = left.shape[:2]

    # Resize to inference resolution
    left_r = cv2.resize(left, (INFER_W, INFER_H), interpolation=cv2.INTER_LINEAR)
    right_r = cv2.resize(right, (INFER_W, INFER_H), interpolation=cv2.INTER_LINEAR)

    # Normalize with ImageNet stats
    left_r = left_r.astype(np.float32) / 255.0
    right_r = right_r.astype(np.float32) / 255.0
    left_r = (left_r - mean) / std
    right_r = (right_r - mean) / std

    # To tensor
    left_t = torch.from_numpy(left_r).permute(2, 0, 1).unsqueeze(0).cuda()
    right_t = torch.from_numpy(right_r).permute(2, 0, 1).unsqueeze(0).cuda()

    # Inference
    with torch.no_grad():
        disp = model(left_t, right_t)[-1].squeeze().cpu().numpy()

    # Post-processing on disparity
    disp = cv2.medianBlur(disp.astype(np.float32), 5)
    disp = cv2.bilateralFilter(disp, d=7, sigmaColor=50, sigmaSpace=50)

    # Resize disparity back to original resolution
    disp = cv2.resize(disp, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    # Scale disparity because width was changed
    scale = orig_w / float(INFER_W)
    disp = disp * scale

    # Convert disparity to metric depth (meters)
    depth = (FX_LEFT * BASELINE) / (disp + 1e-6)
    depth = np.clip(depth, MIN_DEPTH, MAX_DEPTH)
    depth[disp < 1.0] = MAX_DEPTH

    # ===== Save metric depth as .npy =====
    np.save(out_npy, depth.astype(np.float32))

    # ===== Save colored visualization (keep existing style) =====
    colored = (cmap(norm(depth))[:, :, :3] * 255).astype(np.uint8)
    colored_bgr = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_png, colored_bgr)

print("\nAll done!")
print(f"Results saved in: {output_dir}")
print("Both .npy (metric depth) and .png (visualization) are saved.")

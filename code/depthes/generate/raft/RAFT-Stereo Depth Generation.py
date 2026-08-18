#@title RAFT-Stereo Depth Generation 
import sys
sys.path.append('core')

import os
import glob
import argparse
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt
from raft_stereo import RAFTStereo
from utils.utils import InputPadder

# ================== Argument Parser ==================
parser = argparse.ArgumentParser(description="RAFT-Stereo Depth Estimation")
parser.add_argument("--left_dir", type=str, required=True, help="Path to left images folder")
parser.add_argument("--right_dir", type=str, required=True, help="Path to right images folder")
parser.add_argument("--output_dir", type=str, required=True, help="Path to save depth results")
parser.add_argument("--model_ckpt", type=str, required=True, help="Path to RAFT-Stereo checkpoint (.pth)")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
parser.add_argument("--valid_iters", type=int, default=16, help="Number of refinement iterations")
parser.add_argument("--mixed_precision", action="store_true", help="Enable mixed precision")
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(args.output_dir, exist_ok=True)

# ================== Helpers ==================
def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img  # 3,H,W

def paths_for(stem):
    return {
        "npy": os.path.join(args.output_dir, stem + ".npy"),
        "npz": os.path.join(args.output_dir, stem + ".npz"),
        "png": os.path.join(args.output_dir, stem + ".png"),
    }

def load_existing_disp(stem):
    p = paths_for(stem)
    if os.path.exists(p["npz"]) and os.path.getsize(p["npz"]) > 500:
        return np.load(p["npz"])["disp"].astype(np.float32)
    if os.path.exists(p["npy"]) and os.path.getsize(p["npy"]) > 1000:
        return np.load(p["npy"]).astype(np.float32)
    return None

def save_outputs(stem, disp):
    p = paths_for(stem)
    np.savez_compressed(p["npz"], disp=disp.astype(np.float16))
    plt.imsave(p["png"], -disp, cmap="jet")

# ================== Model Args ==================
model_args = argparse.Namespace(
    restore_ckpt=args.model_ckpt,
    mixed_precision=args.mixed_precision,
    valid_iters=args.valid_iters,
    hidden_dims=[128] * 3,
    corr_implementation="reg",
    shared_backbone=False,
    corr_levels=4,
    corr_radius=4,
    n_downsample=2,
    context_norm="batch",
    slow_fast_gru=False,
    n_gru_layers=3,
)

# ================== Load model ==================
print("Loading RAFT-Stereo...")
model = torch.nn.DataParallel(RAFTStereo(model_args), device_ids=[0] if DEVICE == "cuda" else None)
model.load_state_dict(torch.load(args.model_ckpt, map_location=DEVICE))
model = model.module
model.to(DEVICE)
model.eval()
print("Model loaded.\n")

# ================== Collect work ==================
left_images = sorted(
    glob.glob(os.path.join(args.left_dir, "*.jpg")) +
    glob.glob(os.path.join(args.left_dir, "*.png"))
)

pending_infer = []
png_only = []
skipped = 0
missing_right = 0

for left_path in left_images:
    filename = os.path.basename(left_path)
    stem = os.path.splitext(filename)[0]
    right_path = os.path.join(args.right_dir, filename)
    p = paths_for(stem)

    if not os.path.exists(right_path):
        missing_right += 1
        continue

    has_png = os.path.exists(p["png"]) and os.path.getsize(p["png"]) > 500
    disp = load_existing_disp(stem)

    if disp is not None and has_png:
        skipped += 1
        continue
    if disp is not None and not has_png:
        png_only.append((stem, disp))
        continue

    pending_infer.append((left_path, right_path, stem))

print(f"Total left images : {len(left_images)}")
print(f"Skipped complete  : {skipped}")
print(f"PNG only rebuild  : {len(png_only)}")
print(f"Need inference    : {len(pending_infer)}")
print(f"Missing right     : {missing_right}")
print(f"Batch size        : {args.batch_size}\n")

# ================== Rebuild missing PNGs ==================
rebuilt_png = 0
for stem, disp in tqdm(png_only, desc="Rebuild PNG"):
    try:
        plt.imsave(paths_for(stem)["png"], -disp, cmap="jet")
        if not os.path.exists(paths_for(stem)["npz"]):
            np.savez_compressed(paths_for(stem)["npz"], disp=disp.astype(np.float16))
        rebuilt_png += 1
    except Exception as e:
        print(f"PNG rebuild failed {stem}: {e}")

successful = 0
failed = 0

# ================== Inference helpers ==================
@torch.no_grad()
def run_one(left_path, right_path, stem):
    image1 = load_image(left_path)[None].to(DEVICE)
    image2 = load_image(right_path)[None].to(DEVICE)
    padder = InputPadder(image1.shape, divis_by=32)
    image1, image2 = padder.pad(image1, image2)

    with torch.amp.autocast("cuda", enabled=args.mixed_precision and DEVICE == "cuda"):
        _, flow_up = model(image1, image2, iters=args.valid_iters, test_mode=True)

    flow_up = padder.unpad(flow_up).squeeze()
    disp = flow_up.cpu().numpy().squeeze().astype(np.float32)
    save_outputs(stem, disp)

@torch.no_grad()
def run_batch(batch_items):
    global successful, failed
    tensors_left, tensors_right, stems, shapes = [], [], [], []

    for left_path, right_path, stem in batch_items:
        try:
            l = load_image(left_path)
            r = load_image(right_path)
            tensors_left.append(l)
            tensors_right.append(r)
            stems.append(stem)
            shapes.append(l.shape[-2:])
        except Exception as e:
            print(f"Read error {stem}: {e}")
            failed += 1

    if not tensors_left:
        return

    if len(set(shapes)) != 1:
        for left_path, right_path, stem in batch_items:
            try:
                run_one(left_path, right_path, stem)
                successful += 1
            except Exception as e:
                print(f"Failed {stem}: {e}")
                failed += 1
        return

    image1 = torch.stack(tensors_left, dim=0).to(DEVICE)
    image2 = torch.stack(tensors_right, dim=0).to(DEVICE)
    padder = InputPadder(image1.shape, divis_by=32)
    image1, image2 = padder.pad(image1, image2)

    try:
        with torch.amp.autocast("cuda", enabled=args.mixed_precision and DEVICE == "cuda"):
            _, flow_up = model(image1, image2, iters=args.valid_iters, test_mode=True)
        flow_up = padder.unpad(flow_up)
        flow_np = flow_up.cpu().numpy()

        for i, stem in enumerate(stems):
            disp = flow_np[i].squeeze().astype(np.float32)
            save_outputs(stem, disp)
            successful += 1
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("CUDA OOM on batch → fallback to single images")
            torch.cuda.empty_cache()
        for left_path, right_path, stem in batch_items:
            try:
                run_one(left_path, right_path, stem)
                successful += 1
            except Exception as e2:
                print(f"Failed {stem}: {e2}")
                failed += 1

# ================== Run inference ==================
for i in tqdm(range(0, len(pending_infer), args.batch_size), desc="RAFT batches"):
    batch_items = pending_infer[i:i + args.batch_size]
    run_batch(batch_items)

print("\n" + "=" * 70)
print("Finished!")
print(f"Rebuilt PNG only : {rebuilt_png}")
print(f"Inferred new     : {successful}")
print(f"Skipped complete : {skipped}")
print(f"Failed           : {failed}")
print(f"Output           : {args.output_dir}")
print("=" * 70)

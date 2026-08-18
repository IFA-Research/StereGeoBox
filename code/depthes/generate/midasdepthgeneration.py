#@title MiDaS Depth Generation 
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import glob
import os
import cv2
import argparse
from tqdm import tqdm

# ================== Argument Parser ==================
parser = argparse.ArgumentParser(description="MiDaS Depth Estimation")
parser.add_argument("--input_dir", type=str, required=True, help="Path to input images folder (usually left frames)")
parser.add_argument("--output_dir", type=str, required=True, help="Path to save depth results")
parser.add_argument("--model_url", type=str, default="https://tfhub.dev/intel/midas/v2/2",
                    help="TensorFlow Hub URL of the MiDaS model")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# ================== Load Model ==================
print("Loading MiDaS model...")
module = hub.load(args.model_url, tags=["serve"])
print("Model loaded successfully.")

# ================== Image list ==================
image_paths = sorted(
    glob.glob(os.path.join(args.input_dir, "*.jpg")) +
    glob.glob(os.path.join(args.input_dir, "*.png")) +
    glob.glob(os.path.join(args.input_dir, "*.jpeg"))
)
print(f"Found {len(image_paths)} images")

# ================== Process ==================
for image_path in tqdm(image_paths, desc="MiDaS Depth"):
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    relative_path = os.path.join(args.output_dir, f"{base_name}_relative.npz")
    color_path = os.path.join(args.output_dir, f"{base_name}_depth_color.png")

    # Skip if already processed
    if os.path.exists(relative_path) and os.path.exists(color_path):
        continue

    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load: {image_path}")
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    # Preprocess
    img_normalized = img_rgb.astype(np.float32) / 255.0
    img_resized = tf.image.resize(img_normalized, [384, 384], method="bicubic")
    img_input = tf.transpose(img_resized, [2, 0, 1])[tf.newaxis, ...]

    # Inference
    output = module.signatures["serving_default"](tf.convert_to_tensor(img_input))
    prediction = output["default"].numpy().squeeze()
    prediction = cv2.resize(prediction, (w, h), interpolation=cv2.INTER_CUBIC)

    # Convert to relative depth
    relative_depth = 1.0 / (prediction + 1e-6)

    # Save numerical relative depth as NPZ
    np.savez_compressed(relative_path, depth=relative_depth.astype(np.float16))

    # Save colored depth map
    depth_norm = cv2.normalize(relative_depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_norm = depth_norm.astype(np.uint8)
    colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    Image.fromarray(colored_rgb).save(color_path)

print("\nAll done!")
print(f"Results saved in: {args.output_dir}")

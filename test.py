#@title Select 20 Non-Empty Labels + Images (YOLO Structure)
import os
import random
import shutil
from pathlib import Path

# ================== Paths ==================
LABELS_DIR = r"C:\Users\sanaz\Downloads\train-20260817T134455Z-1-001\train\labels"
IMAGES_DIR = r"C:\Users\sanaz\Downloads\train-20260817T134455Z-1-001\train\images"
OUTPUT_DIR = r"F:\StereGeoBox\annotation"

# Create YOLO-style folders
images_out = os.path.join(OUTPUT_DIR, "images")
labels_out = os.path.join(OUTPUT_DIR, "labels")
os.makedirs(images_out, exist_ok=True)
os.makedirs(labels_out, exist_ok=True)

# ================== Step 1: Find non-empty label files ==================
label_files = []
for f in os.listdir(LABELS_DIR):
    if f.endswith(".txt"):
        path = os.path.join(LABELS_DIR, f)
        if os.path.getsize(path) > 0:
            label_files.append(f)

print(f"Found {len(label_files)} non-empty label files")

if len(label_files) < 20:
    raise ValueError(f"Only {len(label_files)} non-empty labels found.")

# ================== Step 2: Select 20 random labels ==================
selected_labels = random.sample(label_files, 20)

# ================== Step 3: Copy labels and matching images ==================
copied = 0
for label_name in selected_labels:
    # Copy label
    src_label = os.path.join(LABELS_DIR, label_name)
    dst_label = os.path.join(labels_out, label_name)
    shutil.copy2(src_label, dst_label)

    # Find and copy matching image
    base_name = Path(label_name).stem
    image_found = False
    for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
        img_name = base_name + ext
        src_img = os.path.join(IMAGES_DIR, img_name)
        if os.path.exists(src_img):
            dst_img = os.path.join(images_out, img_name)
            shutil.copy2(src_img, dst_img)
            image_found = True
            break

    if image_found:
        copied += 1
        print(f"Copied: {label_name}")
    else:
        print(f"Warning: No image found for {label_name}")

print(f"\nSuccessfully copied {copied} pairs")

# ================== Step 4: Create data.yaml ==================
yaml_content = """# YOLO dataset config for StereGeoBox sample annotations
path: .
train: images
val: images

names:
  0: fish
"""

yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
with open(yaml_path, "w", encoding="utf-8") as f:
    f.write(yaml_content)

print(f"data.yaml created at: {yaml_path}")
print("Done.")

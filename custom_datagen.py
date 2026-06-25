import os
import numpy as np
import csv

# -----------------------------
# Dataset paths
# -----------------------------
DATASET_PATH = "D:/datasetcollection"
IMAGE_DIR = os.path.join(DATASET_PATH, "images")
MASK_DIR  = os.path.join(DATASET_PATH, "masks")
CSV_FILE  = os.path.join(DATASET_PATH, "dataset_info.csv")

# -----------------------------
# Optional metadata dictionary
# Fill with your BraTS metadata if available
# Example: {"image_0.npy": {"BraTS20ID": "BraTS_000", "Age": 45, "ResectionStatus": "GTR"}}
# -----------------------------
metadata_dict = {
    # "image_0.npy": {"BraTS20ID": "BraTS_000", "Age": 45, "ResectionStatus": "GTR"},
}

# -----------------------------
# Collect image and mask filenames
# -----------------------------
image_list = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith(".npy")])
mask_list  = sorted([f for f in os.listdir(MASK_DIR) if f.endswith(".npy")])

# -----------------------------
# Create CSV
# -----------------------------
with open(CSV_FILE, mode='w', newline='') as file:
    writer = csv.writer(file)
    # CSV header
    writer.writerow([
        "Image", "Mask", "Shape", "Min", "Max", "TumorClasses", 
        "BraTS20ID", "Age", "ResectionStatus"
    ])

    # Iterate through all images
    for img_file, mask_file in zip(image_list, mask_list):
        img_path = os.path.join(IMAGE_DIR, img_file)
        mask_path = os.path.join(MASK_DIR, mask_file)

        # Load image and mask
        try:
            img = np.load(img_path)
            mask = np.load(mask_path)
        except Exception as e:
            print(f"❌ Failed to load {img_file} or {mask_file}: {e}")
            continue

        # Image stats
        shape = img.shape
        img_min, img_max = img.min(), img.max()
        tumor_classes = np.unique(np.argmax(mask, axis=-1))

        # Metadata
        meta = metadata_dict.get(img_file, {})
        braTSID = meta.get("BraTS20ID", "")
        age     = meta.get("Age", "")
        resection_status = meta.get("ResectionStatus", "")

        # Write row
        writer.writerow([
            img_file, mask_file, shape, img_min, img_max, tumor_classes.tolist(),
            braTSID, age, resection_status
        ])

print(f"✅ Dataset information saved at: {CSV_FILE}")

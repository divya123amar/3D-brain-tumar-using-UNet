import numpy as np
import nibabel as nib
import os
import glob
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# -------------------------
# Initialize scaler
# -------------------------
scaler = MinMaxScaler()

# -------------------------
# Define dataset paths
# -------------------------
TRAIN_DATA_PATH = "D:/BraTS_segmentation_Using_3D_UNet-main/BraTS2020_TrainingData"
VAL_DATA_PATH   = "D:/BraTS_segmentation_Using_3D_UNet-main/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData"
OUTPUT_PATH     = "D:/datasetcollection"

# Create folders
os.makedirs(os.path.join(OUTPUT_PATH, "train_images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "train_masks"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "validation_images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "validation_masks"), exist_ok=True)

# -------------------------
# Collect cases
# -------------------------
train_cases = sorted(glob.glob(os.path.join(TRAIN_DATA_PATH, "BraTS20_*")))
val_cases   = sorted(glob.glob(os.path.join(VAL_DATA_PATH, "BraTS20_validation_*")))

print(f"✅ Total training cases found: {len(train_cases)}")
print(f"✅ Total validation cases found: {len(val_cases)}")

# -------------------------
# Function to process a single case
# -------------------------
def process_case(case_path, save_img_path, save_mask_path, idx, is_validation=False):
    """Load, normalize, crop, save npy for one case"""
    
    # Load image modalities
    t2_file    = glob.glob(os.path.join(case_path, "*_t2.nii*"))[0]
    t1ce_file  = glob.glob(os.path.join(case_path, "*_t1ce.nii*"))[0]
    flair_file = glob.glob(os.path.join(case_path, "*_flair.nii*"))[0]

    t2    = scaler.fit_transform(nib.load(t2_file).get_fdata().reshape(-1,1)).reshape(nib.load(t2_file).shape)
    t1ce  = scaler.fit_transform(nib.load(t1ce_file).get_fdata().reshape(-1,1)).reshape(nib.load(t1ce_file).shape)
    flair = scaler.fit_transform(nib.load(flair_file).get_fdata().reshape(-1,1)).reshape(nib.load(flair_file).shape)

    # Stack channels
    combined_images = np.stack([flair, t1ce, t2], axis=3)
    combined_images = combined_images[56:184, 56:184, 13:141, :]  # crop center

    # Save images
    np.save(os.path.join(save_img_path, f"image_{idx}.npy"), combined_images)

    # Handle masks
    if not is_validation:
        mask_file = glob.glob(os.path.join(case_path, "*_seg.nii*"))[0]
        mask = nib.load(mask_file).get_fdata().astype(np.uint8)
        mask[mask == 4] = 3  # merge label 4 into 3
        mask = mask[56:184, 56:184, 13:141]
        mask_cat = to_categorical(mask, num_classes=4)
        np.save(os.path.join(save_mask_path, f"mask_{idx}.npy"), mask_cat)
    else:
        # Save placeholder mask (zeros) for validation
        mask_placeholder = np.zeros((128, 128, 128, 4), dtype=np.uint8)
        np.save(os.path.join(save_mask_path, f"mask_{idx}.npy"), mask_placeholder)

    print(f"✅ Saved case {idx}: {'validation' if is_validation else 'train'}")

# -------------------------
# Process training cases
# -------------------------
print("Preparing training dataset...")
for i, case in enumerate(train_cases):
    process_case(case, 
                 os.path.join(OUTPUT_PATH, "train_images"), 
                 os.path.join(OUTPUT_PATH, "train_masks"), 
                 i, 
                 is_validation=False)

# -------------------------
# Process validation cases
# -------------------------
print("Preparing validation dataset...")
for i, case in enumerate(val_cases):
    process_case(case, 
                 os.path.join(OUTPUT_PATH, "validation_images"), 
                 os.path.join(OUTPUT_PATH, "validation_masks"), 
                 i, 
                 is_validation=True)

print("🎉 All data saved in:", OUTPUT_PATH)

# -------------------------
# Optional: Create CSV file
# -------------------------
import csv

csv_file = os.path.join(OUTPUT_PATH, "dataset.csv")
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["CaseID", "ImagePath", "MaskPath", "Age", "ResectionStatus"])  # columns

    # Add train cases
    for i in range(len(train_cases)):
        writer.writerow([f"train_{i}", f"train_images/image_{i}.npy", f"train_masks/mask_{i}.npy", "", ""])
    # Add validation cases
    for i in range(len(val_cases)):
        writer.writerow([f"val_{i}", f"validation_images/image_{i}.npy", f"validation_masks/mask_{i}.npy", "", ""])

print(f"✅ CSV saved at {csv_file}")

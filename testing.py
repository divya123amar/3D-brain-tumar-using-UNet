import os
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------
# SAME SETTINGS AS TRAINING
# --------------------------
VAL_IMG_DIR   = r"D:/datasetcollection/validation_images"
VAL_MSK_DIR   = r"D:/datasetcollection/validation_masks"
IMG_SHAPE     = (128, 128, 128, 3)
N_CLASSES     = 4
SEED          = 42
BATCH_SIZE    = 1

# --------------------------
# Custom functions (same as training)
# --------------------------
def dice_coef(y_true, y_pred, epsilon=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
    axes = tuple(range(1, len(y_pred.shape)-1))
    intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
    denom = tf.reduce_sum(y_true + y_pred, axis=axes)
    dice = (2.0 * intersection + epsilon) / (denom + epsilon)
    return tf.reduce_mean(dice)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def iou_score(y_true, y_pred, epsilon=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
    axes = tuple(range(1, len(y_pred.shape)-1))
    intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
    union = tf.reduce_sum(y_true + y_pred - y_true * y_pred, axis=axes)
    iou = (intersection + epsilon) / (union + epsilon)
    return tf.reduce_mean(iou)

def combo_loss(y_true, y_pred, alpha=0.5):
    ce = tf.keras.losses.CategoricalCrossentropy()(y_true, y_pred)
    dl = dice_loss(y_true, y_pred)
    return alpha * ce + (1.0 - alpha) * dl

# --------------------------
# Reload validation dataset
# --------------------------
def list_pairs(img_dir, msk_dir):
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".npy")])
    msk_files = sorted([f for f in os.listdir(msk_dir) if f.endswith(".npy")])
    assert len(img_files) == len(msk_files), \
        f"Image/Mask mismatch: {len(img_files)} vs {len(msk_files)}"
    return [os.path.join(img_dir, f) for f in img_files], \
           [os.path.join(msk_dir, f) for f in msk_files]

val_imgs, val_msks = list_pairs(VAL_IMG_DIR, VAL_MSK_DIR)

def load_pair(img_path, msk_path):
    def _load(img_p, msk_p):
        img = np.load(img_p.decode("utf-8")).astype(np.float32)
        msk = np.load(msk_p.decode("utf-8"))
        img = img / (np.max(img) + 1e-6)
        if msk.ndim == 3:
            msk_oh = tf.one_hot(msk.astype(np.int32), depth=N_CLASSES).numpy().astype(np.float32)
        else:
            msk_oh = msk.astype(np.float32)
        return img, msk_oh
    img, msk = tf.numpy_function(_load, [img_path, msk_path], [tf.float32, tf.float32])
    img.set_shape(IMG_SHAPE)
    msk.set_shape((*IMG_SHAPE[:3], N_CLASSES))
    return img, msk

def make_dataset(img_paths, msk_paths, training=False):
    ds = tf.data.Dataset.from_tensor_slices((img_paths, msk_paths))
    ds = ds.map(load_pair, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

val_ds = make_dataset(val_imgs, val_msks, training=False)

# --------------------------
# Load trained model
# --------------------------
model = tf.keras.models.load_model(
    "best_model.keras",
    custom_objects={"combo_loss": combo_loss,
                    "iou_score": iou_score,
                    "dice_coef": dice_coef}
)

# --------------------------
# Save combined predictions
# --------------------------
out_dir = "predictions_combined"
os.makedirs(out_dir, exist_ok=True)

def predict_and_save_combined(dataset, out_dir, max_samples=5):
    for idx, (img, mask) in enumerate(dataset.take(max_samples)):
        pred = model.predict(img, verbose=0)
        pred_mask = np.argmax(pred[0], axis=-1)
        true_mask = np.argmax(mask[0].numpy(), axis=-1)
        mid_slice = pred_mask.shape[2] // 2

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1); plt.imshow(img[0, :, :, mid_slice, 0], cmap="gray"); plt.title("Testing Image"); plt.axis("off")
        plt.subplot(1, 3, 2); plt.imshow(true_mask[:, :, mid_slice], cmap="viridis"); plt.title("Testing Label"); plt.axis("off")
        plt.subplot(1, 3, 3); plt.imshow(pred_mask[:, :, mid_slice], cmap="viridis"); plt.title("Prediction"); plt.axis("off")

        save_path = os.path.join(out_dir, f"test_result_{idx}.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved: {save_path}")

predict_and_save_combined(val_ds, out_dir, max_samples=5)
print("All done!")
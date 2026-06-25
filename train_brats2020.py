import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"       # quieter TF logs

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

# --------------------------
# Fixed dataset paths
# --------------------------
TRAIN_IMG_DIR = r"D:/datasetcollection/images"
TRAIN_MSK_DIR = r"D:/datasetcollection/masks"
VAL_IMG_DIR   = r"D:/datasetcollection/validation_images"
VAL_MSK_DIR   = r"D:/datasetcollection/validation_masks"

# --------------------------
# Training config
# --------------------------
IMG_SHAPE   = (128, 128, 128, 3)  # H, W, D, C
N_CLASSES   = 4
BATCH_SIZE  = 1                   # 3D volumes are large; 1–2 typical
EPOCHS      = 10
LR          = 1e-4
SEED        = 42

rng = np.random.default_rng(SEED)
tf.random.set_seed(SEED)

# --------------------------
# Metrics & loss
# --------------------------
def one_hot(mask, n_classes=N_CLASSES):
    if mask.ndim == 4 and mask.shape[-1] == n_classes:
        return mask
    mask = mask.astype(np.int32)
    return tf.one_hot(mask, depth=n_classes, dtype=tf.float32)

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
# 3D U-Net model
# --------------------------
def conv_block(x, filters):
    x = layers.Conv3D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv3D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x

def encoder_block(x, filters):
    c = conv_block(x, filters)
    p = layers.MaxPooling3D((2, 2, 2))(c)
    return c, p

def decoder_block(x, skip, filters):
    x = layers.Conv3DTranspose(filters, 2, strides=2, padding="same")(x)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, filters)
    return x

def build_unet_3d(input_shape=IMG_SHAPE, n_classes=N_CLASSES):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    s1, p1 = encoder_block(inputs, 32)
    s2, p2 = encoder_block(p1, 64)
    s3, p3 = encoder_block(p2, 128)
    s4, p4 = encoder_block(p3, 256)

    # Bottleneck
    b = conv_block(p4, 512)

    # Decoder
    d1 = decoder_block(b, s4, 256)
    d2 = decoder_block(d1, s3, 128)
    d3 = decoder_block(d2, s2, 64)
    d4 = decoder_block(d3, s1, 32)

    outputs = layers.Conv3D(n_classes, 1, activation="softmax")(d4)
    return keras.Model(inputs, outputs, name="unet3d")

model = build_unet_3d()
model.compile(optimizer=keras.optimizers.Adam(LR),
              loss=combo_loss,
              metrics=["accuracy", iou_score, dice_coef])
model.summary()

# --------------------------
# Data loading
# --------------------------
def list_pairs(img_dir, msk_dir):
    img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(".npy")])
    msk_files = sorted([f for f in os.listdir(msk_dir) if f.lower().endswith(".npy")])
    assert len(img_files) == len(msk_files), \
        f"Image/Mask count mismatch: {len(img_files)} vs {len(msk_files)}"
    return [os.path.join(img_dir, f) for f in img_files], \
           [os.path.join(msk_dir, f) for f in msk_files]

train_imgs, train_msks = list_pairs(TRAIN_IMG_DIR, TRAIN_MSK_DIR)
val_imgs,   val_msks   = list_pairs(VAL_IMG_DIR, VAL_MSK_DIR)

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

def make_dataset(img_paths, msk_paths, training=True):
    ds = tf.data.Dataset.from_tensor_slices((img_paths, msk_paths))
    if training:
        ds = ds.shuffle(buffer_size=len(img_paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(load_pair, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_dataset(train_imgs, train_msks, training=True)
val_ds   = make_dataset(val_imgs, val_msks, training=False)

# --------------------------
# Callbacks
# --------------------------
ckpt_best = ModelCheckpoint("best_model.keras", monitor="val_loss",
                            save_best_only=True, verbose=1)
csv_logger = CSVLogger("training_log.csv", append=False)

# --------------------------
# Train
# --------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[ckpt_best, csv_logger]
)

model.save("final_model.keras")
print("✅ Saved best_model.keras, final_model.keras, and training_log.csv with accuracy/IoU/Dice")


try:
    import matplotlib.pyplot as plt
    epochs_range = range(1, len(history.history["loss"]) + 1)

    plt.figure()
    plt.plot(epochs_range, history.history["accuracy"], label="train acc")
    plt.plot(epochs_range, history.history["val_accuracy"], label="val acc")
    plt.title("Training & Validation Accuracy"); plt.legend(); plt.show()
except:
    pass

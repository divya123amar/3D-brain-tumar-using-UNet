import matplotlib.pyplot as plt

# ==============================
# DATA PROVIDED
# ==============================

epochs1 = list(range(10))
acc1 = [0.957828045,0.978434682,0.98112154,0.98197943,0.982960939,
        0.984159291,0.984904528,0.985195518,0.984741271,0.985770524]

dice1 = [0.306986243,0.410570353,0.495357692,0.563765943,0.620134115,
         0.663705409,0.691371202,0.71014899,0.720858037,0.732975841]

iou1 = [0.23653394,0.329287738,0.403774053,0.466308683,0.5202775,
        0.563108563,0.591708541,0.611886203,0.623331904,0.637552381]

loss1 = [0.55119282,0.385371745,0.307289869,0.258939922,0.224328578,
         0.198109537,0.182174295,0.171766922,0.167526871,0.159408033]

# ==============================
# ONE FIGURE PLOT
# ==============================

plt.figure(figsize=(12, 7))

plt.plot(epochs1, acc1, marker='o', label="Accuracy")
plt.plot(epochs1, dice1, marker='o', label="Dice Coefficient")
plt.plot(epochs1, iou1, marker='o', label="IoU Score")
plt.plot(epochs1, loss1, marker='o', label="Loss")

plt.xlabel("Epochs")
plt.ylabel("Metric Value")
plt.title("Training Metrics Across Epochs (All in One Plot)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Piece 1: open ONE Sen1Floods11 chip and look at it.
# Run from the repo root:  python 01_look_at_chip.py

import numpy as np
import rasterio
import matplotlib.pyplot as plt

# --- 1. Which chip do we open? ---------------------------------------
# Every chip has a name like "Country_number".
# The SAR image and its label share that name.
chip_name = "Ghana_103272"
s1_path = "data/sen1floods11_v1.1/data/S1GRDHand/" + chip_name + "_S1Hand.tif"
label_path = "data/sen1floods11_v1.1/data/LabelHand/" + chip_name + "_LabelHand.tif"

# --- 2. Read the files into numpy arrays -----------------------------
# The SAR file has 2 bands: band 1 = VV, band 2 = VH.
with rasterio.open(s1_path) as f:
    s1 = f.read()          # shape: (2, 512, 512)

# The label file has 1 band. read(1) gives a flat 2D array.
with rasterio.open(label_path) as f:
    label = f.read(1)      # shape: (512, 512)

vv = s1[0]
vh = s1[1]

# --- 3. Print the facts we need for later pieces ---------------------
print("SAR shape:  ", s1.shape, s1.dtype)
print("Label shape:", label.shape, label.dtype)

# SAR values are in decibels (dB). Some pixels can be NaN (missing),
# so we use nanmin / nanmax, which skip NaN values.
print("VV min / max:", np.nanmin(vv), np.nanmax(vv))
print("VH min / max:", np.nanmin(vh), np.nanmax(vh))
print("NaN pixels in SAR:", np.isnan(s1).sum())

# Label values: -1 = no data, 0 = not water, 1 = water.
values, counts = np.unique(label, return_counts=True)
print("Label values:", values)
print("Pixel counts:", counts)

# --- 4. Show the three images side by side ---------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(vv, cmap="gray")
axes[0].set_title("VV (dB)")

axes[1].imshow(vh, cmap="gray")
axes[1].set_title("VH (dB)")

axes[2].imshow(label, cmap="viridis", vmin=-1, vmax=1)
axes[2].set_title("Label: -1 no data, 0 dry, 1 water")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.savefig("chip_preview.png")   # saved so you can put it in the README later
plt.show()

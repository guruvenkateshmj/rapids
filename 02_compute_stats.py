# Piece 2a: compute dataset statistics from the TRAIN split only.
# Run from the repo root:  python 02_compute_stats.py
#
# Output: stats.json, which dataset.py reads later. We compute:
#   - per-band mean and std of the SAR values (for normalisation)
#   - how many SAR pixels are NaN (missing)
#   - how many label pixels are -1 / 0 / 1 (no data / dry / water)
#
# Only the train split is used, so the test set stays truly unseen.

import json
import time
import numpy as np
import rasterio

ROOT = "data/sen1floods11_v1.1"
SPLIT_FILE = ROOT + "/splits/flood_train_data.txt"
S1_DIR = ROOT + "/data/S1GRDHand/"
LABEL_DIR = ROOT + "/data/LabelHand/"

# --- 1. Read the list of training chips -------------------------------
with open(SPLIT_FILE) as f:
    chip_names = [line.strip() for line in f if line.strip()]
print("Training chips:", len(chip_names))

# --- 2. Running totals -------------------------------------------------
# We never keep all images in memory. Instead we add each chip's numbers
# into these running totals, then divide at the end.
# float64 (not float32) so the big sums stay precise.
band_sum = np.zeros(2, dtype=np.float64)       # sum of x,  per band
band_sum_sq = np.zeros(2, dtype=np.float64)    # sum of x², per band
band_count = np.zeros(2, dtype=np.int64)       # valid (non-NaN) pixels
nan_count = 0
label_counts = np.zeros(3, dtype=np.int64)     # index 0=-1, 1=0, 2=1

start = time.time()
for i, name in enumerate(chip_names):
    with rasterio.open(S1_DIR + name + "_S1Hand.tif") as f:
        s1 = f.read()                          # (2, 512, 512) float32
    with rasterio.open(LABEL_DIR + name + "_LabelHand.tif") as f:
        label = f.read(1)                      # (512, 512) int16

    # NaN-aware sums: nansum skips missing pixels instead of poisoning
    # the total. axis=(1, 2) collapses height and width, keeping bands.
    s1 = s1.astype(np.float64)
    nan_mask = np.isnan(s1)
    nan_count += int(nan_mask.sum())
    band_count += (~nan_mask).sum(axis=(1, 2))
    band_sum += np.nansum(s1, axis=(1, 2))
    band_sum_sq += np.nansum(s1 * s1, axis=(1, 2))

    # bincount is a fast histogram. Labels are -1/0/1, so we add 1
    # to make them 0/1/2, which bincount needs. minlength=3 keeps the
    # shape fixed even if a chip has no water at all.
    label_counts += np.bincount(label.ravel() + 1, minlength=3)

    if (i + 1) % 50 == 0:
        print(f"  {i + 1}/{len(chip_names)} chips done")

# --- 3. Turn totals into statistics -----------------------------------
# mean = sum / n          std = sqrt(mean of x² - mean²)
mean = band_sum / band_count
std = np.sqrt(band_sum_sq / band_count - mean ** 2)

total_label_pixels = label_counts.sum()
percent = 100.0 * label_counts / total_label_pixels

print(f"\nDone in {time.time() - start:.1f} s")
print(f"VV  mean {mean[0]:8.3f}  std {std[0]:6.3f}")
print(f"VH  mean {mean[1]:8.3f}  std {std[1]:6.3f}")
print(f"NaN SAR pixels: {nan_count}  "
      f"({100.0 * nan_count / (2 * total_label_pixels):.4f} %)")
print(f"Label -1 (no data): {label_counts[0]:>12,}  {percent[0]:6.2f} %")
print(f"Label  0 (dry):     {label_counts[1]:>12,}  {percent[1]:6.2f} %")
print(f"Label  1 (water):   {label_counts[2]:>12,}  {percent[2]:6.2f} %")

# --- 4. Save for dataset.py ------------------------------------------
stats = {
    "mean": mean.tolist(),
    "std": std.tolist(),
    "nan_pixels": nan_count,
    "label_counts": {"no_data": int(label_counts[0]),
                     "dry": int(label_counts[1]),
                     "water": int(label_counts[2])},
    "n_train_chips": len(chip_names),
}
with open("stats.json", "w") as f:
    json.dump(stats, f, indent=2)
print("\nSaved stats.json")

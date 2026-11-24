# Piece 2b: where are the NaN (missing radar) pixels, and what are they labelled?
# Run from the repo root:  python 02b_check_nan.py
#
# The question we want answered:
#   Are NaN pixels already labelled -1 (no data)?
#   Or do many of them carry a real label (0 = dry, 1 = water)?
# That decides option A (fill only) vs option B (fill + ignore) in dataset.py.

import numpy as np
import rasterio

ROOT = "data/sen1floods11_v1.1"
SPLIT_FILE = ROOT + "/splits/flood_train_data.txt"
S1_DIR = ROOT + "/data/S1GRDHand/"
LABEL_DIR = ROOT + "/data/LabelHand/"
CHIP_PIXELS = 512 * 512

with open(SPLIT_FILE) as f:
    chip_names = [line.strip() for line in f if line.strip()]

# Label counts for NaN pixels only. Index 0 = -1, 1 = dry, 2 = water.
nan_label_counts = np.zeros(3, dtype=np.int64)
chips_with_nan = 0
per_chip = []   # (chip name, NaN pixels with a real label)

for name in chip_names:
    with rasterio.open(S1_DIR + name + "_S1Hand.tif") as f:
        s1 = f.read()                      # (2, 512, 512)
    with rasterio.open(LABEL_DIR + name + "_LabelHand.tif") as f:
        label = f.read(1)                  # (512, 512)

    # A pixel counts as NaN if EITHER band is missing.
    # .any(axis=0) squashes the 2 bands into one (512, 512) True/False map.
    nan_map = np.isnan(s1).any(axis=0)
    n_nan = int(nan_map.sum())
    if n_nan == 0:
        continue
    chips_with_nan += 1

    # label[nan_map] keeps only the labels sitting under NaN pixels.
    counts = np.bincount(label[nan_map] + 1, minlength=3)
    nan_label_counts += counts

    real_labels = int(counts[1] + counts[2])   # dry + water
    if real_labels > 0:
        per_chip.append((name, real_labels))

total_nan = nan_label_counts.sum()
pct = 100.0 * nan_label_counts / max(total_nan, 1)

print(f"Chips with any NaN:   {chips_with_nan} / {len(chip_names)}")
print(f"NaN pixels (either band): {total_nan:,}")
print(f"  labelled -1 (no data): {nan_label_counts[0]:>12,}  {pct[0]:6.2f} %")
print(f"  labelled  0 (dry):     {nan_label_counts[1]:>12,}  {pct[1]:6.2f} %")
print(f"  labelled  1 (water):   {nan_label_counts[2]:>12,}  {pct[2]:6.2f} %")

# Small holes or big blocks? If one chip has thousands of labelled NaN
# pixels, that's a block. If many chips have a handful each, those are holes.
print(f"\nChips where NaN pixels carry a real label: {len(per_chip)}")
per_chip.sort(key=lambda item: item[1], reverse=True)
print("Top 5 (chip, labelled NaN pixels, % of chip):")
for name, n in per_chip[:5]:
    print(f"  {name:<20} {n:>8,}  {100.0 * n / CHIP_PIXELS:6.2f} %")

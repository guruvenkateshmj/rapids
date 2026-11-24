# Piece 2c: the dataset class.
# Given a split name ("train", "valid", "test", "bolivia"), it hands back
# one (image, label) pair at a time, ready for PyTorch:
#   image: float32 tensor, shape (2, H, W), normalised, no NaN
#   label: int64 tensor,   shape (H, W),    values -1 / 0 / 1
#
# Test it on its own:  python dataset.py

import json
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

ROOT = "data/sen1floods11_v1.1"
S1_DIR = ROOT + "/data/S1GRDHand/"
LABEL_DIR = ROOT + "/data/LabelHand/"
SPLIT_FILES = {
    "train": ROOT + "/splits/flood_train_data.txt",
    "valid": ROOT + "/splits/flood_valid_data.txt",
    "test": ROOT + "/splits/flood_test_data.txt",
    "bolivia": ROOT + "/splits/flood_bolivia_data.txt",
}


class Sen1Floods11Dataset(Dataset):

    def __init__(self, split, crop_size=None, stats_file="stats.json"):
        # crop_size=256 for training (random crops).
        # crop_size=None for valid/test (full 512 x 512 chips).
        with open(SPLIT_FILES[split]) as f:
            self.chip_names = [line.strip() for line in f if line.strip()]
        self.crop_size = crop_size

        # Mean and std from the TRAIN split (made by 02_compute_stats.py).
        # Shape (2, 1, 1) so numpy can subtract them from a (2, H, W) image:
        # each band gets its own number, applied to every pixel.
        with open(stats_file) as f:
            stats = json.load(f)
        self.mean = np.array(stats["mean"], dtype=np.float32).reshape(2, 1, 1)
        self.std = np.array(stats["std"], dtype=np.float32).reshape(2, 1, 1)

    def __len__(self):
        # PyTorch asks this to know how many samples exist.
        return len(self.chip_names)

    def __getitem__(self, index):
        # PyTorch asks this for sample number `index`.
        name = self.chip_names[index]
        with rasterio.open(S1_DIR + name + "_S1Hand.tif") as f:
            image = f.read().astype(np.float32)        # (2, 512, 512)
        with rasterio.open(LABEL_DIR + name + "_LabelHand.tif") as f:
            label = f.read(1).astype(np.int64)         # (512, 512)

        # 1. Remember where the radar had no data (either band missing).
        nan_map = np.isnan(image).any(axis=0)          # (512, 512) True/False

        # 2. Normalise each band: most values end up between about -2 and +2.
        image = (image - self.mean) / self.std

        # 3. Option B: fill NaN with 0 (= the band mean after normalising)
        #    and tell the loss to ignore those pixels (label -1).
        image[:, nan_map] = 0.0
        label[nan_map] = -1

        # 4. Random crop for training. A new random spot every time this
        #    chip is used, so each epoch sees slightly different data.
        if self.crop_size is not None:
            size = self.crop_size
            top = np.random.randint(0, image.shape[1] - size + 1)
            left = np.random.randint(0, image.shape[2] - size + 1)
            image = image[:, top:top + size, left:left + size]
            label = label[top:top + size, left:left + size]

        # 5. numpy array -> PyTorch tensor.
        # np.ascontiguousarray makes a clean copy of the cropped slice in
        # memory, which torch.from_numpy needs.
        image = torch.from_numpy(np.ascontiguousarray(image))
        label = torch.from_numpy(np.ascontiguousarray(label))
        return image, label


# --- Self-test: runs only with `python dataset.py`, not on import ------
if __name__ == "__main__":
    import time
    from torch.utils.data import DataLoader

    train_set = Sen1Floods11Dataset("train", crop_size=256)
    valid_set = Sen1Floods11Dataset("valid", crop_size=None)
    print("Train chips:", len(train_set))
    print("Valid chips:", len(valid_set))

    image, label = train_set[0]
    print("\nOne train sample")
    print("  image:", tuple(image.shape), image.dtype)
    print("  label:", tuple(label.shape), label.dtype)
    print(f"  image min / max: {image.min():.2f} / {image.max():.2f}")
    print("  any NaN left?  ", bool(torch.isnan(image).any()))
    print("  label values:  ", torch.unique(label).tolist())

    image, label = valid_set[0]
    print("\nOne valid sample (full size)")
    print("  image:", tuple(image.shape), "  label:", tuple(label.shape))

    # Batches: this is exactly how training will pull data.
    loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=0)
    start = time.time()
    for images, labels in loader:
        pass
    print(f"\nOne full pass over train in batches of 8: {time.time() - start:.1f} s")
    print("  last batch image:", tuple(images.shape))
    print("  last batch label:", tuple(labels.shape))

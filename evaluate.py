# Piece 5: evaluation.
# Run from the repo root:  python evaluate.py | tee results.txt
#
# Loads each trained checkpoint and measures it, WITHOUT any training, on:
#   - valid   : sanity check, must match the best IoU from training
#   - test    : the honest, reportable number (touched once)
#   - bolivia : a country the model never saw
# For each: water IoU, precision, recall. Then mean +/- std over the seeds.
#
# YOUR PART: fill the 4 TODOs inside evaluate(). Everything else is ready.
# Self-check: the "valid" IoU of each checkpoint must match (within ~0.001)
# the best val IoU that training printed for that run. If it doesn't,
# one of your TODOs is wrong.

import statistics
import torch
from torch.utils.data import DataLoader

from dataset import Sen1Floods11Dataset
from unet_model import UNet

# --- Settings ----------------------------------------------------------
# Check the exact names with:  ls checkpoints
CHECKPOINTS = [
    "checkpoints/run1_base32_best.pt",          # seed 42
    "checkpoints/run2_base32_seed1_best.pt",    # seed 1
    "checkpoints/run2_base32_seed2_best.pt",    # seed 2
]
BASE = 32
SPLITS = ["valid", "test", "bolivia"]
RESULTS_FILE = "results.csv"

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print("Device:", device)


def evaluate(model, split):
    # Full-size chips, no cropping, fixed order.
    data = Sen1Floods11Dataset(split, crop_size=None)
    loader = DataLoader(data, batch_size=4, shuffle=False, num_workers=0)

    # TODO 1 (1 line): put the model in grading mode.
    model.eval()

    tp = fp = fn = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)

            # TODO 2 (1 line): turn the 2 scores per pixel into 0 (dry) or 1 (water).
            preds = logits.argmax(dim=1)

            # TODO 3 (3 lines): add this batch's counts to tp, fp and fn.
            tp += ((preds == 1) & (labels == 1)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()

    # TODO 4 (3 lines): compute iou, precision and recall from tp, fp, fn.
    iou = tp / max(tp + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    return iou, precision, recall, len(data)


# --- Run every checkpoint on every split -----------------------------
results = {split: [] for split in SPLITS}
with open(RESULTS_FILE, "w") as f:
    f.write("checkpoint,split,chips,iou,precision,recall\n")
    for path in CHECKPOINTS:
        model = UNet(base=BASE).to(device)
        # map_location: load the weights straight onto our device.
        model.load_state_dict(torch.load(path, map_location=device))
        print("\n" + path)
        for split in SPLITS:
            iou, precision, recall, n = evaluate(model, split)
            results[split].append((iou, precision, recall))
            print(f"  {split:8s} ({n:3d} chips)  IoU {iou:.4f}  "
                  f"precision {precision:.4f}  recall {recall:.4f}")
            f.write(f"{path},{split},{n},{iou:.6f},{precision:.6f},{recall:.6f}\n")

# --- Summary: mean +/- std over the seeds ----------------------------
print(f"\nMean +/- std over {len(CHECKPOINTS)} seeds")
for split in SPLITS:
    line = f"  {split:8s}"
    for i, name in enumerate(["IoU", "precision", "recall"]):
        values = [r[i] for r in results[split]]
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        line += f"  {name} {statistics.mean(values):.4f} +/- {std:.4f}"
    print(line)
print(f"\nSaved {RESULTS_FILE}")